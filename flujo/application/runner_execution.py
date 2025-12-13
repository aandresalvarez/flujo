from __future__ import annotations

from datetime import datetime
from typing import Literal, TYPE_CHECKING, TypeVar

from ..exceptions import ContextMergeError, PipelineAbortSignal, PausedException
from ..domain.models import PipelineContext, PipelineResult
from ..domain.commands import AgentCommand
from ..state import WorkflowState
from ..infra import telemetry
from .runner_components import ReplayExecutor, ResumeOrchestrator
from .core.execution_manager import ExecutionManager
from .core.state_manager import StateManager

if TYPE_CHECKING:  # pragma: no cover
    from .runner import Flujo
    from pydantic import TypeAdapter

_CtxT = TypeVar("_CtxT", bound=PipelineContext)
_InT = TypeVar("_InT")
_OutT = TypeVar("_OutT")


async def resume_async_inner(
    runner: "Flujo[_InT, _OutT, _CtxT]",
    paused_result: PipelineResult[_CtxT],
    human_input: object,
    agent_command_adapter: "TypeAdapter[AgentCommand]",
) -> PipelineResult[_CtxT]:
    """Resume a paused pipeline with human input."""
    paused_during_resume = False
    try:
        runner._ensure_pipeline()
        assert runner.pipeline is not None

        orchestrator: ResumeOrchestrator[_CtxT] = ResumeOrchestrator(
            runner.pipeline,
            trace_manager=runner._trace_manager,
            agent_command_adapter=agent_command_adapter,
        )

        ctx = orchestrator.validate_resume(paused_result)
        pause_msg = getattr(ctx, "pause_message", None)

        start_idx, paused_step = orchestrator.resolve_paused_step(paused_result, ctx, human_input)
        if getattr(ctx, "status", None) == "paused":
            try:
                if hasattr(ctx, "hitl_data"):
                    ctx.hitl_data = {}
                if hasattr(ctx, "user_input"):
                    ctx.user_input = human_input
            except Exception:
                pass
        human_input = orchestrator.coerce_human_input(paused_step, human_input)

        paused_step_result = orchestrator.build_step_result(paused_step, human_input)

        if isinstance(ctx, PipelineContext):
            orchestrator.record_hitl_interaction(ctx, human_input, pause_msg)
            orchestrator.update_conversation_history(ctx, human_input, pause_msg)
            orchestrator.apply_sink_to(ctx, paused_step, human_input)
            orchestrator.update_steps_map(ctx, paused_step, human_input)
            orchestrator.record_pending_command_log(ctx, paused_step, human_input)

        orchestrator.add_trace_event(human_input)

        from ..domain.dsl.step import HumanInTheLoopStep as _HITL

        if isinstance(paused_step, _HITL):
            if paused_result.step_history:
                last = paused_result.step_history[-1]
                if last.name == paused_step.name and not last.success:
                    paused_result.step_history[-1] = paused_step_result
                else:
                    paused_result.step_history.append(paused_step_result)
            else:
                paused_result.step_history.append(paused_step_result)
            try:
                if getattr(paused_step, "updates_context", False) and isinstance(
                    ctx, PipelineContext
                ):
                    from .core.context_adapter import _build_context_update, _inject_context

                    update_data = _build_context_update(human_input)
                    if update_data:
                        validation_error = _inject_context(ctx, update_data, type(ctx))
                        if validation_error:
                            raise ContextMergeError(
                                f"Failed to merge human input into context: {validation_error}"
                            )
            except Exception as _merge_err:
                try:
                    telemetry.logfire.warning(
                        f"Resume context merge warning for step '{paused_step.name}': {_merge_err}"
                    )
                except Exception:
                    pass
            try:
                await runner._dispatch_hook(
                    "post_step",
                    step_result=paused_step_result,
                    context=ctx,
                    resources=runner.resources,
                )
            except Exception:
                pass
            data = human_input
            resume_start_idx = start_idx + 1
        else:
            data = human_input
            resume_start_idx = start_idx

        # For conditional branches that will hit a HITL immediately, mark the context so the next
        # HITL can auto-consume this resume input. Avoid applying this to loop steps, which handle
        # their own resume semantics.
        try:
            from ..domain.dsl.conditional import ConditionalStep as _Cond

            if isinstance(paused_step, _Cond):
                if hasattr(ctx, "loop_resume_requires_hitl_output"):
                    ctx.loop_resume_requires_hitl_output = True
                if hasattr(ctx, "status"):
                    ctx.status = "paused"
                if hasattr(ctx, "user_input"):
                    ctx.user_input = ctx.user_input or human_input
                if hasattr(ctx, "hitl_data"):
                    ctx.hitl_data = {"human_response": human_input}
        except Exception:
            pass

        # If we are resuming into a pipeline with remaining HITL steps and only one new
        # human_input was provided, ensure we resume at the next step (start_idx + 1)
        # so subsequent HITLs can execute and pause naturally.
        try:
            from ..domain.dsl.step import HumanInTheLoopStep as _H

            if isinstance(ctx, PipelineContext):
                remaining_steps = (
                    runner.pipeline.steps[resume_start_idx:] if runner.pipeline else []
                )

                # Scan remaining steps (including conditional branches) for the next HITL
                def _find_hitl(steps: list[object]) -> _H | None:
                    for st in steps:
                        if isinstance(st, _H):
                            return st
                        try:
                            from flujo.domain.dsl.conditional import ConditionalStep

                            if isinstance(st, ConditionalStep):
                                for _branch in (st.branches or {}).values():
                                    if hasattr(_branch, "steps"):
                                        found = _find_hitl(
                                            list(getattr(_branch, "steps", []) or [])
                                        )
                                        if found:
                                            return found
                        except Exception:
                            continue
                    return None

                _find_hitl(list(remaining_steps))
                # Intentionally no early return: let the next HITL execute and pause naturally.
        except Exception:
            pass

        run_id_for_state = getattr(ctx, "run_id", None)
        state_created_at: datetime | None = None
        if runner.state_backend is not None and run_id_for_state is not None:
            loaded = await runner.state_backend.load_state(run_id_for_state)
            if loaded is not None:
                wf_state_loaded = WorkflowState.model_validate(loaded)
                state_created_at = wf_state_loaded.created_at
        try:
            async for chunk in runner._execute_steps(
                resume_start_idx,
                data,
                ctx,
                paused_result,
                stream_last=False,
                run_id=run_id_for_state,
                state_backend=runner.state_backend,
                state_created_at=state_created_at,
            ):
                # ExecutionManager yields the PipelineResult on pause without raising;
                # detect that case so we do not mark the run as completed.
                if isinstance(chunk, PipelineResult):
                    try:
                        chunk_ctx = chunk.final_pipeline_context
                        if (
                            isinstance(chunk_ctx, PipelineContext)
                            and getattr(chunk_ctx, "status", None) == "paused"
                        ):
                            paused_during_resume = True
                    except Exception:
                        pass
        except (PipelineAbortSignal, PausedException):
            if isinstance(ctx, PipelineContext):
                ctx.status = "paused"
            paused_during_resume = True

        expected_len = len(runner.pipeline.steps) if runner.pipeline is not None else 0
        history_len = len(paused_result.step_history)
        final_status: Literal["running", "paused", "completed", "failed"]
        if paused_during_resume or getattr(ctx, "status", None) == "paused":
            final_status = "paused"
        elif expected_len and history_len < expected_len:
            # Resume should stay paused until the pipeline covers all steps.
            final_status = "paused"
        elif paused_result.step_history:
            final_status = (
                "completed" if all(s.success for s in paused_result.step_history) else "failed"
            )
        else:
            # If no history exists, treat empty pipelines as completed; otherwise stay paused.
            final_status = "completed" if expected_len == 0 else "paused"

        if isinstance(ctx, PipelineContext):
            try:
                ctx.status = final_status
            except Exception:
                pass

        state_manager: StateManager[PipelineContext] = StateManager[PipelineContext](
            runner.state_backend
        )
        assert runner.pipeline is not None
        execution_manager: ExecutionManager[PipelineContext] = ExecutionManager[PipelineContext](
            runner.pipeline,
            state_manager=state_manager,
        )
        await execution_manager.persist_final_state(
            run_id=run_id_for_state,
            context=ctx,
            result=paused_result,
            # Use the total step history length as the offset so persisted indices reflect prior executions.
            start_idx=len(paused_result.step_history),
            state_created_at=state_created_at,
            final_status=final_status,
        )

        try:
            all_steps_succeeded = all(s.success for s in paused_result.step_history)
            if final_status == "paused":
                paused_result.success = True
            else:
                paused_result.success = (
                    final_status == "completed"
                    and history_len >= expected_len
                    and all_steps_succeeded
                )
        except Exception:
            pass

        if (
            runner.delete_on_completion
            and final_status == "completed"
            and run_id_for_state is not None
        ):
            await state_manager.delete_workflow_state(run_id_for_state)
            try:
                if runner.state_backend is not None:
                    await runner.state_backend.delete_state(run_id_for_state)
            except Exception:
                pass
            try:
                if runner.state_backend is not None:
                    store = getattr(runner.state_backend, "_store", None)
                    if isinstance(store, dict):
                        store.clear()
            except Exception:
                pass

        execution_manager.set_final_context(paused_result, ctx)
        try:
            await runner._dispatch_hook(
                "post_run",
                pipeline_result=paused_result,
                context=ctx,
            )
        except Exception as e:  # noqa: BLE001
            try:
                telemetry.logfire.debug(f"post_run hook after resume failed: {e}")
            except Exception:
                pass
        return paused_result

    finally:
        pass


async def replay_from_trace(
    runner: "Flujo[_InT, _OutT, _CtxT]", run_id: str
) -> PipelineResult[_CtxT]:
    replay_executor = ReplayExecutor[_CtxT](runner)
    return await replay_executor.replay_from_trace(run_id)
