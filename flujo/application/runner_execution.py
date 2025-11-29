from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Literal, Optional, TYPE_CHECKING, TypeVar, cast

from pydantic import ValidationError

from ..exceptions import OrchestratorError, PipelineAbortSignal
from ..domain.dsl.step import HumanInTheLoopStep
from ..domain.models import (
    ExecutedCommandLog,
    HumanInteraction,
    PipelineContext,
    PipelineResult,
    StepResult,
)
from ..domain.commands import AgentCommand
from ..state import WorkflowState
from ..infra import telemetry
from .core.execution_manager import ExecutionManager
from .core.state_manager import StateManager

if TYPE_CHECKING:  # pragma: no cover
    from .runner import Flujo
    from pydantic import TypeAdapter

_CtxT = TypeVar("_CtxT", bound=PipelineContext)


async def resume_async_inner(
    runner: "Flujo[Any, Any, _CtxT]",
    paused_result: PipelineResult[_CtxT],
    human_input: Any,
    agent_command_adapter: "TypeAdapter[AgentCommand]",
) -> PipelineResult[_CtxT]:
    """Resume a paused pipeline with human input."""
    try:
        ctx: PipelineContext | None = paused_result.final_pipeline_context
        if ctx is None:
            raise OrchestratorError("Cannot resume pipeline without context")
        scratch = getattr(ctx, "scratchpad", {})
        if scratch.get("status") != "paused":
            raise OrchestratorError("Pipeline is not paused")
        runner._ensure_pipeline()
        assert runner.pipeline is not None
        try:
            pause_msg = scratch.get("pause_message")
            hist = getattr(ctx, "conversation_history", None)
            if isinstance(hist, list) and pause_msg:
                last_content = hist[-1].content if hist else None
                if last_content != pause_msg:
                    from flujo.domain.models import ConversationTurn, ConversationRole

                    hist.append(
                        ConversationTurn(role=ConversationRole.assistant, content=str(pause_msg))
                    )
                    setattr(ctx, "conversation_history", hist)
        except Exception:
            pass
        start_idx = len(paused_result.step_history)
        if scratch.get("status") == "paused":
            start_idx = min(len(runner.pipeline.steps) - 1, len(paused_result.step_history))
            scratch["hitl_data"] = human_input
            scratch["user_input"] = human_input
        if start_idx >= len(runner.pipeline.steps):
            raise OrchestratorError("No steps remaining to resume")
        paused_step = runner.pipeline.steps[start_idx]

        if isinstance(paused_step, HumanInTheLoopStep) and paused_step.input_schema is not None:
            human_input = paused_step.input_schema.model_validate(human_input)

        if isinstance(ctx, PipelineContext):
            ctx.hitl_history.append(
                HumanInteraction(
                    message_to_human=scratch.get("pause_message", ""),
                    human_response=human_input,
                )
            )
            ctx.scratchpad["status"] = "running"

        try:
            if isinstance(ctx, PipelineContext):
                if not isinstance(getattr(ctx, "conversation_history", None), list):
                    setattr(ctx, "conversation_history", [])
                from flujo.domain.models import ConversationTurn, ConversationRole

                hist = ctx.conversation_history
                last_content = hist[-1].content if hist else None
                text = str(human_input)
                if text and text != last_content:
                    hist.append(ConversationTurn(role=ConversationRole.user, content=text))
        except Exception:
            pass

        paused_step_result = StepResult(
            name=paused_step.name,
            output=human_input,
            success=True,
            attempts=1,
        )

        if hasattr(paused_step, "sink_to") and paused_step.sink_to and ctx is not None:
            try:
                from flujo.utils.context import set_nested_context_field

                set_nested_context_field(ctx, paused_step.sink_to, human_input)
            except Exception:
                pass

        try:
            if isinstance(ctx, PipelineContext):
                sp = getattr(ctx, "scratchpad", None)
                if not isinstance(sp, dict):
                    setattr(ctx, "scratchpad", {"steps": {}})
                    sp = getattr(ctx, "scratchpad", None)
                if isinstance(sp, dict):
                    steps_map = sp.get("steps")
                    if not isinstance(steps_map, dict):
                        steps_map = {}
                        sp["steps"] = steps_map
                    try:
                        val = human_input
                        if isinstance(val, bytes):
                            try:
                                val = val.decode("utf-8", errors="ignore")
                            except Exception:
                                val = str(val)
                        else:
                            val = str(val)
                        if len(val) > 1024:
                            val = val[:1024]
                        steps_map[getattr(paused_step, "name", "")] = val
                    except Exception:
                        steps_map[getattr(paused_step, "name", "")] = ""
        except Exception:
            pass
        if isinstance(ctx, PipelineContext):
            pending = ctx.scratchpad.pop("paused_step_input", None)
            if pending is not None:
                try:
                    from flujo.domain.commands import (
                        RunAgentCommand as _Run,
                        AskHumanCommand as _Ask,
                        FinishCommand as _Fin,
                    )

                    if isinstance(pending, (_Run, _Ask, _Fin)):
                        pending_cmd = pending
                    else:
                        pending_cmd = agent_command_adapter.validate_python(pending)
                except ValidationError:
                    pending_cmd = None
                except Exception:
                    pending_cmd = None
                if pending_cmd is not None:
                    log_entry = ExecutedCommandLog(
                        turn=len(ctx.command_log) + 1,
                        generated_command=pending_cmd,
                        execution_result=human_input,
                    )
                    ctx.command_log.append(log_entry)
                    try:
                        if isinstance(ctx.scratchpad, dict):
                            ctx.scratchpad["loop_last_output"] = log_entry
                    except Exception:
                        pass
                else:
                    try:
                        from flujo.domain.commands import AskHumanCommand as _Ask

                        log_entry = ExecutedCommandLog(
                            turn=len(ctx.command_log) + 1,
                            generated_command=_Ask(question=scratch.get("pause_message", "Paused")),
                            execution_result=human_input,
                        )
                        ctx.command_log.append(log_entry)
                        try:
                            if isinstance(ctx.scratchpad, dict):
                                ctx.scratchpad["loop_last_output"] = log_entry
                        except Exception:
                            pass
                    except Exception:
                        pass
        from ..domain.dsl.step import HumanInTheLoopStep as _HITL

        try:
            if runner._trace_manager is not None:
                summary = str(human_input)
                if isinstance(summary, str) and len(summary) > 500:
                    summary = summary[:500] + "..."
                runner._trace_manager.add_event("flujo.resumed", {"human_input": summary})
        except Exception:
            pass

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
                            raise OrchestratorError(
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

        run_id_for_state = getattr(ctx, "run_id", None)
        state_created_at: datetime | None = None
        if runner.state_backend is not None and run_id_for_state is not None:
            loaded = await runner.state_backend.load_state(run_id_for_state)
            if loaded is not None:
                wf_state_loaded = WorkflowState.model_validate(loaded)
                state_created_at = wf_state_loaded.created_at
        try:
            async for _ in runner._execute_steps(
                resume_start_idx,
                data,
                cast(Optional[_CtxT], ctx),
                paused_result,
                stream_last=False,
                run_id=run_id_for_state,
                state_backend=runner.state_backend,
                state_created_at=state_created_at,
            ):
                pass
        except PipelineAbortSignal:
            if isinstance(ctx, PipelineContext):
                ctx.scratchpad["status"] = "paused"

        final_status: Literal["running", "paused", "completed", "failed", "cancelled"]
        if paused_result.step_history:
            final_status = (
                "completed" if all(s.success for s in paused_result.step_history) else "failed"
            )
        else:
            final_status = "failed"
        if isinstance(ctx, PipelineContext):
            if ctx.scratchpad.get("status") == "paused":
                final_status = "paused"
            ctx.scratchpad["status"] = final_status

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
            start_idx=len(paused_result.step_history),
            state_created_at=state_created_at,
            final_status=final_status,
        )

        try:
            paused_result.success = final_status == "completed"
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

        execution_manager.set_final_context(paused_result, cast(Optional[PipelineContext], ctx))
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


async def replay_from_trace(runner: "Flujo[Any, Any, _CtxT]", run_id: str) -> PipelineResult[_CtxT]:
    if runner.state_backend is None:
        raise OrchestratorError("Replay requires a state_backend with trace support")
    stored = await runner.state_backend.get_run_details(run_id)
    steps = await runner.state_backend.list_run_steps(run_id)
    trace = await runner.state_backend.get_trace(run_id)

    if stored is None:
        raise OrchestratorError(f"No stored run metadata for run_id={run_id}")
    if steps is None:
        steps = []

    initial_input: Any = None
    initial_context_data: Dict[str, Any] = {}
    try:
        if isinstance(trace, dict):
            attrs = trace.get("attributes", {}) if trace else {}
            initial_input = attrs.get("flujo.input", None)
    except Exception:
        initial_input = None
    try:
        loaded_state = await runner.state_backend.load_state(run_id)
        if loaded_state is not None:
            initial_context_data = loaded_state.get("pipeline_context") or {}
    except Exception:
        initial_context_data = {}

    response_map: Dict[str, Any] = {}
    for s in steps:
        step_name = s.get("step_name", "")
        key = f"{step_name}:attempt_1"
        raw_resp = s.get("raw_response")
        if raw_resp is None:
            raw_resp = s.get("output")
        response_map[key] = raw_resp

    human_inputs: list[Any] = []

    def _collect_events(span: Dict[str, Any]) -> None:
        try:
            for ev in span.get("events", []) or []:
                if ev.get("name") == "flujo.resumed":
                    human_inputs.append(ev.get("attributes", {}).get("human_input"))
            for ch in span.get("children", []) or []:
                _collect_events(ch)
        except Exception:
            pass

    if isinstance(trace, dict):
        _collect_events(trace)

    from ..testing.replay import ReplayAgent

    replay_agent = ReplayAgent(response_map)

    runner._ensure_pipeline()
    assert runner.pipeline is not None
    for st in runner.pipeline.steps:
        try:
            setattr(st, "agent", replay_agent)
        except Exception:
            pass

    original_resume = runner.resume_async

    async def _resume_patched(
        paused_result: PipelineResult[_CtxT], human_input: Any
    ) -> PipelineResult[_CtxT]:
        if not human_inputs:
            raise OrchestratorError("ReplayError: no recorded human input available for resume")
        next_input = human_inputs.pop(0)
        return await original_resume(paused_result, next_input)

    runner.resume_async = _resume_patched  # type: ignore[method-assign]

    final_result: PipelineResult[_CtxT] | None = None
    async for item in runner.run_async(initial_input, initial_context_data=initial_context_data):
        final_result = item
    assert final_result is not None

    from ..domain.models import PipelineContext as _PipelineContext

    while True:
        try:
            ctx = getattr(final_result, "final_pipeline_context", None)
            is_paused = False
            if _PipelineContext is not None and isinstance(ctx, _PipelineContext):
                status = ctx.scratchpad.get("status") if hasattr(ctx, "scratchpad") else None
                is_paused = status == "paused"
            if not is_paused:
                break
            if not human_inputs:
                raise OrchestratorError("ReplayError: no recorded human input available for resume")
            next_human_input = human_inputs.pop(0)
            final_result = await original_resume(final_result, next_human_input)
        except OrchestratorError:
            raise
        except Exception as replay_exc:  # noqa: BLE001
            raise OrchestratorError(f"ReplayError: {replay_exc}") from replay_exc

    runner.resume_async = original_resume  # type: ignore[method-assign]
    return final_result
