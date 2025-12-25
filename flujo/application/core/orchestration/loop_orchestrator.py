"""Loop orchestration extracted from ExecutorCore."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ....domain.models import BaseModel as DomainBaseModel
from ....domain.models import PipelineResult, StepResult, UsageLimits
from ....exceptions import PausedException, PipelineAbortSignal, InfiniteRedirectError
from ..execution.executor_helpers import make_execution_frame
from ..types import TContext

if TYPE_CHECKING:  # pragma: no cover
    from ..executor_core import ExecutorCore


class LoopOrchestrator:
    """Handles LoopStep execution and legacy loop semantics for compatibility."""

    async def execute(
        self,
        *,
        core: "ExecutorCore[TContext]",
        loop_step: object,
        data: object,
        context: TContext | None,
        resources: object | None,
        limits: UsageLimits | None,
        context_setter: Callable[[PipelineResult[DomainBaseModel], DomainBaseModel | None], None]
        | None,
        fallback_depth: int = 0,
    ) -> StepResult:
        try:
            from ....domain.dsl.loop import LoopStep as _DSLLoop
        except Exception:
            _DSLLoop = None  # type: ignore[misc,assignment]

        if _DSLLoop is not None and isinstance(loop_step, _DSLLoop):
            frame = make_execution_frame(
                core,
                loop_step,
                data,
                context,
                resources,
                limits,
                context_setter=context_setter,
                stream=False,
                on_chunk=None,
                fallback_depth=fallback_depth,
                result=None,
                quota=core._get_current_quota() if hasattr(core, "_get_current_quota") else None,
            )
            outcome = await core.loop_step_executor.execute(core, frame)
            return core._unwrap_outcome_to_step_result(outcome, core._safe_step_name(loop_step))

        # Legacy lightweight loop execution for ad-hoc objects used in unit tests
        from flujo.domain.models import PipelineContext as _PipelineContext

        name = getattr(loop_step, "name", "loop")
        max_loops = int(getattr(loop_step, "max_loops", 1) or 1)
        body = getattr(loop_step, "loop_body_pipeline", None)
        steps = []
        try:
            steps = list(getattr(body, "steps", []) or [])
        except Exception:
            steps = []

        exit_condition = getattr(loop_step, "exit_condition_callable", None)
        initial_mapper = getattr(loop_step, "initial_input_to_loop_body_mapper", None)
        iter_mapper = getattr(loop_step, "iteration_input_mapper", None)
        output_mapper = getattr(loop_step, "loop_output_mapper", None)

        current_context: DomainBaseModel = context or _PipelineContext(initial_prompt=str(data))
        main_context = current_context
        current_input = data
        attempts = 0
        step_history_tracker = core._step_history_tracker
        last_feedback: str | None = None
        exit_reason: str = "max_loops"
        final_output: object | None = None

        for i in range(1, max_loops + 1):
            attempts = i
            iter_context: DomainBaseModel | None = None
            try:
                if i == 1 and callable(initial_mapper):
                    current_input = initial_mapper(current_input, current_context)
                elif callable(iter_mapper):
                    try:
                        current_input = iter_mapper(current_input, current_context, i)
                    except TypeError:
                        current_input = iter_mapper(current_input, current_context)
            except Exception:
                fb = f"Error in input mapper for LoopStep '{name}'"
                return StepResult(
                    name=name,
                    success=False,
                    output=None,
                    attempts=attempts,
                    latency_s=0.0,
                    token_counts=0,
                    cost_usd=0.0,
                    feedback=fb,
                    branch_context=current_context,
                    metadata_={"iterations": i, "exit_reason": "input_mapper_error"},
                    step_history=step_history_tracker.get_history(),
                )

            body_output = current_input
            try:
                try:
                    from ..context_manager import ContextManager as _CM

                    iter_context = (
                        _CM.isolate_strict(main_context) if main_context is not None else None
                    )
                except Exception as exc:
                    fb = f"LoopStep '{name}' context isolation failed: {exc}"
                    return StepResult(
                        name=name,
                        success=False,
                        output=None,
                        attempts=attempts,
                        latency_s=0.0,
                        token_counts=0,
                        cost_usd=0.0,
                        feedback=fb,
                        branch_context=current_context,
                        metadata_={"iterations": i, "exit_reason": "context_isolation_error"},
                        step_history=step_history_tracker.get_history(),
                    )
                if iter_context is None:
                    fb = f"LoopStep '{name}' context isolation returned None"
                    return StepResult(
                        name=name,
                        success=False,
                        output=None,
                        attempts=attempts,
                        latency_s=0.0,
                        token_counts=0,
                        cost_usd=0.0,
                        feedback=fb,
                        branch_context=current_context,
                        metadata_={"iterations": i, "exit_reason": "context_isolation_error"},
                        step_history=step_history_tracker.get_history(),
                    )

                if body is None or resources is None or limits is None or main_context is None:
                    raise RuntimeError("Legacy loop missing pipeline prerequisites")
                pr = await core._pipeline_orchestrator.execute(
                    core=core,
                    pipeline=body,
                    data=body_output,
                    context=iter_context,
                    resources=resources,
                    limits=limits,
                    context_setter=None,
                )
                try:
                    if pr.step_history:
                        step_history_tracker.extend_history(pr.step_history)
                        body_output = pr.step_history[-1].output
                except Exception:
                    pass
                try:
                    if (
                        hasattr(pr, "final_pipeline_context")
                        and pr.final_pipeline_context is not None
                        and main_context is not None
                    ):
                        from ..context_manager import ContextManager as _CM

                        _CM.merge(main_context, pr.final_pipeline_context)
                        current_context = main_context
                except Exception:
                    current_context = main_context
            except (PausedException, PipelineAbortSignal, InfiniteRedirectError):
                raise
            except Exception:
                for s in steps:
                    frame = make_execution_frame(
                        core,
                        s,
                        body_output,
                        iter_context,
                        resources,
                        limits,
                        context_setter=None,
                        stream=False,
                        on_chunk=None,
                        fallback_depth=0,
                        result=None,
                        quota=core._get_current_quota()
                        if hasattr(core, "_get_current_quota")
                        else None,
                    )
                    step_outcome = await core.execute(frame)
                    sr = (
                        step_outcome
                        if isinstance(step_outcome, StepResult)
                        else core._unwrap_outcome_to_step_result(
                            step_outcome, core._safe_step_name(s)
                        )
                    )
                    step_history_tracker.add_step_result(sr)
                    if not sr.success:
                        fb = f"Loop body failed: {sr.feedback or 'Unknown error'}"
                        return StepResult(
                            name=name,
                            success=False,
                            output=None,
                            attempts=attempts,
                            latency_s=0.0,
                            token_counts=0,
                            cost_usd=0.0,
                            feedback=fb,
                            branch_context=iter_context,
                            metadata_={"iterations": i, "exit_reason": "body_step_error"},
                            step_history=step_history_tracker.get_history(),
                        )
                    body_output = sr.output
                try:
                    if iter_context is not None and main_context is not None:
                        from ..context_manager import ContextManager as _CM

                        _CM.merge(main_context, iter_context)
                        current_context = main_context
                except Exception:
                    current_context = main_context

            try:
                if callable(exit_condition) and exit_condition(body_output, current_context):
                    final_output = body_output
                    if callable(output_mapper):
                        final_output = output_mapper(final_output, current_context)
                    exit_reason = "condition"
                    return StepResult(
                        name=name,
                        success=True,
                        output=final_output,
                        attempts=attempts,
                        latency_s=0.0,
                        token_counts=0,
                        cost_usd=0.0,
                        feedback=None,
                        branch_context=current_context,
                        metadata_={"iterations": i, "exit_reason": exit_reason},
                        step_history=step_history_tracker.get_history(),
                    )
            except (PausedException, PipelineAbortSignal, InfiniteRedirectError):
                raise
            except Exception as exc:
                try:
                    from ....infra import telemetry

                    telemetry.logfire.warning(
                        f"Exit condition evaluation failed in LoopStep '{name}': {exc}"
                    )
                except Exception:
                    pass

            final_output = body_output
            try:
                if callable(output_mapper):
                    final_output = output_mapper(final_output, current_context)
            except Exception as e:
                last_feedback = str(e)
                exit_reason = "output_mapper_error"
                final_output = None
                continue

            current_input = final_output

        success = exit_reason == "condition"
        return StepResult(
            name=name,
            success=success,
            output=final_output,
            attempts=attempts,
            latency_s=0.0,
            token_counts=0,
            cost_usd=0.0,
            feedback=(f"Output mapper failed: {last_feedback}" if last_feedback else None)
            if not success
            else None,
            branch_context=current_context,
            metadata_={"iterations": attempts, "exit_reason": exit_reason},
            step_history=step_history_tracker.get_history(),
        )
