"""Loop orchestration extracted from ExecutorCore."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from ...domain.models import PipelineResult, StepResult
from .executor_helpers import make_execution_frame

if TYPE_CHECKING:  # pragma: no cover
    from .executor_core import ExecutorCore


class LoopOrchestrator:
    """Handles LoopStep execution and legacy loop semantics for compatibility."""

    async def execute(
        self,
        *,
        core: "ExecutorCore[Any]",
        loop_step: Any,
        data: Any,
        context: Any,
        resources: Any,
        limits: Any,
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
        fallback_depth: int = 0,
    ) -> StepResult:
        try:
            from ...domain.dsl.loop import LoopStep as _DSLLoop
        except Exception:
            _DSLLoop = None  # type: ignore

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

        current_context = context or _PipelineContext(initial_prompt=str(data))
        main_context = current_context
        current_input = data
        attempts = 0
        step_history_tracker = core._step_history_tracker
        last_feedback: Optional[str] = None
        exit_reason: str = "max_loops"
        final_output: Any = None

        for i in range(1, max_loops + 1):
            attempts = i
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
                    from .context_manager import ContextManager as _CM

                    iter_context = _CM.isolate(main_context) if main_context is not None else None
                except Exception:
                    iter_context = main_context

                pr = await core._execute_pipeline(
                    body,
                    body_output,
                    iter_context,
                    resources,
                    limits,
                    None,
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
                        from .context_manager import ContextManager as _CM

                        _CM.merge(main_context, pr.final_pipeline_context)
                        current_context = main_context
                except Exception:
                    current_context = main_context
            except Exception:
                for s in steps:
                    sr = await core._execute_simple_step(
                        s,
                        body_output,
                        current_context,
                        resources,
                        limits,
                        False,
                        None,
                        None,
                        0,
                        False,
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
                            branch_context=current_context,
                            metadata_={"iterations": i, "exit_reason": "body_step_error"},
                            step_history=step_history_tracker.get_history(),
                        )
                    body_output = sr.output

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
