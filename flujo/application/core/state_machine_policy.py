from __future__ import annotations

from typing import Any, Optional, Any as _Any

from flujo.domain.models import StepResult, StepOutcome, Success, Failure, PipelineResult
from flujo.application.core.context_manager import ContextManager
from flujo.application.core.types import ExecutionFrame
from flujo.infra import telemetry


class StateMachinePolicyExecutor:
    """Policy executor for StateMachineStep.

    Iteratively executes the pipeline for the current state until an end state is reached.
    State is tracked in the context scratchpad under 'current_state'.
    Next state may be specified by setting 'next_state' in the context scratchpad.
    """

    async def execute(self, core: Any, frame: ExecutionFrame[_Any]) -> StepOutcome[StepResult]:
        step = frame.step
        data = frame.data
        context = frame.context
        resources = frame.resources
        limits = frame.limits

        # Defensive imports to avoid tight coupling
        try:
            from flujo.domain.dsl.state_machine import StateMachineStep  # noqa: F401
        except Exception:
            pass

        # Initialize current_state in context if absent
        current_state: Optional[str] = None
        try:
            if context is not None and hasattr(context, "scratchpad"):
                sp = getattr(context, "scratchpad")
                if isinstance(sp, dict):
                    current_state = sp.get("current_state")
        except Exception:
            current_state = None
        if not isinstance(current_state, str):
            current_state = getattr(step, "start_state", None)

        total_cost = 0.0
        total_tokens = 0
        total_latency = 0.0
        step_history: list[StepResult] = []
        last_context = context

        telemetry.logfire.info(f"[StateMachinePolicy] starting at state={current_state!r}")

        max_hops = max(1, len(getattr(step, "states", {})) * 10)
        for _hop in range(max_hops):
            if current_state is None:
                break

            end_states = getattr(step, "end_states", []) or []
            if isinstance(end_states, list) and current_state in end_states:
                telemetry.logfire.info(
                    f"[StateMachinePolicy] reached terminal state={current_state!r}"
                )
                break

            state_pipeline = getattr(step, "states", {}).get(current_state)
            if state_pipeline is None:
                # Unknown state → fail
                failure = StepResult(
                    name=getattr(step, "name", "StateMachine"),
                    output=None,
                    success=False,
                    feedback=f"Unknown state: {current_state}",
                    branch_context=last_context,
                )
                return Failure(
                    error=Exception("unknown_state"), feedback=failure.feedback, step_result=failure
                )

            # Build internal view (for future extension) but the pipeline is already provided
            try:
                _ = step.build_internal_pipeline()
            except Exception:
                _ = None

            # Isolate per-iteration context
            iteration_context = (
                ContextManager.isolate(last_context) if last_context is not None else None
            )

            # Execute current state's pipeline
            pipeline_result: PipelineResult[Any] = await core._execute_pipeline_via_policies(
                state_pipeline,
                data,
                iteration_context,
                resources,
                limits,
                None,
                frame.context_setter,
            )

            # Aggregate accounting
            total_cost += float(getattr(pipeline_result, "total_cost_usd", 0.0))
            total_tokens += int(getattr(pipeline_result, "total_tokens", 0))
            try:
                # Sum step latencies conservatively
                for sr in getattr(pipeline_result, "step_history", []) or []:
                    if isinstance(sr, StepResult):
                        total_latency += float(getattr(sr, "latency_s", 0.0))
                        step_history.append(sr)
            except Exception:
                pass

            # Determine next state from updated context (prefer iteration context)
            last_context = getattr(pipeline_result, "final_pipeline_context", iteration_context)
            next_state: Optional[str] = None
            try:
                if last_context is not None and hasattr(last_context, "scratchpad"):
                    sp2 = getattr(last_context, "scratchpad")
                    if isinstance(sp2, dict):
                        next_state = sp2.get("next_state")
            except Exception:
                next_state = None

            # Update current_state for next hop
            current_state = next_state if isinstance(next_state, str) else current_state
            # If next_state was not provided, check termination
            if not isinstance(next_state, str):
                if isinstance(end_states, list) and current_state in end_states:
                    break
                # Without an explicit transition, stop after one hop
                break

        # Build final StepResult that updates context
        result = StepResult(
            name=getattr(step, "name", "StateMachine"),
            output=None,
            success=True,
            attempts=1,
            latency_s=total_latency,
            token_counts=total_tokens,
            cost_usd=total_cost,
            feedback=None,
            branch_context=last_context,
            step_history=step_history,
        )
        return Success(step_result=result)
