from __future__ import annotations
from flujo.type_definitions.common import JSONObject

from typing import Optional

from flujo.domain.base_model import BaseModel as DomainBaseModel

from flujo.application.core.context_manager import ContextManager
from flujo.domain.models import Failure, PipelineResult, StepOutcome, StepResult, Success
from flujo.exceptions import PausedException
from flujo.infra import telemetry

from ..types import ExecutionFrame
from flujo.domain.dsl.state_machine import StateMachineStep  # noqa: F401

# --- StateMachine policy executor (FSD-025) ---


class StateMachinePolicyExecutor:
    """Policy executor for StateMachineStep.

    Iteratively executes the pipeline for the current state until an end state is reached.
    State is tracked in typed context fields (current_state/next_state).
    """

    async def execute(
        self, core: object, frame: ExecutionFrame[DomainBaseModel]
    ) -> StepOutcome[StepResult]:
        step = frame.step
        data = frame.data
        context = frame.context
        resources = frame.resources
        limits = frame.limits

        current_state: Optional[str] = (
            getattr(context, "current_state", None) if context is not None else None
        )
        if not isinstance(current_state, str):
            current_state = getattr(step, "start_state", None)

        # Defensive: immediately persist the starting state to the caller-visible context.
        # Some CI paths observed missing 'current_state' when orchestration short‑circuits
        # (e.g., pause/empty-state pipelines). Setting it up‑front prevents loss during
        # later merges and ensures tests can assert on it even if a pause occurs early.
        try:
            if context is not None and isinstance(current_state, str):
                if hasattr(context, "current_state"):
                    context.current_state = current_state
        except Exception:
            pass

        total_cost = 0.0
        total_tokens = 0
        total_latency = 0.0
        step_history: list[StepResult] = []
        last_context = context

        telemetry.logfire.info(f"[StateMachinePolicy] starting at state={current_state!r}")

        max_hops = max(1, len(getattr(step, "states", {})) * 10)

        # Helper: resolve transitions with first-match-wins semantics
        def _resolve_transition(
            _step: object,
            _from_state: Optional[str],
            _event: str,
            _payload: JSONObject,
            _context: DomainBaseModel | None,
        ) -> Optional[str]:
            try:
                trs = getattr(_step, "transitions", None) or []
            except Exception:
                trs = []
            if not trs:
                return None
            for tr in trs:
                try:
                    frm = getattr(tr, "from_state", None)
                    ev = getattr(tr, "on", None)
                    if ev != _event:
                        continue
                    if frm not in ("*", _from_state):
                        continue
                    # Evaluate predicate if present
                    when_fn = getattr(tr, "_when_fn", None)
                    if when_fn is not None:
                        try:
                            ok = bool(when_fn(_payload, _context))
                        except Exception:
                            telemetry.logfire.warning(
                                "[StateMachinePolicy] when() evaluation failed; skipping rule"
                            )
                            ok = False
                        if not ok:
                            continue
                    return getattr(tr, "to", None)
                except Exception:
                    # Skip malformed rules defensively
                    continue
            return None

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

            try:
                _ = step.build_internal_pipeline()
            except Exception:
                _ = None

            # Persist control metadata so resume knows current state
            try:
                if last_context is not None and isinstance(current_state, str):
                    if hasattr(last_context, "current_state"):
                        last_context.current_state = current_state
            except Exception:
                pass

            isolated_ctx = (
                ContextManager.isolate(last_context) if last_context is not None else None
            )
            iteration_context: DomainBaseModel | None = (
                isolated_ctx if isinstance(isolated_ctx, DomainBaseModel) else last_context
            )

            # Disable cache during state transitions to avoid stale context
            original_cache_enabled = getattr(core, "_enable_cache", True)
            try:
                setattr(core, "_enable_cache", False)
                try:
                    exec_fn = getattr(core, "_execute_pipeline_via_policies", None)
                    if not callable(exec_fn):
                        raise TypeError("ExecutorCore missing _execute_pipeline_via_policies()")
                    pipeline_result: PipelineResult[DomainBaseModel] = await exec_fn(
                        state_pipeline,
                        data,
                        iteration_context,
                        resources,
                        limits,
                        frame.context_setter,
                    )
                except PausedException as e:
                    # On pause, do not merge iteration context; resolve pause transition and re-raise
                    try:
                        # Build minimal payload for expressions
                        pause_payload: JSONObject = {
                            "event": "pause",
                            "last_output": None,
                            "last_step": None,
                        }
                        target = _resolve_transition(
                            step, current_state, "pause", pause_payload, iteration_context
                        )
                        if isinstance(target, str) and last_context is not None:
                            if hasattr(last_context, "current_state"):
                                last_context.current_state = target
                            if hasattr(last_context, "next_state"):
                                last_context.next_state = target
                            # Mark paused status and message on main context for persistence
                            try:
                                if hasattr(last_context, "status"):
                                    last_context.status = "paused"
                                msg = getattr(e, "message", None)
                                if hasattr(last_context, "pause_message"):
                                    last_context.pause_message = (
                                        msg if isinstance(msg, str) else getattr(e, "message", "")
                                    )
                            except Exception:
                                pass
                        # Best‑effort: reflect pause metadata on the outer context as well so
                        # finalization code that uses the original reference also sees it.
                        try:
                            if context is not None:
                                if isinstance(target, str):
                                    if hasattr(context, "current_state"):
                                        context.current_state = target
                                    if hasattr(context, "next_state"):
                                        context.next_state = target
                                if hasattr(context, "status"):
                                    context.status = "paused"
                                if (
                                    hasattr(context, "pause_message")
                                    and context.pause_message is None
                                ):
                                    context.pause_message = (
                                        getattr(e, "message", "")
                                        if isinstance(getattr(e, "message", None), str)
                                        else getattr(e, "message", "")
                                    )
                        except Exception:
                            pass
                        telemetry.logfire.info(
                            f"[StateMachinePolicy] pause in state={current_state!r}; transitioning to={target!r}"
                        )
                    except Exception:
                        pass
                    raise e
            finally:
                try:
                    setattr(core, "_enable_cache", original_cache_enabled)
                except Exception:
                    pass
                # Fast-path: removed for state machine policy; agent success is handled per-step

            total_cost += float(getattr(pipeline_result, "total_cost_usd", 0.0))
            total_tokens += int(getattr(pipeline_result, "total_tokens", 0))
            try:
                hist = getattr(pipeline_result, "step_history", []) or []
                if not hist:
                    # Ensure state visibility even when sub-pipeline produced no step_history
                    step_history.append(
                        StepResult(
                            name=current_state,
                            success=bool(getattr(pipeline_result, "success", True)),
                            output=getattr(pipeline_result, "output", None)
                            if hasattr(pipeline_result, "output")
                            else None,
                            attempts=1,
                            latency_s=float(getattr(pipeline_result, "latency_s", 0.0) or 0.0),
                            token_counts=int(getattr(pipeline_result, "total_tokens", 0) or 0),
                            cost_usd=float(getattr(pipeline_result, "total_cost_usd", 0.0) or 0.0),
                            feedback=None,
                            step_history=list(hist),
                        )
                    )
                for sr in hist:
                    if isinstance(sr, StepResult):
                        total_latency += float(getattr(sr, "latency_s", 0.0))
                        step_history.append(sr)
            except Exception:
                pass

            # Defensive: Ensure current_state is preserved in the final result context
            # (typed field; no scratchpad writes).
            if current_state is not None:
                try:
                    if getattr(pipeline_result, "final_pipeline_context", None) is not None:
                        ctx = getattr(pipeline_result, "final_pipeline_context")
                        if ctx is not None and hasattr(ctx, "current_state"):
                            if getattr(ctx, "current_state", None) is None:
                                ctx.current_state = current_state
                except Exception:
                    pass

            # Merge sub-pipeline's final context back into the state machine's main context
            sub_ctx_raw = getattr(pipeline_result, "final_pipeline_context", iteration_context)
            sub_ctx: DomainBaseModel | None = (
                sub_ctx_raw if isinstance(sub_ctx_raw, DomainBaseModel) else iteration_context
            )
            # Capture next_state/current_state from the sub-context BEFORE merge.
            # ContextManager.merge may exclude some fields for noise reduction; we extract
            # the intended hop here and re-apply it after the merge.
            intended_next: Optional[str] = None
            intended_curr: Optional[str] = None
            try:
                # Prefer typed fields
                if sub_ctx is not None:
                    nxt = getattr(sub_ctx, "next_state", None)
                    cur = getattr(sub_ctx, "current_state", None)
                    intended_next = str(nxt) if isinstance(nxt, str) else None
                    intended_curr = str(cur) if isinstance(cur, str) else None
            except Exception:
                intended_next = None
                intended_curr = None
            try:
                merged_ctx = ContextManager.merge(last_context, sub_ctx)
                last_context = merged_ctx if isinstance(merged_ctx, DomainBaseModel) else sub_ctx
            except Exception:
                # Defensive: if merge fails, fall back to sub_ctx to avoid losing progress
                last_context = sub_ctx
            # NOTE: scratchpad deep-merges removed. State propagation is via typed fields only.
            # Re-apply intended state transition to the merged context when available (typed fields).
            try:
                if last_context is not None:
                    if isinstance(intended_next, str) and hasattr(last_context, "next_state"):
                        last_context.next_state = intended_next
                    if hasattr(last_context, "current_state"):
                        if isinstance(intended_curr, str):
                            last_context.current_state = intended_curr
                        elif isinstance(intended_next, str):
                            last_context.current_state = intended_next
            except Exception:
                pass
            # Decide transition based on pipeline_result event before legacy fallbacks
            next_state: Optional[str] = None
            try:
                # Determine event from last step result
                last_sr = None
                try:
                    if getattr(pipeline_result, "step_history", None):
                        last_sr = pipeline_result.step_history[-1]
                except Exception:
                    last_sr = None
                event = "success"
                if last_sr is not None and isinstance(getattr(last_sr, "success", None), bool):
                    event = "success" if last_sr.success else "failure"
                # Prepare payload for expressions
                event_payload: JSONObject = {
                    "event": event,
                    "last_output": getattr(last_sr, "output", None)
                    if last_sr is not None
                    else None,
                    "last_step": {
                        "name": getattr(last_sr, "name", None) if last_sr is not None else None,
                        "success": getattr(last_sr, "success", None)
                        if last_sr is not None
                        else None,
                        "feedback": getattr(last_sr, "feedback", None)
                        if last_sr is not None
                        else None,
                    },
                }
                target = _resolve_transition(
                    step, current_state, event, event_payload, last_context
                )
                if isinstance(target, str):
                    next_state = target
                    try:
                        if last_context is not None:
                            if hasattr(last_context, "next_state"):
                                last_context.next_state = str(target)
                            if hasattr(last_context, "current_state"):
                                last_context.current_state = str(target)
                        telemetry.logfire.info(
                            f"[StateMachinePolicy] event={event} from={current_state!r} matched to={target!r}"
                        )
                    except Exception:
                        pass
            except Exception:
                pass

            # If no transition rule matched, look for explicit next_state in context
            try:
                if last_context is not None and not isinstance(next_state, str):
                    nxt = getattr(last_context, "next_state", None)
                    if isinstance(nxt, str):
                        next_state = nxt
            except Exception:
                next_state = None

            # Fallback: derive next_state from step outputs when context wasn't updated
            if not isinstance(next_state, str):
                try:
                    telemetry.logfire.info(
                        "[StateMachinePolicy] Looking for next_state in step outputs"
                    )
                    for sr in reversed(getattr(pipeline_result, "step_history", []) or []):
                        out = getattr(sr, "output", None)
                        telemetry.logfire.info(f"[StateMachinePolicy] Step output: {out}")
                        if isinstance(out, dict):
                            if isinstance(out.get("next_state"), str):
                                next_state = out.get("next_state")
                                telemetry.logfire.info(
                                    f"[StateMachinePolicy] Found next_state: {next_state}"
                                )
                                # Best-effort: persist into typed fields for downstream steps
                                try:
                                    if last_context is not None:
                                        if hasattr(last_context, "next_state"):
                                            last_context.next_state = str(next_state)
                                        if (
                                            hasattr(last_context, "current_state")
                                            and getattr(last_context, "current_state", None) is None
                                        ):
                                            last_context.current_state = str(next_state)
                                except Exception:
                                    pass
                                break
                except Exception:
                    pass

            current_state = next_state if isinstance(next_state, str) else current_state
            if not isinstance(next_state, str):
                if isinstance(end_states, list) and current_state in end_states:
                    break
                break

        # Final safeguard: ensure the final state is visible on the caller context (typed fields only).
        try:
            if isinstance(current_state, str):
                for ctx_obj in (last_context, context):
                    if ctx_obj is not None:
                        if hasattr(ctx_obj, "current_state"):
                            ctx_obj.current_state = current_state
                        if hasattr(ctx_obj, "next_state") and not isinstance(
                            getattr(ctx_obj, "next_state", None), str
                        ):
                            ctx_obj.next_state = current_state
        except Exception:
            pass

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
        # Best-effort: inform the context_setter of the final merged context so
        # ExecutionManager can persist it consistently across policy types.
        try:
            if frame.context_setter is not None:
                pr: PipelineResult[DomainBaseModel] = PipelineResult(
                    step_history=step_history,
                    total_cost_usd=total_cost,
                    total_tokens=total_tokens,
                    final_pipeline_context=last_context,
                )
                frame.context_setter(pr, last_context)
        except Exception:
            pass
        return Success(step_result=result)
