from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Optional
from pydantic import BaseModel

from ....domain.models import StepOutcome, StepResult, Success, Failure
from ....infra import telemetry
from ....utils.performance import time_perf_ns, time_perf_ns_to_seconds


@dataclass
class FallbackState:
    """Accounting data for fallback aggregation."""

    prompt_tokens_latest: int
    completion_tokens_latest: int
    best_primary_tokens: int
    best_primary_cost_usd: float
    primary_tokens_total: int
    primary_tokens_known: bool
    last_plugin_failure_feedback: Optional[str]


@dataclass
class FallbackHandlingResult:
    """Result of running a fallback step after a failed attempt."""

    outcome: StepOutcome[StepResult]
    resources_closed: bool


class AgentFallbackHandler:
    """Handles fallback execution paths for agent attempts."""

    async def handle_failure(
        self,
        *,
        core: object,
        step: object,
        data: object,
        attempt_context: BaseModel | None,
        attempt_resources: object | None,
        limits: object | None,
        stream: bool,
        on_chunk: Optional[Callable[[object], Awaitable[None]]],
        fallback_depth: int,
        start_ns: int,
        result: StepResult,
        primary_feedback: str,
        attempt_exc: BaseException,
        attempt: int,
        total_attempts: int,
        pre_attempt_context: BaseModel | None,
        _context: BaseModel | None,
        close_resources: Callable[[BaseException | None], Awaitable[None]],
        state: FallbackState,
    ) -> FallbackHandlingResult:
        def _safe_step_name(step_obj: object) -> str:
            safe_fn = getattr(core, "_safe_step_name", None)
            if callable(safe_fn):
                try:
                    return str(safe_fn(step_obj))
                except Exception:
                    pass
            return str(getattr(step_obj, "name", "<unnamed>"))

        fb_step = getattr(step, "fallback_step", None)
        if hasattr(fb_step, "_mock_name") and not hasattr(fb_step, "agent"):
            fb_step = None
        if fb_step is None:
            await close_resources(attempt_exc)
            branch_context: BaseModel | None = None
            if getattr(step, "updates_context", False):
                branch_context = attempt_context
                # Preserve attempt mutations, but roll back HITL-specific fields that should not
                # commit when a run fails (mirrors AgentExecutionRunner safeguards).
                if branch_context is not None and pre_attempt_context is not None:
                    for attr in (
                        "total_interactions",
                        "interaction_history",
                        "hitl_data",
                        "hitl_history",
                        "current_interaction",
                        "human_interactions",
                        "approval_count",
                        "rejection_count",
                    ):
                        if hasattr(pre_attempt_context, attr):
                            try:
                                setattr(branch_context, attr, getattr(pre_attempt_context, attr))
                            except Exception:
                                pass
            return FallbackHandlingResult(
                outcome=Failure(
                    error=attempt_exc,
                    feedback=primary_feedback,
                    step_result=StepResult(
                        name=_safe_step_name(step),
                        output=None,
                        success=False,
                        attempts=result.attempts,
                        latency_s=time_perf_ns_to_seconds(time_perf_ns() - start_ns),
                        token_counts=state.best_primary_tokens or result.token_counts,
                        cost_usd=state.best_primary_cost_usd or result.cost_usd,
                        feedback=primary_feedback,
                        branch_context=branch_context,
                        metadata_={},
                        step_history=[],
                    ),
                ),
                resources_closed=True,
            )

        try:
            telemetry.logfire.debug(
                f"[AgentExecutionRunner] Invoking fallback step '{getattr(fb_step, 'name', '<unnamed>')}' after exception for '{getattr(step, 'name', '<unnamed>')}' attempt={attempt}/{total_attempts}"
            )
        except Exception:
            pass

        try:
            execute_fn = getattr(core, "execute", None)
            if not callable(execute_fn):
                raise TypeError("Executor core must provide execute()")
            fallback_result_sr = await execute_fn(
                step=fb_step,
                data=data,
                context=attempt_context,
                resources=attempt_resources,
                limits=limits,
                stream=stream,
                on_chunk=on_chunk,
                _fallback_depth=fallback_depth + 1,
            )
        except Exception as fb_exc:  # noqa: BLE001
            if getattr(fb_exc, "__class__", type(fb_exc)).__name__ == "InfiniteFallbackError":
                fb_txt = (
                    f"Fallback loop detected for step '{getattr(fb_step, 'name', '<unnamed>')}'"
                )
                return FallbackHandlingResult(
                    outcome=Failure(
                        error=fb_exc,
                        feedback=fb_txt,
                        step_result=StepResult(
                            name=_safe_step_name(step),
                            output=None,
                            success=False,
                            attempts=result.attempts,
                            latency_s=time_perf_ns_to_seconds(time_perf_ns() - start_ns),
                            token_counts=result.token_counts,
                            cost_usd=result.cost_usd,
                            feedback=fb_txt,
                            branch_context=None,
                            metadata_={"fallback_triggered": True},
                            step_history=[],
                        ),
                    ),
                    resources_closed=False,
                )
            return FallbackHandlingResult(
                outcome=Failure(
                    error=fb_exc,
                    feedback=f"Original error: {primary_feedback}; Fallback error: {fb_exc}",
                    step_result=StepResult(
                        name=_safe_step_name(step),
                        output=None,
                        success=False,
                        attempts=result.attempts,
                        latency_s=time_perf_ns_to_seconds(time_perf_ns() - start_ns),
                        token_counts=result.token_counts,
                        cost_usd=result.cost_usd,
                        feedback=f"Original error: {primary_feedback}; Fallback error: {fb_exc}",
                        branch_context=(
                            attempt_context if getattr(step, "updates_context", False) else None
                        ),
                        metadata_={
                            "fallback_triggered": True,
                            "original_error": primary_feedback,
                        },
                        step_history=[],
                    ),
                ),
                resources_closed=False,
            )

        from ....domain.models import StepOutcome as _StepOutcome

        if isinstance(fallback_result_sr, (StepResult, Success)):
            # Normalize to StepResult for metric processing
            final_sr: StepResult
            if isinstance(fallback_result_sr, Success):
                final_sr = fallback_result_sr.step_result
            else:
                final_sr = fallback_result_sr

            preserve_diagnostics = False
            try:
                cfg = getattr(step, "config", None)
                preserve_diagnostics = getattr(cfg, "preserve_fallback_diagnostics", False)
            except Exception:
                pass
            if final_sr.metadata_ is None:
                final_sr.metadata_ = {}
            final_sr.metadata_.update(
                {
                    "fallback_triggered": True,
                    "original_error": primary_feedback,
                    "preserve_fallback_diagnostics": preserve_diagnostics,
                }
            )

            if final_sr.success:
                try:
                    fb_tokens = int(final_sr.token_counts or 0)
                except Exception:
                    fb_tokens = 0
                try:
                    primary_unit_tokens = int(result.token_counts or 0)
                except Exception:
                    primary_unit_tokens = 0
                if primary_unit_tokens == 0:
                    try:
                        primary_unit_tokens = int(state.best_primary_tokens or 0)
                    except Exception:
                        primary_unit_tokens = 0
                if primary_unit_tokens == 0:
                    try:
                        primary_unit_tokens = int(state.primary_tokens_total or 0) // max(
                            1, total_attempts
                        )
                    except Exception:
                        primary_unit_tokens = 0
                primary_total = (
                    state.primary_tokens_total
                    if state.primary_tokens_total not in (None, 0)
                    else primary_unit_tokens * max(1, total_attempts)
                )
                fb_unit = fb_tokens if fb_tokens not in (None, 0) else primary_unit_tokens
                final_sr.token_counts = primary_total + fb_unit
                try:
                    final_sr.token_counts = int(final_sr.token_counts)
                except Exception:
                    pass
                try:
                    usage_meter = getattr(core, "_usage_meter", None)
                    add_fn = getattr(usage_meter, "add", None) if usage_meter is not None else None
                    if callable(add_fn):
                        await add_fn(
                            float(final_sr.cost_usd or 0.0),
                            int(state.prompt_tokens_latest or 0) + fb_tokens,
                            0,
                        )
                except Exception:
                    pass
                try:
                    fb_attempts = int(getattr(final_sr, "attempts", 1) or 1)
                    final_sr.attempts = int(total_attempts + fb_attempts)
                except Exception:
                    pass
                final_sr.feedback = final_sr.feedback if final_sr.feedback else None
                try:
                    fb_handler = getattr(core, "_fallback_handler", None)
                    reset_fn = (
                        getattr(fb_handler, "reset", None) if fb_handler is not None else None
                    )
                    if callable(reset_fn):
                        reset_fn()
                except Exception:
                    pass
                await close_resources(None)
                return FallbackHandlingResult(
                    outcome=Success(step_result=final_sr),
                    resources_closed=True,
                )

            # Fallback failed but returned a StepResult/Success
            primary_fb_failure = state.last_plugin_failure_feedback or primary_feedback
            combo_fb = f"Original error: {primary_fb_failure}; Fallback error: {final_sr.feedback}"
            await close_resources(attempt_exc)
            return FallbackHandlingResult(
                outcome=Failure(
                    error=attempt_exc,
                    feedback=combo_fb,
                    step_result=StepResult(
                        name=_safe_step_name(step),
                        output=None,
                        success=False,
                        attempts=result.attempts,
                        latency_s=time_perf_ns_to_seconds(time_perf_ns() - start_ns),
                        token_counts=(
                            (state.best_primary_tokens or state.primary_tokens_total)
                            or int(result.token_counts or 0)
                        )
                        + int(getattr(final_sr, "token_counts", 0) or 0),
                        cost_usd=float(getattr(final_sr, "cost_usd", 0.0) or 0.0),
                        feedback=combo_fb,
                        branch_context=(
                            pre_attempt_context
                            if getattr(step, "updates_context", False)
                            and hasattr(pre_attempt_context, "hitl_data")
                            else (
                                attempt_context if getattr(step, "updates_context", False) else None
                            )
                        ),
                        metadata_={
                            "fallback_triggered": True,
                            "original_error": primary_fb_failure,
                        },
                        step_history=[],
                    ),
                ),
                resources_closed=True,
            )

        if isinstance(fallback_result_sr, _StepOutcome):
            unwrap_fn = getattr(core, "_unwrap_outcome_to_step_result", None)
            if not callable(unwrap_fn):
                raise TypeError("Executor core must provide _unwrap_outcome_to_step_result()")
            sr_fb = unwrap_fn(fallback_result_sr, _safe_step_name(step))
            failure_fb = state.last_plugin_failure_feedback or primary_feedback
            combo_fb = f"Original error: {failure_fb}; Fallback error: {getattr(fallback_result_sr, 'feedback', 'Unknown')}"
            return FallbackHandlingResult(
                outcome=Failure(
                    error=attempt_exc,
                    feedback=combo_fb,
                    step_result=StepResult(
                        name=_safe_step_name(step),
                        output=None,
                        success=False,
                        attempts=result.attempts,
                        latency_s=time_perf_ns_to_seconds(time_perf_ns() - start_ns),
                        token_counts=int(result.token_counts or 0)
                        + int(getattr(sr_fb, "token_counts", 0) or 0),
                        cost_usd=float(getattr(sr_fb, "cost_usd", 0.0) or 0.0),
                        feedback=combo_fb,
                        branch_context=None,
                        metadata_={"fallback_triggered": True, "original_error": failure_fb},
                        step_history=[],
                    ),
                ),
                resources_closed=False,
            )

        await close_resources(attempt_exc)
        return FallbackHandlingResult(
            outcome=Failure(
                error=attempt_exc,
                feedback=primary_feedback,
                step_result=StepResult(
                    name=_safe_step_name(step),
                    output=None,
                    success=False,
                    attempts=result.attempts,
                    latency_s=time_perf_ns_to_seconds(time_perf_ns() - start_ns),
                    token_counts=state.best_primary_tokens or result.token_counts,
                    cost_usd=state.best_primary_cost_usd or result.cost_usd,
                    feedback=primary_feedback,
                    branch_context=(
                        pre_attempt_context
                        if getattr(step, "updates_context", False)
                        and hasattr(pre_attempt_context, "hitl_data")
                        else (attempt_context if getattr(step, "updates_context", False) else None)
                    ),
                    metadata_={},
                    step_history=[],
                ),
            ),
            resources_closed=True,
        )
