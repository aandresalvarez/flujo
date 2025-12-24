from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from ....domain.models import BackgroundLaunched, Failure, Paused, StepOutcome, StepResult, Success
from ....exceptions import MissingAgentError, PausedException
from ....infra import telemetry

if TYPE_CHECKING:
    from ..executor_core import ExecutorCore
    from ..types import ExecutionFrame
    from ..types import TContext


class ResultHandler:
    """Handles cache persistence and exception wrapping for ExecutorCore."""

    def __init__(self, core: "ExecutorCore[TContext]") -> None:
        self._core: "ExecutorCore[TContext]" = core

    def handle_missing_agent_exception(
        self, err: MissingAgentError, step: object, *, called_with_frame: bool
    ) -> StepOutcome[StepResult] | StepResult:
        """Optimized handling for MissingAgentError when configured."""
        if not self._core.enable_optimized_error_handling:
            raise err
        if getattr(step.__class__, "__name__", "") == "Step":
            raise err
        safe_name = self._core._safe_step_name(step)
        result = StepResult(
            name=safe_name,
            output=None,
            success=False,
            attempts=0,
            latency_s=0.0,
            token_counts=0,
            cost_usd=0.0,
            feedback=str(err) if str(err) else "Missing agent configuration",
            branch_context=None,
            metadata_={
                "optimized_error_handling": True,
                "error_type": type(err).__name__,
            },
            step_history=[],
        )
        try:
            telemetry.logfire.warning(
                f"ExecutorCore handled missing agent for step '{safe_name}' gracefully"
            )
        except Exception as exc:
            try:
                telemetry.logfire.debug(
                    f"Shadow evaluator scheduling failed for step '{getattr(step, 'name', '<unnamed>')}': {exc}"
                )
            except Exception:
                pass
        if called_with_frame:
            return Failure(error=err, feedback=result.feedback, step_result=result)
        return result

    def unwrap_outcome_to_step_result(
        self, outcome: StepOutcome[StepResult] | StepResult, step_name: str
    ) -> StepResult:
        """Normalize any StepOutcome/StepResult into a StepResult (propagates Paused)."""
        # Already a StepResult
        if isinstance(outcome, StepResult):
            # Restore diagnostic feedback if explicitly requested via metadata (see StepConfig)
            try:
                md = getattr(outcome, "metadata_", None)
                if isinstance(md, dict) and md.get("preserve_fallback_diagnostics") is True:
                    if outcome.success and (outcome.feedback is None):
                        original_error = md.get("original_error")
                        base_msg = (
                            f"Primary agent failed: {original_error}"
                            if original_error
                            else "Primary agent failed"
                        )
                        outcome.feedback = base_msg
            except Exception:
                pass

        def _unwrap_sr(obj: object) -> StepResult:
            try:
                from ....domain.models import (
                    StepOutcome as _StepOutcome,
                    Success as _Success,
                    Failure as _Failure,
                )

                if isinstance(obj, _StepOutcome):
                    if (
                        isinstance(obj, (_Success, _Failure))
                        and hasattr(obj, "step_result")
                        and obj.step_result is not None
                    ):
                        return obj.step_result
            except Exception:
                pass
            if isinstance(obj, StepResult):
                return obj
            return StepResult(
                name="unknown",
                output=str(obj),
                success=False,
                attempts=1,
                latency_s=0.0,
                token_counts={"total": 0},
                cost_usd=0.0,
                feedback=f"Could not unwrap: {type(obj).__name__}",
            )

        res = None
        if isinstance(outcome, StepResult):
            res = outcome
        elif isinstance(outcome, Success):
            res = outcome.step_result
        elif isinstance(outcome, Failure):
            if outcome.step_result is not None:
                res = outcome.step_result
            else:
                res = StepResult(
                    name=step_name,
                    output=None,
                    success=False,
                    feedback=outcome.feedback
                    or (str(outcome.error) if outcome.error is not None else None),
                )
        elif isinstance(outcome, Paused):
            raise PausedException(outcome.message)
        elif isinstance(outcome, BackgroundLaunched):
            res = StepResult(
                name=step_name,
                output=None,
                success=True,
                feedback=f"Launched in background (task_id={outcome.task_id})",
                metadata_={"background_task_id": outcome.task_id},
            )
        else:
            res = StepResult(
                name=step_name,
                output=None,
                success=False,
                feedback=f"Unsupported outcome type: {type(outcome).__name__}",
            )

        # Final safety check for metadata_
        if res is not None:
            if getattr(res, "metadata_", None) is None:
                try:
                    res.metadata_ = {}
                except Exception:
                    pass
        return res

    async def persist_and_finalize(
        self,
        *,
        step: object,
        result: StepResult,
        cache_key: Optional[str],
        called_with_frame: bool,
        frame: ExecutionFrame[TContext] | None = None,
    ) -> StepOutcome[StepResult] | StepResult:
        """Persist cache if applicable and return standardized result."""
        await self._core._cache_manager.maybe_persist_step_result(
            step, result, cache_key, ttl_s=3600
        )
        try:
            shadow_eval = getattr(self._core, "_shadow_evaluator", None)
            if shadow_eval is not None:
                shadow_eval.maybe_schedule(core=self._core, step=step, result=result, frame=frame)
        except Exception as exc:
            try:
                telemetry.logfire.debug(
                    f"Memory indexing failed for step '{getattr(step, 'name', '<unnamed>')}': {exc}"
                )
            except Exception:
                pass
        try:
            memory_manager = getattr(self._core, "_memory_manager", None)
            if memory_manager is not None:
                # Best-effort indexing; non-blocking if manager uses background tasks
                await memory_manager.index_step_output(
                    step_name=self._core._safe_step_name(step),
                    result=result,
                    context=getattr(frame, "context", None) if frame is not None else None,
                )
        except Exception:
            pass
        if called_with_frame:
            return Success(step_result=result)
        return result

    def handle_unexpected_exception(
        self,
        *,
        step: object,
        frame: ExecutionFrame[TContext],
        exc: Exception,
        called_with_frame: bool,
    ) -> StepOutcome[StepResult] | StepResult:
        """Log and wrap unexpected exceptions into Failure/StepResult."""
        step_name = self._core._safe_step_name(step)
        if not self._core.enable_optimized_error_handling:
            self._core._log_execution_error(step_name, exc)
        failure_outcome = self._core._build_failure_outcome(
            step=step,
            frame=frame,
            exc=exc,
            called_with_frame=called_with_frame,
        )
        if called_with_frame:
            return failure_outcome
        return self._core._unwrap_outcome_to_step_result(failure_outcome, step_name)

    async def maybe_use_cache(
        self, frame: ExecutionFrame[TContext], *, called_with_frame: bool
    ) -> tuple[Optional[StepOutcome[StepResult] | StepResult], Optional[str]]:
        """Return cached outcome if present and compute cache key when enabled."""
        cache_key: Optional[str] = None
        try:
            cached_outcome = await self._core._cache_manager.maybe_return_cached(
                frame, called_with_frame=called_with_frame
            )
            if cached_outcome is not None:
                return cached_outcome, None
            if self._core._enable_cache:
                cache_key = self._core._cache_key(frame)
        except Exception as e:  # pragma: no cover - defensive log path
            try:
                telemetry.logfire.warning(
                    f"Cache error for step {getattr(frame.step, 'name', '<unnamed>')}: {e}"
                )
            except Exception:
                pass
            cache_key = None
        return None, cache_key
