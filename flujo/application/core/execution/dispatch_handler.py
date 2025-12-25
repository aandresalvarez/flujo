from __future__ import annotations

from typing import TYPE_CHECKING

from ....domain.models import Failure, Paused, StepOutcome, StepResult, Success
from ....exceptions import (
    InfiniteFallbackError,
    InfiniteRedirectError,
    MissingAgentError,
    PausedException,
    PricingNotConfiguredError,
    UsageLimitExceededError,
)
from ..types import ExecutionFrame, TContext

if TYPE_CHECKING:
    from ..executor_core import ExecutorCore


class DispatchHandler:
    """Encapsulates policy dispatch + hydration persistence for ExecutorCore."""

    def __init__(self, core: "ExecutorCore[TContext]") -> None:
        self._core: "ExecutorCore[TContext]" = core

    async def dispatch(
        self, frame: ExecutionFrame[TContext], *, called_with_frame: bool
    ) -> StepOutcome[StepResult] | StepResult:
        """Dispatch via policy registry and handle control-flow outcomes."""
        step = frame.step
        step_name = self._core._safe_step_name(step)
        try:
            outcome = await self._core._dispatcher.dispatch(frame)
            await self._core._hydration_manager.persist_context(getattr(frame, "context", None))
            if called_with_frame:
                if isinstance(outcome, Paused):
                    raise PausedException(outcome.message)
                if isinstance(outcome, (StepOutcome, StepResult)):
                    return outcome
                raise TypeError(
                    f"Dispatcher returned unsupported type {type(outcome).__name__} for step '{step_name}'"
                )
            if isinstance(outcome, Success):
                return self._core._unwrap_outcome_to_step_result(outcome.step_result, step_name)
            if isinstance(outcome, Failure):
                return self._core._unwrap_outcome_to_step_result(outcome, step_name)
            if isinstance(outcome, Paused):
                raise PausedException(outcome.message)
            return self._core._unwrap_outcome_to_step_result(outcome, step_name)
        except InfiniteFallbackError:
            raise
        except PausedException:
            await self._core._hydration_manager.persist_context(getattr(frame, "context", None))
            # Control-flow exception: must propagate to the runner regardless of call mode.
            raise
        except (
            UsageLimitExceededError,
            PricingNotConfiguredError,
            MissingAgentError,
            InfiniteRedirectError,
        ) as e:
            raise e
        except Exception as e:  # pragma: no cover - mirrors core handler
            if not self._core.enable_optimized_error_handling:
                self._core._log_execution_error(step_name, e)
            failure_outcome = self._core._build_failure_outcome(
                step=step,
                frame=frame,
                exc=e,
                called_with_frame=called_with_frame,
            )
            if called_with_frame:
                return failure_outcome
            return self._core._unwrap_outcome_to_step_result(failure_outcome, step_name)
