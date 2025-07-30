"""Usage limit checking and enforcement for pipeline execution."""

from __future__ import annotations

from typing import Any, Optional, TypeVar, Generic

from ...domain.models import BaseModel, PipelineResult, UsageLimits
from ...exceptions import UsageLimitExceededError
from ...infra import telemetry

ContextT = TypeVar("ContextT", bound=BaseModel)


class UsageGovernor(Generic[ContextT]):
    """Manages usage limits and cost tracking during pipeline execution."""

    def __init__(self, usage_limits: Optional[UsageLimits] = None) -> None:
        self.usage_limits = usage_limits

    def check_usage_limits_efficient(
        self,
        current_total_cost: float,
        current_total_tokens: int,
        step_cost: float,
        step_tokens: int,
        span: Any | None,
    ) -> bool:
        """
        ✅ NEW: Efficiently checks limits using running totals.
        Returns True if a limit is breached, False otherwise.
        """
        if self.usage_limits is None:
            return False

        # Check cost limits
        if (
            self.usage_limits.total_cost_usd_limit is not None
            and (current_total_cost + step_cost) > self.usage_limits.total_cost_usd_limit
        ):
            return True

        # Check token limits
        if (
            self.usage_limits.total_tokens_limit is not None
            and (current_total_tokens + step_tokens) > self.usage_limits.total_tokens_limit
        ):
            return True

        return False

    def check_usage_limits(
        self,
        pipeline_result: PipelineResult[ContextT],
        span: Any | None,
    ) -> None:
        """
        ✅ REFACTORED: This method now raises the exception but relies on the caller
        to provide the complete PipelineResult.
        """
        if self.usage_limits is None:
            return

        # Check cost limits
        if (
            self.usage_limits.total_cost_usd_limit is not None
            and pipeline_result.total_cost_usd > self.usage_limits.total_cost_usd_limit
        ):
            error = UsageLimitExceededError(
                f"Cost limit of ${self.usage_limits.total_cost_usd_limit} exceeded",
                pipeline_result,
            )
            if span is not None:
                try:
                    span.record_exception(error)
                except AttributeError:
                    # Mock spans may not have record_exception
                    pass
            raise error

        # Check token limits
        total_tokens = sum(
            getattr(step, "token_counts", 0) for step in pipeline_result.step_history
        )
        if (
            self.usage_limits.total_tokens_limit is not None
            and total_tokens > self.usage_limits.total_tokens_limit
        ):
            error = UsageLimitExceededError(
                f"Token limit of {self.usage_limits.total_tokens_limit} exceeded",
                pipeline_result,
            )
            if span is not None:
                try:
                    span.record_exception(error)
                except AttributeError:
                    # Mock spans may not have record_exception
                    pass
            raise error

    def update_telemetry_span(
        self,
        span: Any | None,
        pipeline_result: PipelineResult[ContextT],
    ) -> None:
        """Update telemetry span with usage metrics."""
        if span is None:
            return

        try:
            span.set_attribute("flujo.total_cost_usd", pipeline_result.total_cost_usd)
            span.set_attribute("flujo.step_count", len(pipeline_result.step_history))

            total_tokens = sum(
                getattr(step, "token_counts", 0) for step in pipeline_result.step_history
            )
            span.set_attribute("flujo.total_tokens", total_tokens)

            if self.usage_limits is not None:
                if self.usage_limits.total_cost_usd_limit is not None:
                    span.set_attribute(
                        "flujo.cost_limit_usd", self.usage_limits.total_cost_usd_limit
                    )
                if self.usage_limits.total_tokens_limit is not None:
                    span.set_attribute("flujo.token_limit", self.usage_limits.total_tokens_limit)

        except Exception as e:
            telemetry.logfire.error(f"Error setting usage span attributes: {e}")
