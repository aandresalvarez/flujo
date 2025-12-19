"""Quota management for proactive resource budgeting."""

from __future__ import annotations

import contextvars
from typing import Optional, Tuple

from ....domain.models import Quota, QuotaExceededError, UsageEstimate, UsageLimits
from ....exceptions import UsageLimitExceededError


def build_root_quota(limits: Optional[UsageLimits]) -> Optional[Quota]:
    """Build the root quota from usage limits.

    This is the single boundary where legacy `UsageLimits` translate into the quota system.
    When no limits are provided, return `None` to indicate "unlimited" for callers that use
    `None` as the sentinel.
    """
    if limits is None:
        return None
    cost = (
        float(limits.total_cost_usd_limit)
        if limits.total_cost_usd_limit is not None
        else float("inf")
    )
    # Use a large sentinel for "unlimited" tokens when not provided.
    tokens = int(limits.total_tokens_limit) if limits.total_tokens_limit is not None else 2**31 - 1
    return Quota(cost, tokens)


class QuotaManager:
    """Manages quota lifecycle: creation, reservation, reconciliation."""

    def __init__(self, limits: Optional[UsageLimits] = None) -> None:
        self._limits = limits
        self._root_quota: Optional[Quota] = None
        self._current_quota_var: contextvars.ContextVar[Optional[Quota]] = contextvars.ContextVar(
            "quota_manager_current_quota", default=None
        )

    def create_root_quota(self) -> Quota:
        """Create the root quota from usage limits."""
        root = build_root_quota(self._limits)
        if root is None:
            root = Quota(float("inf"), 2**31 - 1)
        self._root_quota = root
        return root

    def get_current_quota(self) -> Optional[Quota]:
        """Get the quota from the current async context."""
        return self._current_quota_var.get()

    def set_current_quota(self, quota: Optional[Quota]) -> contextvars.Token[Optional[Quota]]:
        """Set the quota for the current async context."""
        return self._current_quota_var.set(quota)

    def reserve(self, estimate: UsageEstimate) -> bool:
        """Reserve resources from current quota. Returns True if successful."""
        quota = self.get_current_quota()
        if quota is None:
            return True  # No quota = unlimited
        return quota.reserve(estimate)

    def reconcile(
        self,
        estimate: UsageEstimate,
        actual: UsageEstimate,
        *,
        limits: Optional[UsageLimits] = None,
    ) -> None:
        """Reconcile estimated vs actual usage after execution."""
        quota = self.get_current_quota()
        if quota is not None:
            try:
                quota.reclaim(estimate, actual)
            except QuotaExceededError as e:
                try:
                    from .usage_messages import format_reservation_denial

                    effective_limits = limits if limits is not None else self._limits
                    denial = format_reservation_denial(
                        UsageEstimate(cost_usd=e.extra_cost_usd, tokens=e.extra_tokens),
                        effective_limits,
                        remaining=(e.remaining_cost_usd, e.remaining_tokens),
                    )
                    raise UsageLimitExceededError(denial.human) from None
                except UsageLimitExceededError:
                    raise
                except Exception:
                    raise UsageLimitExceededError("Insufficient quota") from None

    def split_for_parallel(self, n: int) -> list[Quota]:
        """Split current quota for parallel branches."""
        quota = self.get_current_quota()
        if quota is None:
            return [Quota(float("inf"), 2**31 - 1) for _ in range(n)]
        return quota.split(n)

    def get_remaining(self) -> Tuple[float, int]:
        """Get remaining (cost, tokens) from current quota."""
        quota = self.get_current_quota()
        if quota is None:
            return (float("inf"), 2**31 - 1)
        return quota.get_remaining()
