"""Quota management for proactive resource budgeting."""

from __future__ import annotations
import contextvars
from typing import Optional, Tuple
from ...domain.models import Quota, UsageEstimate, UsageLimits

# Propagate quota across async calls
CURRENT_QUOTA: contextvars.ContextVar[Optional[Quota]] = contextvars.ContextVar(
    "CURRENT_QUOTA", default=None
)


class QuotaManager:
    """Manages quota lifecycle: creation, reservation, reconciliation."""

    def __init__(self, limits: Optional[UsageLimits] = None) -> None:
        self._limits = limits
        self._root_quota: Optional[Quota] = None

    def create_root_quota(self) -> Quota:
        """Create the root quota from usage limits."""
        cost = float("inf")
        tokens = 2**31 - 1  # Max int for "unlimited"

        if self._limits:
            if self._limits.total_cost_usd_limit is not None:
                cost = self._limits.total_cost_usd_limit
            if self._limits.total_tokens_limit is not None:
                tokens = self._limits.total_tokens_limit

        self._root_quota = Quota(cost, tokens)
        return self._root_quota

    def get_current_quota(self) -> Optional[Quota]:
        """Get the quota from the current async context."""
        return CURRENT_QUOTA.get()

    def set_current_quota(self, quota: Optional[Quota]) -> contextvars.Token[Optional[Quota]]:
        """Set the quota for the current async context."""
        return CURRENT_QUOTA.set(quota)

    def reserve(self, estimate: UsageEstimate) -> bool:
        """Reserve resources from current quota. Returns True if successful."""
        quota = self.get_current_quota()
        if quota is None:
            return True  # No quota = unlimited
        return quota.reserve(estimate)

    def reconcile(self, estimate: UsageEstimate, actual: UsageEstimate) -> None:
        """Reconcile estimated vs actual usage after execution."""
        quota = self.get_current_quota()
        if quota is not None:
            quota.reclaim(estimate, actual)

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
