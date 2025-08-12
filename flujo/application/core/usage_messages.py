from __future__ import annotations

from typing import Final

from flujo.domain.models import UsageEstimate, UsageLimits
from flujo.utils.formatting import format_cost


class ReservationFailureMessage:
    def __init__(self, human: str, code: str) -> None:
        self.human: str = human
        self.code: str = code


_CODE_COST: Final[str] = "COST_LIMIT_EXCEEDED"
_CODE_TOKENS: Final[str] = "TOKEN_LIMIT_EXCEEDED"


def format_reservation_denial(
    estimate: UsageEstimate,
    limits: UsageLimits | None,
) -> ReservationFailureMessage:
    """Translate a reservation denial into a legacy-compatible message.

    Rules (kept stable for test string equality):
    - Prefer cost message when cost estimate exceeds configured cost limit.
    - Else prefer token message when token estimate exceeds configured token limit.
    - When both are exceeded, favor the most constrained resource by remaining ratio;
      if limits are missing, default to cost-style message.
    - If no limits provided, return a generic insufficient quota message.
    """

    # No configured limits → fall back to generic message
    if limits is None or (
        limits.total_cost_usd_limit is None and limits.total_tokens_limit is None
    ):
        return ReservationFailureMessage("Insufficient quota", _CODE_COST)

    cost_limit = limits.total_cost_usd_limit
    token_limit = limits.total_tokens_limit

    cost_exceeded = cost_limit is not None and float(estimate.cost_usd) > float(cost_limit)
    token_exceeded = token_limit is not None and int(estimate.tokens) > int(token_limit)

    # If only one dimension exceeded, format accordingly
    if cost_exceeded and not token_exceeded:
        assert cost_limit is not None  # We know it's not None because cost_exceeded is True
        return ReservationFailureMessage(
            f"Cost limit of ${format_cost(cost_limit)} exceeded", _CODE_COST
        )
    if token_exceeded and not cost_exceeded:
        assert token_limit is not None  # We know it's not None because token_exceeded is True
        return ReservationFailureMessage(f"Token limit of {token_limit} exceeded", _CODE_TOKENS)

    # If both exceeded, prefer cost-style message for legacy compatibility
    if cost_exceeded and token_exceeded:
        assert cost_limit is not None  # We know it's not None because cost_exceeded is True
        return ReservationFailureMessage(
            f"Cost limit of ${format_cost(cost_limit)} exceeded", _CODE_COST
        )

    # If neither exceeded against configured limits, still return generic
    return ReservationFailureMessage("Insufficient quota", _CODE_COST)
