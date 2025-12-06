from __future__ import annotations

from dataclasses import dataclass

from flujo.application.core.governance_policy import GovernanceDecision, GovernancePolicy


@dataclass
class DenyIfContainsSecret(GovernancePolicy):
    """Example governance policy: deny when 'secret' appears in input text."""

    def __init__(self, keyword: str = "secret") -> None:
        self.keyword = keyword.lower()

    async def evaluate(self, payload: dict[str, object]) -> GovernanceDecision:
        text = str(payload.get("input", "")).lower()
        if self.keyword in text:
            return GovernanceDecision(
                allowed=False,
                reason=f"Input contained forbidden keyword '{self.keyword}'",
            )
        return GovernanceDecision(allowed=True, reason="OK")

