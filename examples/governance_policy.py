from __future__ import annotations

from flujo.application.core.governance_policy import GovernanceDecision, GovernancePolicy


class DenyIfContainsSecret(GovernancePolicy):
    """Example governance policy: deny when 'secret' appears in input text."""

    def __init__(self, keyword: str = "secret") -> None:
        self.keyword = keyword.lower()

    async def evaluate(
        self,
        *,
        core: object,
        step: object,
        data: dict[str, object] | None,
        context: object,
        resources: object,
    ) -> GovernanceDecision:
        text = str((data or {}).get("input", "")).lower()
        if self.keyword in text:
            return GovernanceDecision(
                allow=False,
                reason=f"Input contained forbidden keyword '{self.keyword}'",
            )
        return GovernanceDecision(allow=True, reason="OK")
