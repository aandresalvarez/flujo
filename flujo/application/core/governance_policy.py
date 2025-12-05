from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from ...exceptions import ConfigurationError
from ...infra import telemetry


@dataclass(frozen=True)
class GovernanceDecision:
    allow: bool
    reason: str | None = None


class GovernancePolicy(Protocol):
    async def evaluate(
        self,
        *,
        core: Any,
        step: Any,
        data: Any,
        context: Any,
        resources: Any,
    ) -> GovernanceDecision: ...


class AllowAllGovernancePolicy:
    """Default policy that allows all agent executions."""

    async def evaluate(
        self,
        *,
        core: Any,
        step: Any,
        data: Any,
        context: Any,
        resources: Any,
    ) -> GovernanceDecision:
        return GovernanceDecision(allow=True)


class GovernanceEngine:
    """Evaluates governance policies before agent execution."""

    def __init__(self, policies: Sequence[GovernancePolicy] | None = None) -> None:
        self._policies: Sequence[GovernancePolicy] = (
            policies
            if policies is not None and len(policies) > 0
            else (AllowAllGovernancePolicy(),)
        )

    async def enforce(
        self,
        *,
        core: Any,
        step: Any,
        data: Any,
        context: Any,
        resources: Any,
    ) -> None:
        for policy in self._policies:
            decision = await policy.evaluate(
                core=core, step=step, data=data, context=context, resources=resources
            )
            if not decision.allow:
                telemetry.logfire.warning(
                    f"[Governance] Deny agent execution "
                    f"(step={getattr(step, 'name', '<unnamed>')}, "
                    f"reason={decision.reason or 'unspecified'})"
                )
                raise ConfigurationError(
                    decision.reason or "Agent execution blocked by governance policy"
                )
