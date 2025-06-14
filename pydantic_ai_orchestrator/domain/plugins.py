from __future__ import annotations

from typing import Protocol, runtime_checkable, Any
from pydantic import BaseModel
from .agent_protocol import AgentProtocol


class PluginOutcome(BaseModel):
    """Result returned by a validation plugin."""

    model_config = {"arbitrary_types_allowed": True}

    ok: bool
    new_solution: Any | None = None
    new_feedback: str | None = None
    redirect_agent: AgentProtocol | None = None


@runtime_checkable
class ValidationPlugin(Protocol):
    """Protocol that all validation plugins must implement."""

    async def validate(self, data: dict[str, Any]) -> PluginOutcome:  # pragma: no cover - signature only
        ...
