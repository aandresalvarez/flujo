from __future__ import annotations

from typing import Protocol, runtime_checkable, Any
from pydantic import BaseModel


class PluginOutcome(BaseModel):
    """Result returned by a validation plugin."""

    model_config = {"arbitrary_types_allowed": True}

    success: bool
    feedback: str | None = None
    redirect_to: Any | None = None
    new_solution: Any | None = None


@runtime_checkable
class ValidationPlugin(Protocol):
    """Protocol that all validation plugins must implement."""

    async def validate(
        self, data: dict[str, Any]
    ) -> PluginOutcome:  # pragma: no cover - protocol signature only, cannot be covered by tests
        ...


# Explicit exports
__all__ = ["PluginOutcome", "ValidationPlugin"]
