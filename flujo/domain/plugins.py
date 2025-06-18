from __future__ import annotations

from typing import Protocol, runtime_checkable, Any
from flujo.domain.models import BaseModel
from pydantic import ConfigDict
from typing import ClassVar


class PluginOutcome(BaseModel):
    """Result returned by a validation plugin."""

    model_config: ClassVar[ConfigDict] = {"arbitrary_types_allowed": True}

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
