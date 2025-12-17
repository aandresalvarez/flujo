from __future__ import annotations

from typing import Protocol, runtime_checkable, Any

from flujo.domain.models import BaseModel
from .types import ContextT


class PluginOutcome(BaseModel):
    """Result returned by a validation plugin."""

    # model_config inherited from BaseModel

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


@runtime_checkable
class ContextAwarePluginProtocol(Protocol[ContextT]):
    """A protocol for plugins that are aware of a specific pipeline context type."""

    __context_aware__: bool = True

    async def validate(
        self,
        data: dict[str, Any],
        *,
        context: ContextT,
        **kwargs: Any,
    ) -> PluginOutcome: ...


# Explicit exports
__all__ = [
    "PluginOutcome",
    "ValidationPlugin",
    "ContextAwarePluginProtocol",
]
