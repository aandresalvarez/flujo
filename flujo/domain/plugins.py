from __future__ import annotations

from typing import Protocol, runtime_checkable, Any, Dict, Type

from flujo.domain.models import BaseModel
from .types import ContextT
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


plugin_registry: Dict[str, Type[ValidationPlugin]] = {}


def register_plugin(cls: Type[ValidationPlugin]) -> Type[ValidationPlugin]:
    """Register a ValidationPlugin class for IR loading."""
    plugin_registry[cls.__name__] = cls
    return cls


# Explicit exports
__all__ = [
    "PluginOutcome",
    "ValidationPlugin",
    "ContextAwarePluginProtocol",
    "plugin_registry",
    "register_plugin",
]
