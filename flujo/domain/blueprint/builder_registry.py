from __future__ import annotations

from typing import TYPE_CHECKING, Callable, TypeAlias

if TYPE_CHECKING:
    from .loader_steps import AnyStep

# Define the builder function signature
BuilderFn: TypeAlias = Callable[..., "AnyStep"]


class BlueprintBuilderRegistry:
    """Registry for step builders used by the blueprint loader."""

    def __init__(self) -> None:
        self._builders: dict[str, BuilderFn] = {}

    def register(self, kind: str, builder: BuilderFn) -> None:
        """Register a builder function for a specific step kind."""
        self._builders[kind] = builder

    def get_builder(self, kind: str) -> BuilderFn | None:
        """Retrieve a builder function for a given step kind."""
        return self._builders.get(kind)


# Global registry instance
_registry = BlueprintBuilderRegistry()


def register_builder(kind: str, builder: BuilderFn) -> None:
    """Public API to register a builder."""
    _registry.register(kind, builder)


def get_builder(kind: str) -> BuilderFn | None:
    """Public API to get a builder."""
    return _registry.get_builder(kind)
