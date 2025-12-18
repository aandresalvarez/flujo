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
        self._framework_builders: dict[str, BuilderFn] = {}

    def register(self, kind: str, builder: BuilderFn) -> None:
        """Register a builder function for a specific step kind."""
        self._builders[kind] = builder

    def get_builder(self, kind: str) -> BuilderFn | None:
        """Retrieve a builder function for a given step kind."""
        builder = self._builders.get(kind)
        if builder is not None:
            return builder

        cached = self._framework_builders.get(kind)
        if cached is not None:
            return cached

        # Framework extensions: registered Step subclasses become YAML-loadable by `kind`.
        try:
            from ...framework import registry as _fwreg

            step_cls = _fwreg.get_step_class(kind)
        except Exception:
            step_cls = None

        if step_cls is None:
            return None

        from .loader_models import BlueprintError

        def _build_framework_step(
            model: object,
            *,
            yaml_path: str | None = None,
            **_kwargs: object,
        ) -> AnyStep:
            try:
                step_obj = step_cls.model_validate(model)
            except Exception as e:
                raise BlueprintError(
                    f"Failed to instantiate custom step for kind '{kind}': {e}"
                ) from e
            if yaml_path:
                try:
                    step_obj.meta["yaml_path"] = yaml_path
                except Exception:
                    pass
            return step_obj

        self._framework_builders[kind] = _build_framework_step
        return _build_framework_step


# Global registry instance
_registry = BlueprintBuilderRegistry()


def register_builder(kind: str, builder: BuilderFn) -> None:
    """Public API to register a builder."""
    _registry.register(kind, builder)


def get_builder(kind: str) -> BuilderFn | None:
    """Public API to get a builder."""
    return _registry.get_builder(kind)
