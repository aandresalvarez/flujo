from __future__ import annotations

from typing import Any, Dict, Optional, Type, Any as _Any

from flujo.exceptions import ConfigurationError
from flujo.domain.dsl.step import Step


# Internal registries
_step_type_registry: Dict[str, Type[Step[_Any, _Any]]] = {}
_policy_registry: Dict[Type[Step[_Any, _Any]], Any] = {}


def register_step_type(step_class: Type[Step[_Any, _Any]]) -> None:
    """Register a custom Step subclass by its declared kind.

    The class must:
    - Be a subclass of flujo.domain.dsl.Step
    - Expose a class-level attribute `kind: str`
    - Not conflict with an already-registered kind
    """
    if not isinstance(step_class, type) or not issubclass(step_class, Step):
        raise ConfigurationError("register_step_type: step_class must be a subclass of Step")
    kind = getattr(step_class, "kind", None)
    if not isinstance(kind, str) or not kind:
        raise ConfigurationError("register_step_type: step_class.kind must be a non-empty string")
    if kind in _step_type_registry:
        raise ConfigurationError(f"Step kind already registered: {kind}")
    _step_type_registry[kind] = step_class


def register_policy(step_class: Type[Step[_Any, _Any]], policy_instance: Any) -> None:
    """Register an execution policy instance for a Step subclass.

    The policy instance may be:
    - A callable that accepts an ExecutionFrame and returns an awaitable StepOutcome
    - An object exposing `execute(core, frame)` used by an adapter in ExecutorCore
    """
    if not isinstance(step_class, type) or not issubclass(step_class, Step):
        raise ConfigurationError("register_policy: step_class must be a subclass of Step")
    _policy_registry[step_class] = policy_instance


def unregister_policy(step_class: Type[Step[_Any, _Any]]) -> None:
    """Unregister an execution policy instance for a Step subclass (best-effort)."""
    if not isinstance(step_class, type) or not issubclass(step_class, Step):
        raise ConfigurationError("unregister_policy: step_class must be a subclass of Step")
    _policy_registry.pop(step_class, None)


def get_step_class(kind: str) -> Optional[Type[Step[_Any, _Any]]]:
    """Return the registered Step class for a kind string, if any."""
    return _step_type_registry.get(kind)


def get_registered_step_kinds() -> Dict[str, Type[Step[_Any, _Any]]]:
    """Expose a copy of the step-type registry for read-only use."""
    return dict(_step_type_registry)


def get_registered_policies() -> Dict[Type[Step[_Any, _Any]], Any]:
    """Expose a copy of the policy registry for initialization in the core."""
    return dict(_policy_registry)
