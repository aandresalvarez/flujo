from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

__all__ = [
    "StepValueProxy",
    "TemplateContextProxy",
    "get_steps_map_from_context",
    "render_template",
]


@dataclass(frozen=True)
class StepValueProxy:
    """Expose common aliases for a step's value.

    - .output/.result/.value all return the underlying value
    - str(proxy) stringifies to the value for convenience
    """

    _value: object

    def __getattr__(self, name: str) -> object:  # pragma: no cover - trivial
        if name in {"output", "result", "value"}:
            return self._value
        raise AttributeError(name)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self._value)

    def unwrap(self) -> object:
        return self._value


class TemplateContextProxy:
    """Proxy base context with a fallback to steps outputs by name.

    Supports both mapping-like and attribute-style access on the base context.
    """

    def __init__(
        self,
        base: object | None = None,
        *,
        steps: Mapping[str, object] | None = None,
    ) -> None:
        self._base: object = base if base is not None else {}
        self._steps: Mapping[str, object] = steps or {}

    def __getattr__(self, name: str) -> object:
        # Mapping-style lookup
        try:
            if isinstance(self._base, Mapping) and name in self._base:
                return self._base[name]
        except Exception:
            pass
        # Attribute-style lookup
        try:
            val = getattr(self._base, name, None)
            if val is not None:
                return val
        except Exception:
            pass
        # Fallback to prior step outputs by name
        if name in self._steps:
            v = self._steps[name]
            return v if isinstance(v, StepValueProxy) else StepValueProxy(v)
        raise AttributeError(name)

    def __getitem__(self, key: str) -> object:  # pragma: no cover - simple delegator
        try:
            if isinstance(self._base, Mapping) and key in self._base:
                return self._base[key]
        except Exception:
            pass
        try:
            val = getattr(self._base, key, None)
            if val is not None:
                return val
        except Exception:
            pass
        if key in self._steps:
            v = self._steps[key]
            return v if isinstance(v, StepValueProxy) else StepValueProxy(v)
        raise KeyError(key)


def get_steps_map_from_context(context: object) -> dict[str, object]:
    """Extract mapping of prior step outputs from context.step_outputs when present."""
    try:
        # Prefer typed field if available
        if hasattr(context, "step_outputs"):
            outputs = getattr(context, "step_outputs")
            if isinstance(outputs, Mapping):
                return dict(outputs)
    except Exception:
        pass
    return {}


def render_template(
    template: str,
    *,
    context: Mapping[str, object] | None = None,
    steps: Mapping[str, object] | None = None,
    previous_step: object = None,
) -> str:
    """Helper for tests: render with AdvancedPromptFormatter using proxies.

    Supports dotted access and the variables: context, previous_step, steps.
    """
    try:
        from .prompting import AdvancedPromptFormatter
    except Exception:  # pragma: no cover - defensive
        # Fallback simple replacement if formatter import breaks in isolated tests
        return template

    steps_map: dict[str, object] = {}
    if steps:
        for k, v in steps.items():
            steps_map[k] = v if isinstance(v, StepValueProxy) else StepValueProxy(v)
    ctx_proxy = TemplateContextProxy(context, steps=steps_map)
    fmt = AdvancedPromptFormatter(template)
    return fmt.format(context=ctx_proxy, steps=steps_map, previous_step=previous_step)
