from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

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

    _value: Any

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - trivial
        if name in {"output", "result", "value"}:
            return self._value
        raise AttributeError(name)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self._value)

    def unwrap(self) -> Any:
        return self._value


class TemplateContextProxy:
    """Proxy base context with a fallback to steps outputs by name."""

    def __init__(
        self,
        base: Optional[Mapping[str, Any]] = None,
        *,
        steps: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._base: Mapping[str, Any] = base or {}
        self._steps: Mapping[str, Any] = steps or {}

    def __getattr__(self, name: str) -> Any:
        if name in self._base:
            return self._base[name]
        if name in self._steps:
            v = self._steps[name]
            return v if isinstance(v, StepValueProxy) else StepValueProxy(v)
        raise AttributeError(name)

    def __getitem__(self, key: str) -> Any:  # pragma: no cover - simple delegator
        if key in self._base:
            return self._base[key]
        if key in self._steps:
            v = self._steps[key]
            return v if isinstance(v, StepValueProxy) else StepValueProxy(v)
        raise KeyError(key)


def get_steps_map_from_context(context: Any) -> Dict[str, Any]:
    """Extract mapping of prior step outputs from context.scratchpad['steps'] when present."""
    try:
        scratchpad = getattr(context, "scratchpad", None)
        if isinstance(scratchpad, Mapping):
            steps = scratchpad.get("steps")
            if isinstance(steps, Mapping):
                return dict(steps)
    except Exception:
        pass
    return {}


def render_template(
    template: str,
    *,
    context: Optional[Mapping[str, Any]] = None,
    steps: Optional[Mapping[str, Any]] = None,
    previous_step: Any = None,
) -> str:
    """Helper for tests: render with AdvancedPromptFormatter using proxies.

    Supports dotted access and the variables: context, previous_step, steps.
    """
    try:
        from .prompting import AdvancedPromptFormatter
    except Exception:  # pragma: no cover - defensive
        # Fallback simple replacement if formatter import breaks in isolated tests
        return template

    steps_map: Dict[str, Any] = {}
    if steps:
        for k, v in steps.items():
            steps_map[k] = v if isinstance(v, StepValueProxy) else StepValueProxy(v)
    ctx_proxy = TemplateContextProxy(context, steps=steps_map)
    fmt = AdvancedPromptFormatter(template)
    return fmt.format(context=ctx_proxy, steps=steps_map, previous_step=previous_step)
