from __future__ import annotations

from typing import Any, Callable, Dict, Optional


class SkillRegistry:
    """Minimal registry for resolving skills/agents by ID.

    Stores factories or callables alongside optional metadata.
    """

    def __init__(self) -> None:
        self._entries: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        id: str,
        factory: Callable[..., Any] | Any,
        *,
        description: Optional[str] = None,
        input_schema: Optional[dict[str, Any]] = None,
        output_schema: Optional[dict[str, Any]] = None,
    ) -> None:
        self._entries[id] = {
            "factory": factory,
            "description": description,
            "input_schema": input_schema,
            "output_schema": output_schema,
        }

    def get(self, id: str) -> Optional[dict[str, Any]]:
        return self._entries.get(id)


_GLOBAL: Optional[SkillRegistry] = None


def get_skill_registry() -> SkillRegistry:
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = SkillRegistry()
    return _GLOBAL
