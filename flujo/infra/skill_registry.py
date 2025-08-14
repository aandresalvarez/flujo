from __future__ import annotations

from typing import Any, Callable, Dict, Optional, List, Literal


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
        # FSD-020 naming alias: arg_schema is accepted as alias of input_schema
        output_schema: Optional[dict[str, Any]] = None,
        capabilities: Optional[List[str]] = None,
        safety_level: Optional[Literal["none", "low", "medium", "high"]] = None,
        auth_required: Optional[bool] = None,
        auth_scope: Optional[str] = None,
        side_effects: Optional[bool] = None,
        arg_schema: Optional[dict[str, Any]] = None,
    ) -> None:
        # Prefer explicit input_schema; fall back to arg_schema for compatibility with FSD specs
        effective_input_schema: Optional[dict[str, Any]] = input_schema or arg_schema

        self._entries[id] = {
            "factory": factory,
            "description": description,
            "input_schema": effective_input_schema,
            "output_schema": output_schema,
            "capabilities": capabilities or [],
            "safety_level": safety_level or "none",
            "auth_required": bool(auth_required) if auth_required is not None else False,
            "auth_scope": auth_scope,
            "side_effects": bool(side_effects) if side_effects is not None else False,
        }

    def get(self, id: str) -> Optional[dict[str, Any]]:
        return self._entries.get(id)


_GLOBAL: Optional[SkillRegistry] = None


def get_skill_registry() -> SkillRegistry:
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = SkillRegistry()
    return _GLOBAL
