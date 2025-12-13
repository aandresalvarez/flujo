from __future__ import annotations

from typing import Any, List, Optional

from flujo.type_definitions.common import JSONObject

from flujo.domain.models import BaseModel as DomainBaseModel
from flujo.infra.skill_registry import get_skill_registry


async def context_set(
    path: str,
    value: Any,
    *,
    context: Optional[DomainBaseModel] = None,
) -> JSONObject:
    """Set a context field at the specified dot-separated path."""
    from flujo.utils.context import set_nested_context_field

    success = False
    if context is not None:
        try:
            set_nested_context_field(context, path, value)
            success = True
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(f"Failed to set context path '{path}': {e}")
    return {"path": path, "value": value, "success": success}


async def context_merge(
    path: str,
    value: JSONObject,
    *,
    context: Optional[DomainBaseModel] = None,
) -> JSONObject:
    """Merge a dictionary into the context at the specified path."""
    from flujo.utils.context import set_nested_context_field

    merged_keys: List[str] = []
    if context is not None and isinstance(value, dict):
        try:
            parts = path.split(".")
            target = context
            for part in parts:
                try:
                    target = getattr(target, part)
                except AttributeError:
                    if isinstance(target, dict) and part in target:
                        target = target[part]
                    else:
                        set_nested_context_field(context, path, {})
                        target = context
                        for p in parts:
                            if hasattr(target, p):
                                target = getattr(target, p)
                            elif isinstance(target, dict):
                                target = target[p]
                            else:
                                break

            if isinstance(target, dict):
                target.update(value)
                merged_keys = list(value.keys())
            else:
                for key, val in value.items():
                    try:
                        setattr(target, key, val)
                        merged_keys.append(key)
                    except (AttributeError, TypeError):
                        pass
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(f"Failed to merge into context path '{path}': {e}")
    return {"path": path, "merged_keys": merged_keys, "success": len(merged_keys) > 0}


async def context_get(
    path: str,
    default: Any = None,
    *,
    context: Optional[DomainBaseModel] = None,
) -> Any:
    """Get a value from the context at the specified dot-separated path."""
    if context is None:
        return default
    try:
        parts = path.split(".")
        target = context
        for part in parts:
            try:
                target = getattr(target, part)
            except AttributeError:
                if isinstance(target, dict) and part in target:
                    target = target[part]
                else:
                    return default
        return target
    except Exception:
        return default


def register_context_builtins() -> None:
    reg = get_skill_registry()
    reg.register(
        "flujo.builtins.context_set",
        lambda **_params: context_set,
        description="Set a context field at a dot-separated path (e.g., 'counter' or 'settings.theme')",
        arg_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "value": {},
            },
            "required": ["path", "value"],
        },
        side_effects=True,
    )
    reg.register(
        "flujo.builtins.context_merge",
        lambda **_params: context_merge,
        description="Merge a dictionary into context at a path (e.g., 'settings')",
        arg_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "value": {"type": "object"},
            },
            "required": ["path", "value"],
        },
        side_effects=True,
    )
    reg.register(
        "flujo.builtins.context_get",
        lambda **_params: context_get,
        description="Get a value from context at a dot-separated path with optional default",
        arg_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "default": {},
            },
            "required": ["path"],
        },
        side_effects=False,
    )


__all__ = ["context_get", "context_merge", "context_set", "register_context_builtins"]
