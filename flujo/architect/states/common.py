from __future__ import annotations

from typing import Any, Awaitable, Callable, List, Optional

from flujo.domain.base_model import BaseModel
from flujo.domain.interfaces import (
    SkillResolver,
    TelemetrySink,
    get_skill_resolver,
    get_telemetry_sink,
)
from flujo.type_definitions.common import JSONObject


def telemetry() -> TelemetrySink:
    return get_telemetry_sink()


def skill_resolver() -> SkillResolver:
    return get_skill_resolver()


def normalize_name_from_goal(goal: Optional[str]) -> str:
    safe_name = "generated_pipeline"
    try:
        g = (goal or "").strip()
        if g:
            import re as _re

            norm = _re.sub(r"[^A-Za-z0-9\s]+", "", g)[:40].strip().lower()
            if norm:
                safe_name = ("_".join(norm.split()) or safe_name)[:40]
    except Exception:
        pass
    return safe_name


async def goto(state: str, context: BaseModel | None = None) -> JSONObject:
    """Set next_state in typed context for SM transitions."""
    try:
        try:
            telemetry().info(f"[ArchitectSM] goto -> {state}")
        except Exception:
            pass
        return {"next_state": state}
    except Exception:
        return {"next_state": state}


def make_transition_guard(target_state: str) -> Callable[[Any], Awaitable[JSONObject]]:
    async def _guard(_x: Any = None, context: BaseModel | None = None) -> JSONObject:
        """Force next_state to target_state unconditionally to break stale loops."""
        try:
            telemetry().info(f"[ArchitectSM] guard -> forcing next_state={target_state}")
        except Exception:
            pass
        return {"next_state": target_state}

    return _guard


async def trace_next_state(_x: Any = None, context: BaseModel | None = None) -> JSONObject:
    """Pure observer of next_state; does not modify context."""
    try:
        ns = getattr(context, "next_state", None) if context is not None else None
        try:
            telemetry().info(f"[ArchitectSM] trace next_state={ns}")
        except Exception:
            pass
    except Exception:
        pass
    return {}


def skill_available(skill_id: str, *, available: Optional[List[JSONObject]]) -> bool:
    try:
        if isinstance(available, list):
            found = any(isinstance(x, dict) and x.get("id") == skill_id for x in available)
            if found:
                return True
    except Exception:
        pass
    try:
        entry = skill_resolver().get(skill_id, scope=None)
        if isinstance(entry, dict):
            return bool(entry)
        return entry is not None
    except Exception:
        return False
