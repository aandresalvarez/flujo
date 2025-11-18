from __future__ import annotations

from typing import Any, Dict
from flujo.domain.base_model import BaseModel


async def init_counter(data: Any = None, *, context: BaseModel | None = None, **_: Any) -> Dict[str, Any]:
    """Initialize the counter in context to 1.

    Returns a mapping that `updates_context: true` steps will merge into the context.
    """
    return {"count": 1}


async def counter_agent(data: Any = None, *, context: BaseModel | None = None, **_: Any) -> Dict[str, Any]:
    """Stub agent that increments a count until 3, then finishes.

    - If count < 3: return action='ask', with next count and a question.
    - If count >= 3: return action='finish', keep count, and a final confirmation message.
    """
    c = 0
    try:
        if context is not None:
            c = int(getattr(context, "count", 0) or (context["count"] if isinstance(context, dict) and "count" in context else 0))
    except Exception:
        c = 0

    if c < 3:
        c_next = c + 1
        return {
            "action": "ask",
            "question": f"Question {c_next}?",
            "count": c_next,
        }

    return {
        "action": "finish",
        "question": "Done! Press Enter.",
        "count": c,
    }
