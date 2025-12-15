from __future__ import annotations

import inspect
from typing import Any


async def aclose_if_possible(obj: Any) -> None:
    """Best-effort `aclose()` for async generators/iterators that support it."""
    try:
        aclose = getattr(obj, "aclose", None)
        if callable(aclose):
            res = aclose()
            if inspect.isawaitable(res):
                await res
    except Exception:
        pass
