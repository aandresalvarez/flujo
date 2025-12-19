from __future__ import annotations

import inspect


async def aclose_if_possible(obj: object) -> None:
    """Best-effort `aclose()` for async generators/iterators that support it."""
    try:
        aclose = getattr(obj, "aclose", None)
        if callable(aclose):
            res = aclose()
            if inspect.isawaitable(res):
                await res
    except Exception:
        pass
