"""Unified async/sync bridge utilities.

Flujo's sync entrypoints are intentionally *not* callable from a thread that already has a
running event loop. In those environments (e.g., Jupyter, FastAPI), prefer the async APIs.
"""

from __future__ import annotations

import anyio
import asyncio
from typing import Coroutine, TypeVar

T = TypeVar("T")


def run_sync(coro: Coroutine[object, object, T], *, running_loop_error: str | None = None) -> T:
    """Run an async coroutine from synchronous code using anyio.

    Raises:
        TypeError: If called from a running event loop thread.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, safe to run
        return anyio.run(lambda: coro)

    # We were handed a coroutine object that will never be awaited in this thread.
    # Close it to avoid "coroutine was never awaited" warnings.
    coro.close()

    raise TypeError(
        running_loop_error
        or "run_sync() cannot be called from a running event loop thread. Use async APIs (await) instead."
    )


__all__ = ["run_sync"]
