"""Unified async/sync bridge utilities.

This module provides a canonical way to run async coroutines from synchronous
contexts, handling the common case where code may already be running inside
an event loop.

The primary utility is `run_sync`, which safely executes async code by:
- Using `asyncio.run()` directly when no loop is running
- Spawning a daemon thread with a new loop when inside an existing loop

This prevents "Event loop is closed" errors and shutdown leaks that occur
with ad-hoc thread-based solutions.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Coroutine, TypeVar, cast

T = TypeVar("T")


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run async coroutine from sync context safely.

    Uses thread-based isolation when already inside an event loop,
    preventing "Event loop is closed" errors and shutdown leaks.

    This is the canonical way to run async code from sync in Flujo.

    Args:
        coro: The coroutine to execute.

    Returns:
        The result of the coroutine.

    Raises:
        Any exception raised by the coroutine.

    Example:
        >>> async def fetch_data():
        ...     return {"key": "value"}
        >>> result = run_sync(fetch_data())
        >>> print(result)
        {'key': 'value'}
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop - safe to use asyncio.run directly
        return asyncio.run(coro)

    # Already inside a loop - use thread isolation
    result: Any = None
    exc: BaseException | None = None

    def _target() -> None:
        nonlocal result, exc
        try:
            result = asyncio.run(coro)
        except BaseException as e:  # pragma: no cover - unlikely
            exc = e

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join()

    if exc:
        raise exc
    return cast(T, result)


__all__ = ["run_sync"]
