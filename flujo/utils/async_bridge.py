"""Unified async/sync bridge utilities.

This module provides a canonical way to run async coroutines from synchronous contexts.

Key goals:
- Avoid per-call thread spawning (which can leak and cause teardown flakiness)
- Provide a predictable implementation for "run coroutine from sync while a loop is already running"

Implementation:
- If no event loop is running in the current thread: use `asyncio.run()`
- If an event loop *is* running: execute the coroutine on a shared `anyio` `BlockingPortal`
  (a dedicated background event loop thread) and block until completion.
"""

from __future__ import annotations

import atexit
import asyncio
import threading
from contextlib import AbstractContextManager
from typing import Any, Coroutine, TypeVar, cast

from anyio.from_thread import BlockingPortal, start_blocking_portal

T = TypeVar("T")

_PORTAL_LOCK = threading.Lock()
_PORTAL_MANAGER: AbstractContextManager[BlockingPortal] | None = None
_PORTAL: BlockingPortal | None = None


def _get_blocking_portal() -> BlockingPortal:
    global _PORTAL_MANAGER, _PORTAL
    with _PORTAL_LOCK:
        if _PORTAL is not None:
            return _PORTAL
        _PORTAL_MANAGER = start_blocking_portal()
        _PORTAL = _PORTAL_MANAGER.__enter__()
        return _PORTAL


def _shutdown_portal() -> None:
    global _PORTAL_MANAGER, _PORTAL
    with _PORTAL_LOCK:
        manager = _PORTAL_MANAGER
        _PORTAL = None
        _PORTAL_MANAGER = None

    if manager is not None:
        try:
            manager.__exit__(None, None, None)
        except Exception:
            pass


atexit.register(_shutdown_portal)


async def _await_coro(coro: Coroutine[Any, Any, T]) -> T:
    return await coro


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run async coroutine from sync context safely.

    This is the canonical way to run async code from sync in Flujo.

    Args:
        coro: The coroutine to execute.

    Returns:
        The result of the coroutine.

    Raises:
        Any exception raised by the coroutine.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    portal = _get_blocking_portal()
    return cast(T, portal.call(_await_coro, coro))


__all__ = ["run_sync"]
