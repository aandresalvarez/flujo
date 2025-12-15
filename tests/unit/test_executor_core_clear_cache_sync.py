from __future__ import annotations

import pytest

from flujo.application.core.executor_core import ExecutorCore

pytestmark = pytest.mark.fast


def test_clear_cache_sync_runs_without_loop() -> None:
    core = ExecutorCore()
    core.clear_cache_sync()


@pytest.mark.asyncio
async def test_clear_cache_sync_raises_in_running_loop() -> None:
    core = ExecutorCore()
    with pytest.raises(RuntimeError, match=r"clear_cache_sync\(\) cannot be called"):
        core.clear_cache_sync()
