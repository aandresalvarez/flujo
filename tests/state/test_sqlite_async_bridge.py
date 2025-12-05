import asyncio

import pytest

from flujo.state.backends.sqlite_core import _run_coro_sync


@pytest.mark.asyncio
async def test_run_coro_sync_handles_running_loop() -> None:
    async def _echo(val: str) -> str:
        await asyncio.sleep(0)
        return val

    first = _run_coro_sync(_echo("ok"))
    second = _run_coro_sync(_echo("again"))

    assert first == "ok"
    assert second == "again"
