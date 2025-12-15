import asyncio

import pytest

from flujo.utils.async_bridge import run_sync


@pytest.mark.asyncio
async def test_run_coro_sync_raises_in_running_loop() -> None:
    async def _echo(val: str) -> str:
        await asyncio.sleep(0)
        return val

    with pytest.raises(TypeError, match="running event loop"):
        coro = _echo("ok")
        try:
            run_sync(coro)
        finally:
            coro.close()


def test_run_coro_sync_handles_aiosqlite_close_sync(tmp_path) -> None:
    import aiosqlite

    async def _open_and_close(db_path: str) -> str:
        async with aiosqlite.connect(db_path) as conn:
            await conn.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER)")
            await conn.commit()
            await conn.close()
        return "closed"

    db_file = tmp_path / "test.db"
    result = run_sync(_open_and_close(str(db_file)))
    assert result == "closed"
