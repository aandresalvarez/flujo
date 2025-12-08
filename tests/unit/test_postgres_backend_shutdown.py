from unittest.mock import AsyncMock

import pytest

from flujo.state.backends.postgres import PostgresBackend


@pytest.mark.asyncio
async def test_postgres_backend_pool_closure() -> None:
    """Shutdown must close the asyncpg pool when present."""

    backend = PostgresBackend("postgres://example", auto_migrate=False)
    fake_pool = AsyncMock()
    backend._pool = fake_pool

    await backend.shutdown()

    fake_pool.close.assert_awaited_once()
    assert backend._pool is None
