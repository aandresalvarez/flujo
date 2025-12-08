import pytest

from flujo.state.backends.postgres import PostgresBackend


class _FakeTransaction:
    def __init__(self, conn: "_FakeConnection") -> None:
        self.conn = conn
        self.rolled_back = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if exc_type:
            self.rolled_back = True
        return False


class _FakeConnection:
    def __init__(self) -> None:
        self.calls = 0
        self.transaction_obj = _FakeTransaction(self)

    def transaction(self) -> _FakeTransaction:
        return self.transaction_obj

    async def execute(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("boom")


class _FakePool:
    def __init__(self, conn: _FakeConnection) -> None:
        self.conn = conn

    def acquire(self):
        return self

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_postgres_step_persistence_atomicity():
    """Step persistence should rollback when mid-transaction failure occurs."""

    backend = PostgresBackend("postgres://example", auto_migrate=False)
    conn = _FakeConnection()
    backend._pool = _FakePool(conn)  # type: ignore[assignment]
    backend._initialized = True  # Skip schema verify path

    step_data = {
        "run_id": "r1",
        "step_name": "s",
        "step_index": 0,
        "output": {},
        "raw_response": {},
        "cost_usd": 0.0,
        "token_counts": 0,
        "execution_time_ms": 0,
        "created_at": None,
    }

    with pytest.raises(RuntimeError):
        await backend.save_step_result(step_data)

    assert conn.transaction_obj.rolled_back is True
    assert conn.calls == 2  # second execute raised, no further statements run
