import json

import pytest

from flujo.exceptions import PipelineAbortSignal
from flujo.state.backends.postgres import PostgresBackend, _jsonb, _validate_run_id_param


class _FakeTransaction:
    def __init__(self, conn: "_FakeConnection") -> None:
        self.conn = conn

    async def __aenter__(self) -> "_FakeTransaction":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self.conn.rollback_count += 1
        return False


class _FakeConnection:
    def __init__(
        self,
        *,
        fail_on_update: bool = False,
        fail_first_insert: bool = False,
        first_insert_error: Exception | None = None,
    ) -> None:
        self.calls = 0
        self.rollback_count = 0
        self.insert_attempts = 0
        self.updated_run_id: str | None = None
        self.fail_on_update = fail_on_update
        self.fail_first_insert = fail_first_insert
        self.first_insert_error = first_insert_error
        self.fallback_output_json: str | None = None
        self.fallback_raw_response_json: str | None = None

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction(self)

    async def execute(self, query: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        if "INSERT INTO steps" in query:
            self.insert_attempts += 1
            if self.insert_attempts == 1 and self.first_insert_error is not None:
                raise self.first_insert_error
            if self.fail_first_insert and self.insert_attempts == 1:
                raise RuntimeError("unsupported \x00 payload")
            if self.insert_attempts == 2:
                self.fallback_output_json = args[4]
                self.fallback_raw_response_json = args[5]
            return "INSERT 0 1"
        if "UPDATE runs SET updated_at = NOW()" in query:
            self.updated_run_id = args[0]
            if self.fail_on_update:
                raise RuntimeError("boom")
            return "UPDATE 1"
        return "OK"


class _FakePool:
    def __init__(self, conn: _FakeConnection) -> None:
        self.conn = conn

    def acquire(self) -> "_FakePool":
        return self

    async def __aenter__(self) -> _FakeConnection:
        return self.conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


@pytest.mark.asyncio
async def test_postgres_step_persistence_atomicity() -> None:
    """Step persistence should rollback when post-insert failure occurs."""

    backend = PostgresBackend("postgres://example", auto_migrate=False)
    conn = _FakeConnection(fail_on_update=True)
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

    with pytest.raises(RuntimeError, match="boom"):
        await backend.save_step_result(step_data)

    assert conn.insert_attempts == 1
    assert conn.calls == 2
    assert conn.rollback_count >= 1


def test_jsonb_strips_raw_nul_recursively() -> None:
    payload = {
        "k\x00ey": "va\x00lue",
        "nested": ["a\x00", {"inner\x00": "b\x00"}],
        "tuple_value": ("x\x00",),
        "literal": "a\\u0000b",
    }
    encoded = _jsonb(payload)
    assert encoded is not None

    decoded = json.loads(encoded)
    assert "key" in decoded
    assert decoded["key"] == "value"
    assert decoded["nested"][0] == "a"
    assert decoded["nested"][1]["inner"] == "b"
    assert decoded["tuple_value"][0] == "x"
    # Literal escaped sequence should remain untouched (it's not a raw NUL char).
    assert decoded["literal"] == "a\\u0000b"


@pytest.mark.asyncio
async def test_postgres_step_persistence_fallback_placeholder() -> None:
    backend = PostgresBackend("postgres://example", auto_migrate=False)
    conn = _FakeConnection(fail_first_insert=True)
    backend._pool = _FakePool(conn)  # type: ignore[assignment]
    backend._initialized = True  # Skip schema verify path

    step_data = {
        "run_id": "r1",
        "step_name": "s",
        "step_index": 0,
        "output": {"bad": "a\x00b"},
        "raw_response": {"bad": "a\x00b"},
        "cost_usd": 0.0,
        "token_counts": 0,
        "execution_time_ms": 0,
        "created_at": None,
    }

    await backend.save_step_result(step_data)

    assert conn.insert_attempts == 2
    assert conn.updated_run_id == "r1"
    assert conn.fallback_output_json is not None
    assert conn.fallback_raw_response_json is not None

    output_payload = json.loads(conn.fallback_output_json)
    raw_response_payload = json.loads(conn.fallback_raw_response_json)
    assert output_payload["output_redacted"] is True
    assert raw_response_payload["raw_response_redacted"] is True
    assert output_payload["_flujo_persistence"]["degraded"] is True
    assert output_payload["_flujo_persistence"]["reason"] == "postgres_step_persistence_fallback"


@pytest.mark.asyncio
async def test_postgres_step_persistence_reraises_control_flow_error() -> None:
    backend = PostgresBackend("postgres://example", auto_migrate=False)
    conn = _FakeConnection(first_insert_error=PipelineAbortSignal("abort"))
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

    with pytest.raises(PipelineAbortSignal):
        await backend.save_step_result(step_data)

    assert conn.insert_attempts == 1
    assert conn.updated_run_id is None


def test_validate_run_id_param_rejects_raw_nul() -> None:
    with pytest.raises(ValueError, match="run_id contains raw NUL"):
        _validate_run_id_param("a\x00b")


@pytest.mark.asyncio
async def test_load_state_rejects_raw_nul_run_id() -> None:
    backend = PostgresBackend("postgres://example", auto_migrate=False)
    with pytest.raises(ValueError, match="run_id contains raw NUL"):
        await backend.load_state("a\x00b")
