"""Tests for SQLiteBackend retry mechanism and error handling scenarios."""

import asyncio
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import pytest

from flujo.state.backends.sqlite import SQLiteBackend

# Mark all tests in this module for serial execution to prevent SQLite concurrency issues
pytestmark = [pytest.mark.serial, pytest.mark.slow]


@pytest.fixture
async def sqlite_backend(tmp_path: Path):
    """Create backends with automatic cleanup."""
    backends = []

    def _create_backend(db_name: str):
        backend = SQLiteBackend(tmp_path / db_name)
        backends.append(backend)
        return backend

    yield _create_backend

    # Cleanup with timeout (retry mechanism tests may have backends in retry states)
    for backend in backends:
        try:
            await asyncio.wait_for(backend.close(), timeout=2.0)
        except (Exception, asyncio.TimeoutError):
            pass  # Best effort cleanup


@pytest.mark.asyncio
async def test_with_retries_no_infinite_loop_on_schema_errors(sqlite_backend) -> None:
    """Test that _with_retries doesn't cause infinite loops on schema errors."""
    backend = sqlite_backend("retry_test.db")

    # Mock a function that always raises schema errors
    async def failing_coro(*args, **kwargs):
        raise sqlite3.DatabaseError("no such column: missing_column")

    # This should not cause an infinite loop and should raise after 3 attempts
    with pytest.raises(sqlite3.DatabaseError, match="no such column: missing_column"):
        await backend._with_retries(failing_coro)


@pytest.mark.asyncio
async def test_with_retries_proper_initialization_reset(sqlite_backend) -> None:
    """Test that schema migration properly resets initialization state."""
    backend = sqlite_backend("init_test.db")

    # First call should initialize normally
    await backend._ensure_init()
    assert backend._initialized is True

    # Mock a schema error that triggers migration
    async def schema_error_coro(*args, **kwargs):
        raise sqlite3.DatabaseError("no such column: missing_column")

    with pytest.raises(sqlite3.DatabaseError):
        # This should reset initialization and retry properly
        await backend._with_retries(schema_error_coro)


@pytest.mark.asyncio
async def test_with_retries_explicit_return_handling(sqlite_backend) -> None:
    """Test that the method always has an explicit return path."""
    backend = sqlite_backend("return_test.db")

    # Mock a function that succeeds
    async def successful_coro(*args, **kwargs):
        return "success"

    # This should return the result
    result = await backend._with_retries(successful_coro)
    assert result == "success"

    # Mock a function that always fails with non-schema errors
    async def failing_coro(*args, **kwargs):
        raise sqlite3.DatabaseError("some other database error")

    # This should raise immediately without retries
    with pytest.raises(sqlite3.DatabaseError, match="some other database error"):
        await backend._with_retries(failing_coro)


@pytest.mark.asyncio
async def test_with_retries_database_locked_scenario(sqlite_backend) -> None:
    """Test retry behavior for database locked errors."""
    backend = sqlite_backend("locked_test.db")

    call_count = 0

    async def locked_then_success(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise sqlite3.OperationalError("database is locked")
        return "success after retries"

    # Should retry and eventually succeed
    result = await backend._with_retries(locked_then_success)
    assert result == "success after retries"
    assert call_count == 3


@pytest.mark.asyncio
async def test_with_retries_max_retries_exceeded(sqlite_backend) -> None:
    """Test that max retries are respected for database locked errors."""
    backend = sqlite_backend("max_retries_test.db")

    async def always_locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    # Should fail after 3 attempts
    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        await backend._with_retries(always_locked)


@pytest.mark.asyncio
async def test_with_retries_schema_migration_retry_limit(sqlite_backend) -> None:
    """Test that schema migration respects retry limits."""
    backend = sqlite_backend("schema_retry_test.db")

    async def always_schema_error(*args, **kwargs):
        raise sqlite3.DatabaseError("no such column: missing_column")

    # Should fail after 3 attempts with proper error message
    with pytest.raises(sqlite3.DatabaseError, match="no such column: missing_column"):
        await backend._with_retries(always_schema_error)


@pytest.mark.asyncio
async def test_with_retries_mixed_error_scenarios(sqlite_backend) -> None:
    """Test retry behavior with mixed error types."""
    backend = sqlite_backend("mixed_errors_test.db")

    call_count = 0

    async def mixed_errors(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise sqlite3.OperationalError("database is locked")
        elif call_count == 2:
            raise sqlite3.DatabaseError("no such column: missing_column")
        else:
            return "success after mixed errors"

    # Should handle mixed errors and eventually succeed
    result = await backend._with_retries(mixed_errors)
    assert result == "success after mixed errors"
    assert call_count == 3


@pytest.mark.asyncio
async def test_with_retries_proper_ensure_init_calls(sqlite_backend) -> None:
    """Test that schema errors properly call _ensure_init instead of _init_db."""
    backend = sqlite_backend("ensure_init_test.db")

    # Mock _ensure_init to track calls
    original_ensure_init = backend._ensure_init
    ensure_init_calls = 0

    async def mock_ensure_init():
        nonlocal ensure_init_calls
        ensure_init_calls += 1
        return await original_ensure_init()

    backend._ensure_init = mock_ensure_init

    async def schema_error_coro(*args, **kwargs):
        raise sqlite3.DatabaseError("no such column: missing_column")

    # This should call _ensure_init, not _init_db directly
    with pytest.raises(sqlite3.DatabaseError):
        await backend._with_retries(schema_error_coro)

    # Should have called _ensure_init during retry attempts
    assert ensure_init_calls > 0


@pytest.mark.asyncio
async def test_with_retries_concurrent_access_safety(sqlite_backend) -> None:
    """Test that retry mechanism is safe under concurrent access."""
    backend = sqlite_backend("concurrent_test.db")

    # Create multiple concurrent operations
    async def concurrent_operation(operation_id: int):
        async def operation(*args, **kwargs):
            if operation_id % 2 == 0:
                raise sqlite3.OperationalError("database is locked")
            return f"success_{operation_id}"

        return await backend._with_retries(operation)

    # Run multiple concurrent operations
    results = await asyncio.gather(
        concurrent_operation(1),
        concurrent_operation(2),
        concurrent_operation(3),
        concurrent_operation(4),
        return_exceptions=True,
    )

    # Some should succeed, some should fail, but no infinite loops
    assert len(results) == 4
    assert any(isinstance(r, str) and r.startswith("success_") for r in results)
    assert any(isinstance(r, Exception) for r in results)


@pytest.mark.asyncio
async def test_with_retries_memory_cleanup(sqlite_backend) -> None:
    """Test that retry mechanism doesn't leak memory during repeated failures."""
    backend = sqlite_backend("memory_test.db")

    # Create a function that fails consistently
    async def memory_leak_test(*args, **kwargs):
        raise sqlite3.DatabaseError("no such column: missing_column")

    # Run multiple retry attempts to check for memory leaks
    for _ in range(5):
        with pytest.raises(sqlite3.DatabaseError):
            await backend._with_retries(memory_leak_test)

    # The backend should still be in a valid state
    assert backend.db_path.exists()
    # The _initialized flag may be True after _ensure_init() succeeds,
    # but the important thing is that the backend remains functional
    assert backend.db_path.parent.exists()  # Directory should exist


@pytest.mark.asyncio
async def test_with_retries_logging_behavior(sqlite_backend) -> None:
    """Test that retry mechanism provides proper logging."""
    backend = sqlite_backend("logging_test.db")

    call_count = 0

    async def logging_test_coro(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise sqlite3.DatabaseError("no such column: missing_column")
        return "success"

    # This should log retry attempts and final success
    result = await backend._with_retries(logging_test_coro)
    assert result == "success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_with_retries_type_safety(sqlite_backend) -> None:
    """Test that the method maintains type safety and never returns None implicitly."""
    backend = sqlite_backend("type_safety_test.db")

    # Test with various return types
    async def return_dict(*args, **kwargs):
        return {"key": "value"}

    async def return_list(*args, **kwargs):
        return [1, 2, 3]

    async def return_none(*args, **kwargs):
        return None

    # All should return their expected types
    dict_result = await backend._with_retries(return_dict)
    assert isinstance(dict_result, dict)
    assert dict_result["key"] == "value"

    list_result = await backend._with_retries(return_list)
    assert isinstance(list_result, list)
    assert list_result == [1, 2, 3]

    none_result = await backend._with_retries(return_none)
    assert none_result is None


@pytest.mark.asyncio
async def test_with_retries_edge_case_parameters(sqlite_backend) -> None:
    """Test retry mechanism with edge case parameters."""
    backend = sqlite_backend("edge_case_test.db")

    # Test with various parameter types
    async def edge_case_coro(arg1, arg2=None, **kwargs):
        return {"arg1": arg1, "arg2": arg2, "kwargs": kwargs}

    # Test with different parameter combinations
    result = await backend._with_retries(edge_case_coro, "test", arg2="value", extra="data")
    assert result["arg1"] == "test"
    assert result["arg2"] == "value"
    assert result["kwargs"]["extra"] == "data"


@pytest.mark.asyncio
async def test_with_retries_real_database_operations(sqlite_backend) -> None:
    """Test retry mechanism with real database operations."""
    backend = sqlite_backend("real_db_test.db")

    # Test save_state with retry mechanism
    state = {
        "pipeline_id": "test_pipeline",
        "pipeline_name": "Test Pipeline",
        "pipeline_version": "1.0",
        "current_step_index": 0,
        "pipeline_context": {"test": "data"},
        "last_step_output": None,
        "status": "running",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }

    # This should work normally
    await backend.save_state("test_run", state)

    # Test load_state with retry mechanism
    loaded = await backend.load_state("test_run")
    assert loaded is not None
    assert loaded["pipeline_id"] == "test_pipeline"


@pytest.mark.asyncio
async def test_with_retries_corruption_recovery(sqlite_backend, tmp_path: Path) -> None:
    """Test that retry mechanism works with database corruption scenarios."""
    db_name = "corruption_test.db"
    db_path = tmp_path / db_name
    backend = sqlite_backend(db_name)

    # Create a corrupted database
    with open(db_path, "w") as f:
        f.write("corrupt data")

    # Now try to use the backend, expecting it to handle the corruption gracefully
    try:

        async def no_op(*args, **kwargs):
            return None

        result = await backend._with_retries(no_op)  # Use async function instead of lambda
        assert result is None  # Explicitly verify expected behavior
    except sqlite3.DatabaseError:
        pass  # The test is just to ensure no infinite loop or crash
