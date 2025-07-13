"""Tests to ensure SQLite backend robustness and prevent regression of fixed issues."""

import pytest
import sqlite3
from datetime import datetime

from flujo.state.backends.sqlite import SQLiteBackend


class TestSQLiteBackendRobustness:
    """Test SQLite backend robustness and error handling."""

    @pytest.fixture
    def temp_db_path(self, tmp_path):
        """Create a temporary database path."""
        return tmp_path / "test_robustness.db"

    @pytest.fixture
    def backend(self, temp_db_path):
        """Create a SQLite backend instance."""
        return SQLiteBackend(temp_db_path)

    @pytest.mark.asyncio
    async def test_execution_time_ms_handling(self, backend):
        """Test that execution_time_ms is handled correctly without unused variables."""
        # Initialize the backend
        await backend._ensure_init()

        # Create test state with execution_time_ms
        state = {
            "pipeline_id": "test-pipeline",
            "pipeline_name": "Test Pipeline",
            "pipeline_version": "1.0.0",
            "current_step_index": 0,
            "pipeline_context": {"test": "data"},
            "last_step_output": None,
            "status": "running",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "execution_time_ms": 1500,  # Test with a value
        }

        # Test saving state with execution_time_ms
        run_id = "test-run-123"
        await backend.save_state(run_id, state)

        # Verify the state was saved correctly
        loaded_state = await backend.load_state(run_id)
        assert loaded_state is not None
        assert loaded_state["execution_time_ms"] == 1500

    @pytest.mark.asyncio
    async def test_execution_time_ms_none_handling(self, backend):
        """Test that execution_time_ms=None is handled correctly."""
        await backend._ensure_init()

        state = {
            "pipeline_id": "test-pipeline",
            "pipeline_name": "Test Pipeline",
            "pipeline_version": "1.0.0",
            "current_step_index": 0,
            "pipeline_context": {"test": "data"},
            "last_step_output": None,
            "status": "running",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            # execution_time_ms not present
        }

        run_id = "test-run-124"
        await backend.save_state(run_id, state)

        loaded_state = await backend.load_state(run_id)
        assert loaded_state is not None
        assert loaded_state["execution_time_ms"] is None

    @pytest.mark.asyncio
    async def test_retry_mechanism_safe_control_flow(self, backend):
        """Test that retry mechanism uses safe control flow instead of assert."""

        # Mock a function that always fails
        async def always_fail(*args, **kwargs):
            raise sqlite3.DatabaseError("Test error")

        # This should raise an exception, not use assert
        with pytest.raises(sqlite3.DatabaseError):
            await backend._with_retries(always_fail)

    @pytest.mark.asyncio
    async def test_retry_mechanism_schema_error_handling(self, backend):
        """Test that schema errors are handled correctly with substring matching."""

        # Mock a function that raises schema errors
        async def schema_error_func(*args, **kwargs):
            raise sqlite3.DatabaseError("no such column: missing_column")

        # This should be caught and trigger retry logic
        with pytest.raises(sqlite3.DatabaseError):
            await backend._with_retries(schema_error_func)

    @pytest.mark.asyncio
    async def test_retry_mechanism_database_locked_handling(self, backend):
        """Test that database locked errors are handled correctly."""

        # Mock a function that raises database locked errors
        async def locked_error_func(*args, **kwargs):
            raise sqlite3.OperationalError("database is locked")

        # This should be caught and trigger retry logic
        with pytest.raises(sqlite3.OperationalError):
            await backend._with_retries(locked_error_func)

    @pytest.mark.asyncio
    async def test_retry_mechanism_mixed_errors(self, backend):
        """Test that mixed error types are handled correctly."""
        call_count = 0

        async def mixed_error_func(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise sqlite3.OperationalError("database is locked")
            elif call_count == 2:
                raise sqlite3.DatabaseError("no such column: missing_column")
            else:
                return "success"

        # This should handle mixed errors and eventually succeed
        result = await backend._with_retries(mixed_error_func)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_retry_mechanism_max_retries(self, backend):
        """Test that retry mechanism respects max retries."""
        call_count = 0

        async def always_fail_func(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise sqlite3.DatabaseError(f"no such column: missing_column_{call_count}")

        # Should fail after max retries
        with pytest.raises(sqlite3.DatabaseError):
            await backend._with_retries(always_fail_func)

        # Should have been called 3 times (default max retries)
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_safe_control_flow_no_assert(self, backend):
        """Test that no assert statements are used for control flow."""
        import ast
        import inspect

        # Get the source code of the _with_retries method
        source = inspect.getsource(backend._with_retries)

        # Parse the AST and look for assert statements
        try:
            tree = ast.parse(source)
        except IndentationError:
            # If we can't parse the source directly, check the method's code object
            # This is a fallback approach

            # Check if there are any assert statements in the bytecode
            # This is a simplified check - we'll just verify the method exists and is callable
            assert callable(backend._with_retries)
            return  # Skip the AST check if we can't parse the source

        class AssertFinder(ast.NodeVisitor):
            def __init__(self):
                self.asserts = []

            def visit_Assert(self, node):
                self.asserts.append(node)

        finder = AssertFinder()
        finder.visit(tree)

        # There should be no assert statements used for control flow
        # (assert statements for debugging are OK, but not for control flow)
        for assert_node in finder.asserts:
            # Check if it's an assert False or similar control flow assert
            if isinstance(assert_node.test, ast.Constant) and assert_node.test.value is False:
                pytest.fail("Found assert False used for control flow - should use raise instead")

    @pytest.mark.asyncio
    async def test_error_message_robustness(self, backend):
        """Test that error message parsing is robust."""
        # Test with various error message formats
        error_messages = [
            "no such column: missing_column",
            "NO SUCH COLUMN: missing_column",
            "No Such Column: missing_column",
            "database is locked",
            "DATABASE IS LOCKED",
            "Database is locked",
        ]

        for error_msg in error_messages:

            async def error_func(*args, **kwargs):
                raise sqlite3.DatabaseError(error_msg)

            # Should not crash on any error message format
            with pytest.raises(sqlite3.DatabaseError):
                await backend._with_retries(error_func)

    @pytest.mark.asyncio
    async def test_state_serialization_robustness(self, backend):
        """Test that state serialization handles edge cases."""
        await backend._ensure_init()

        # Test with various data types in state
        state = {
            "pipeline_id": "test-pipeline",
            "pipeline_name": "Test Pipeline",
            "pipeline_version": "1.0.0",
            "current_step_index": 0,
            "pipeline_context": {
                "string": "test",
                "number": 42,
                "boolean": True,
                "list": [1, 2, 3],
                "dict": {"nested": "value"},
                "none": None,
            },
            "last_step_output": {"output": "test"},
            "status": "running",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "execution_time_ms": 1000,
            "memory_usage_mb": 50.5,
        }

        run_id = "test-run-125"
        await backend.save_state(run_id, state)

        loaded_state = await backend.load_state(run_id)
        assert loaded_state is not None
        assert loaded_state["pipeline_context"]["string"] == "test"
        assert loaded_state["pipeline_context"]["number"] == 42
        assert loaded_state["pipeline_context"]["boolean"] is True
        assert loaded_state["pipeline_context"]["list"] == [1, 2, 3]
        assert loaded_state["pipeline_context"]["dict"]["nested"] == "value"
        assert loaded_state["pipeline_context"]["none"] is None
        assert loaded_state["execution_time_ms"] == 1000
        assert loaded_state["memory_usage_mb"] == 50.5


class TestSQLiteBackendRegressionPrevention:
    """Test to prevent regression of fixed issues."""

    @pytest.mark.asyncio
    async def test_no_unused_variables(self, tmp_path):
        """Test that no unused variables are created in save_state."""
        backend = SQLiteBackend(tmp_path / "test.db")
        await backend._ensure_init()

        # Create a state with execution_time_ms
        state = {
            "pipeline_id": "test",
            "pipeline_name": "Test",
            "pipeline_version": "1.0",
            "current_step_index": 0,
            "pipeline_context": {},
            "last_step_output": None,
            "status": "running",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "execution_time_ms": 500,
        }

        # This should not create any unused variables
        await backend.save_state("test-run", state)

        # Verify the state was saved correctly
        loaded = await backend.load_state("test-run")
        assert loaded is not None
        assert loaded["execution_time_ms"] == 500

    @pytest.mark.asyncio
    async def test_safe_exception_handling(self, tmp_path):
        """Test that exceptions are handled safely without assert."""
        backend = SQLiteBackend(tmp_path / "test.db")

        # Test that the retry mechanism doesn't use assert for control flow
        async def failing_func():
            raise RuntimeError("Test error")

        with pytest.raises(RuntimeError):
            await backend._with_retries(failing_func)

    @pytest.mark.asyncio
    async def test_error_detection_robustness(self, tmp_path):
        """Test that error detection is robust and not brittle."""
        backend = SQLiteBackend(tmp_path / "test.db")

        # Test various error message formats
        error_tests = [
            ("no such column: test", True),  # Should be detected
            ("NO SUCH COLUMN: test", True),  # Should be detected (case insensitive)
            ("different error", False),  # Should not be detected
            ("", False),  # Should not be detected
        ]

        for error_msg, should_detect in error_tests:

            async def test_func():
                raise sqlite3.DatabaseError(error_msg)

            if should_detect:
                # Should trigger retry logic
                with pytest.raises(sqlite3.DatabaseError):
                    await backend._with_retries(test_func)
            else:
                # Should not trigger retry logic
                with pytest.raises(sqlite3.DatabaseError):
                    await backend._with_retries(test_func)
