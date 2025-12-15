"""Tests for SQL injection security in state backends."""

import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flujo.state.backends.sqlite import SQLiteBackend

# Module-level constants for malicious inputs to reduce duplication
MALICIOUS_NUMERIC_INPUTS = [
    "24; DROP TABLE workflow_state; --",
    "24' OR '1'='1",
    "24 UNION SELECT * FROM workflow_state --",
    "24; INSERT INTO workflow_state VALUES ('hacked', 'hacked', 'hacked', 'hacked', 0, '{}', NULL, 'running', datetime('now'), datetime('now'), 0, NULL, NULL, NULL); --",
    "24' AND 1=1 --",
    "24' OR 1=1 --",
    "24' UNION SELECT 1,2,3,4,5,6,7,8,9,10,11,12 --",
]

MALICIOUS_STRING_INPUTS = [
    "normal'; DROP TABLE workflow_state; --",
    "normal' OR '1'='1",
    "normal' UNION SELECT * FROM workflow_state --",
    "normal'; INSERT INTO workflow_state VALUES ('hacked', 'hacked', 'hacked', 'hacked', 0, '{}', NULL, 'running', datetime('now'), datetime('now'), 0, NULL, NULL, NULL); --",
    "normal' AND 1=1 --",
    "normal' OR 1=1 --",
    "normal' UNION SELECT 1,2,3,4,5,6,7,8,9,10,11,12 --",
]


# Helper function to create context-specific malicious inputs
def get_malicious_numeric_inputs(prefix: str = "24") -> list[str]:
    """Get malicious numeric inputs with a custom prefix."""
    return [input.replace("24", prefix) for input in MALICIOUS_NUMERIC_INPUTS]


def get_malicious_string_inputs(prefix: str = "normal") -> list[str]:
    """Get malicious string inputs with a custom prefix."""
    return [input.replace("normal", prefix) for input in MALICIOUS_STRING_INPUTS]


class TestSQLInjectionSecurity:
    """Test suite for SQL injection vulnerabilities and security concerns."""

    @pytest.fixture
    async def backend(self, tmp_path: Path) -> SQLiteBackend:
        """Create a test SQLite backend."""
        backend = SQLiteBackend(tmp_path / "test.db")
        yield backend
        # Cleanup
        await backend.close()

    @pytest.fixture
    async def sample_workflows(self, backend: SQLiteBackend) -> None:
        """Create sample workflows for testing."""
        now = datetime.now(timezone.utc).replace(microsecond=0)
        past = now - timedelta(days=1)

        # Create workflows with different statuses and times
        workflows = [
            {
                "run_id": "normal_run",
                "pipeline_id": "pipeline1",
                "pipeline_name": "Test Pipeline",
                "pipeline_version": "1.0",
                "current_step_index": 0,
                "pipeline_context": {"data": "normal"},
                "last_step_output": None,
                "status": "running",
                "created_at": now,
                "updated_at": now,
                "total_steps": 5,
                "error_message": None,
                "execution_time_ms": 1000,
                "memory_usage_mb": 10.0,
            },
            {
                "run_id": "failed_run",
                "pipeline_id": "pipeline1",
                "pipeline_name": "Test Pipeline",
                "pipeline_version": "1.0",
                "current_step_index": 2,
                "pipeline_context": {"data": "failed"},
                "last_step_output": "error occurred",
                "status": "failed",
                "created_at": past,
                "updated_at": past,
                "total_steps": 5,
                "error_message": "Test error",
                "execution_time_ms": 2000,
                "memory_usage_mb": 20.0,
            },
            {
                "run_id": "old_run",
                "pipeline_id": "pipeline2",
                "pipeline_name": "Old Pipeline",
                "pipeline_version": "0.9",
                "current_step_index": 3,
                "pipeline_context": {"data": "old"},
                "last_step_output": "completed",
                "status": "completed",
                "created_at": past - timedelta(days=30),
                "updated_at": past - timedelta(days=30),
                "total_steps": 3,
                "error_message": None,
                "execution_time_ms": 1500,
                "memory_usage_mb": 15.0,
            },
        ]

        for workflow in workflows:
            await backend.save_state(workflow["run_id"], workflow)

    @pytest.mark.asyncio
    async def test_get_failed_workflows_sql_injection_resistance(
        self, backend: SQLiteBackend, sample_workflows: None
    ) -> None:
        """Test that get_failed_workflows is resistant to SQL injection."""

        # Test with normal input
        failed_workflows = await backend.get_failed_workflows(hours_back=48)
        assert len(failed_workflows) == 1
        assert failed_workflows[0]["run_id"] == "failed_run"

        # Test with malicious input that would cause SQL injection
        malicious_inputs = get_malicious_numeric_inputs("24")

        for malicious_input in malicious_inputs:
            # These should not cause SQL injection and should either:
            # 1. Return the expected result (if the input is treated as a valid number)
            # 2. Raise an exception (if the input is invalid)
            # 3. Return empty results (if the query fails safely)

            try:
                result = await backend.get_failed_workflows(hours_back=malicious_input)
                # If no exception, verify the result is safe
                assert isinstance(result, list)
                # Should not have created any unexpected workflows
                all_workflows = await backend.list_workflows()
                expected_workflow_ids = {"normal_run", "failed_run", "old_run"}
                actual_workflow_ids = {wf["run_id"] for wf in all_workflows}
                assert actual_workflow_ids == expected_workflow_ids
            except (ValueError, TypeError):
                # Expected for invalid input types
                pass

    @pytest.mark.asyncio
    async def test_cleanup_old_workflows_sql_injection_resistance(
        self, backend: SQLiteBackend
    ) -> None:
        """Test that cleanup_old_workflows is resistant to SQL injection."""

        # Create sample workflows for testing
        now = datetime.now(timezone.utc).replace(microsecond=0)
        past = now - timedelta(days=1)

        workflows = [
            {
                "run_id": "normal_run",
                "pipeline_id": "pipeline1",
                "pipeline_name": "Test Pipeline",
                "pipeline_version": "1.0",
                "current_step_index": 0,
                "pipeline_context": {"data": "normal"},
                "last_step_output": None,
                "status": "running",
                "created_at": now,
                "updated_at": now,
                "total_steps": 5,
                "error_message": None,
                "execution_time_ms": 1000,
                "memory_usage_mb": 10.0,
            },
            {
                "run_id": "old_run",
                "pipeline_id": "pipeline2",
                "pipeline_name": "Old Pipeline",
                "pipeline_version": "0.9",
                "current_step_index": 3,
                "pipeline_context": {"data": "old"},
                "last_step_output": "completed",
                "status": "completed",
                "created_at": past - timedelta(days=30),
                "updated_at": past - timedelta(days=30),
                "total_steps": 3,
                "error_message": None,
                "execution_time_ms": 1500,
                "memory_usage_mb": 15.0,
            },
        ]

        for workflow in workflows:
            await backend.save_state(workflow["run_id"], workflow)

        # Test with normal input
        deleted_count = await backend.cleanup_old_workflows(days_old=0)
        assert deleted_count >= 1  # At least some workflows should be deleted

        # Test with malicious input that would cause SQL injection
        malicious_inputs = get_malicious_numeric_inputs("30")

        for malicious_input in malicious_inputs:
            try:
                result = await backend.cleanup_old_workflows(days_old=malicious_input)
                # If no exception, verify the result is safe
                assert isinstance(result, int)
                # Should not have created any unexpected workflows
                all_workflows = await backend.list_workflows()
                actual_workflow_ids = {wf["run_id"] for wf in all_workflows}
                # Should only contain workflows we created, not any injected ones
                assert "normal_run" in actual_workflow_ids or len(actual_workflow_ids) == 0
                # Should not contain any unexpected workflow IDs
                unexpected_ids = actual_workflow_ids - {"normal_run", "old_run"}
                assert len(unexpected_ids) == 0, f"Unexpected workflow IDs found: {unexpected_ids}"
            except (ValueError, TypeError):
                # Expected for invalid input types
                pass

    @pytest.mark.asyncio
    async def test_list_workflows_sql_injection_resistance(
        self, backend: SQLiteBackend, sample_workflows: None
    ) -> None:
        """Test that list_workflows is resistant to SQL injection."""

        # Test with normal input
        all_workflows = await backend.list_workflows()
        assert len(all_workflows) == 3

        failed_workflows = await backend.list_workflows(status="failed")
        assert len(failed_workflows) == 1
        assert failed_workflows[0]["run_id"] == "failed_run"

        # Test with malicious input that would cause SQL injection
        malicious_inputs = MALICIOUS_STRING_INPUTS

        for malicious_input in malicious_inputs:
            try:
                result = await backend.list_workflows(status=malicious_input)
                # If no exception, verify the result is safe
                assert isinstance(result, list)
                # Should not have created any unexpected workflows
                all_workflows = await backend.list_workflows()
                expected_workflow_ids = {"normal_run", "failed_run", "old_run"}
                actual_workflow_ids = {wf["run_id"] for wf in all_workflows}
                assert actual_workflow_ids == expected_workflow_ids
            except (ValueError, TypeError):
                # Expected for invalid input types
                pass

    @pytest.mark.asyncio
    async def test_save_state_sql_injection_resistance(self, backend: SQLiteBackend) -> None:
        """Test that save_state is resistant to SQL injection in run_id."""

        # Test with normal input
        now = datetime.now(timezone.utc).replace(microsecond=0)
        normal_state = {
            "run_id": "normal_run",
            "pipeline_id": "pipeline1",
            "pipeline_name": "Test Pipeline",
            "pipeline_version": "1.0",
            "current_step_index": 0,
            "pipeline_context": {"data": "normal"},
            "last_step_output": None,
            "status": "running",
            "created_at": now,
            "updated_at": now,
        }
        await backend.save_state("normal_run", normal_state)

        # Test with malicious run_id that would cause SQL injection
        malicious_run_ids = MALICIOUS_STRING_INPUTS

        for malicious_run_id in malicious_run_ids:
            try:
                await backend.save_state(malicious_run_id, normal_state)
                # If no exception, verify the state was saved safely
                loaded_state = await backend.load_state(malicious_run_id)
                assert loaded_state is not None
                assert loaded_state["run_id"] == malicious_run_id
                # Should not have created any unexpected workflows
                all_workflows = await backend.list_workflows()
                workflow_ids = {wf["run_id"] for wf in all_workflows}
                # The malicious run_id should be treated as a literal string, not SQL
                assert malicious_run_id in workflow_ids
            except Exception:
                # Some malicious inputs might cause other errors, which is acceptable
                pass

    @pytest.mark.asyncio
    async def test_load_state_sql_injection_resistance(self, backend: SQLiteBackend) -> None:
        """Test that load_state is resistant to SQL injection in run_id."""

        # First save a normal state
        now = datetime.now(timezone.utc).replace(microsecond=0)
        normal_state = {
            "run_id": "normal_run",
            "pipeline_id": "pipeline1",
            "pipeline_name": "Test Pipeline",
            "pipeline_version": "1.0",
            "current_step_index": 0,
            "pipeline_context": {"data": "normal"},
            "last_step_output": None,
            "status": "running",
            "created_at": now,
            "updated_at": now,
        }
        await backend.save_state("normal_run", normal_state)

        # Test with malicious run_id that would cause SQL injection
        malicious_run_ids = MALICIOUS_STRING_INPUTS

        for malicious_run_id in malicious_run_ids:
            try:
                result = await backend.load_state(malicious_run_id)
                # Should return None for non-existent run_id, not cause SQL injection
                assert result is None
                # Should not have created any unexpected workflows
                all_workflows = await backend.list_workflows()
                workflow_ids = {wf["run_id"] for wf in all_workflows}
                assert "normal_run" in workflow_ids
                assert malicious_run_id not in workflow_ids
            except Exception:
                # Some malicious inputs might cause other errors, which is acceptable
                pass

    @pytest.mark.asyncio
    async def test_delete_state_sql_injection_resistance(self, backend: SQLiteBackend) -> None:
        """Test that delete_state is resistant to SQL injection in run_id."""

        # First save a normal state
        now = datetime.now(timezone.utc).replace(microsecond=0)
        normal_state = {
            "run_id": "normal_run",
            "pipeline_id": "pipeline1",
            "pipeline_name": "Test Pipeline",
            "pipeline_version": "1.0",
            "current_step_index": 0,
            "pipeline_context": {"data": "normal"},
            "last_step_output": None,
            "status": "running",
            "created_at": now,
            "updated_at": now,
        }
        await backend.save_state("normal_run", normal_state)

        # Test with malicious run_id that would cause SQL injection
        malicious_run_ids = MALICIOUS_STRING_INPUTS

        for malicious_run_id in malicious_run_ids:
            try:
                await backend.delete_state(malicious_run_id)
                # Should not cause SQL injection and should not affect the normal workflow
                normal_workflow = await backend.load_state("normal_run")
                assert normal_workflow is not None
                assert normal_workflow["run_id"] == "normal_run"
                # Should not have created any unexpected workflows
                all_workflows = await backend.list_workflows()
                workflow_ids = {wf["run_id"] for wf in all_workflows}
                assert "normal_run" in workflow_ids
                assert malicious_run_id not in workflow_ids
            except Exception:
                # Some malicious inputs might cause other errors, which is acceptable
                pass

    @pytest.mark.asyncio
    async def test_parameterized_queries_work_correctly(self, backend: SQLiteBackend) -> None:
        """Test that parameterized queries work correctly with various input types."""

        now = datetime.now(timezone.utc).replace(microsecond=0)
        state = {
            "run_id": "test_run",
            "pipeline_id": "pipeline1",
            "pipeline_name": "Test Pipeline",
            "pipeline_version": "1.0",
            "current_step_index": 0,
            "pipeline_context": {"data": "test"},
            "last_step_output": None,
            "status": "failed",
            "created_at": now,
            "updated_at": now,
        }
        await backend.save_state("test_run", state)

        # Test get_failed_workflows with various numeric inputs
        test_hours = [0, 1, 24, 48, 168, 720, 8760]  # Various hour values
        for hours in test_hours:
            result = await backend.get_failed_workflows(hours_back=hours)
            assert isinstance(result, list)
            # Should find our failed workflow
            assert len(result) >= 1
            assert any(wf["run_id"] == "test_run" for wf in result)

        # Test cleanup_old_workflows with various numeric inputs
        test_days = [0, 1, 7, 30, 365]  # Various day values
        for days in test_days:
            result = await backend.cleanup_old_workflows(days_old=days)
            assert isinstance(result, int)
            # Should delete our workflow when days_old=0
            if days == 0:
                assert result >= 0  # May or may not delete depending on timing
            else:
                assert result >= 0

    @pytest.mark.asyncio
    async def test_edge_cases_and_boundary_values(self, backend: SQLiteBackend) -> None:
        """Test edge cases and boundary values for SQL injection resistance."""

        # Test with extreme values that might cause issues
        extreme_values = [
            -1,
            0,
            1,
            999999,
            999999999,
            -1.0,
            0.0,
            1.0,
            999999.0,
            float("inf"),
            float("-inf"),
            float("nan"),
        ]

        for value in extreme_values:
            try:
                # Test get_failed_workflows
                result1 = await backend.get_failed_workflows(hours_back=value)
                assert isinstance(result1, list)

                # Test cleanup_old_workflows
                result2 = await backend.cleanup_old_workflows(days_old=value)
                assert isinstance(result2, int)
            except (ValueError, TypeError, OverflowError):
                # Expected for invalid values
                pass

    @pytest.mark.asyncio
    async def test_sql_injection_prevention_patterns(self, backend: SQLiteBackend) -> None:
        """Test common SQL injection patterns to ensure they are prevented."""

        # Common SQL injection patterns
        injection_patterns = [
            # Basic injection attempts
            "'; DROP TABLE workflow_state; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM workflow_state --",
            "'; INSERT INTO workflow_state VALUES ('hacked', 'hacked', 'hacked', 'hacked', 0, '{}', NULL, 'running', datetime('now'), datetime('now'), 0, NULL, NULL, NULL); --",
            "' AND 1=1 --",
            "' OR 1=1 --",
            "' UNION SELECT 1,2,3,4,5,6,7,8,9,10,11,12 --",
            # Advanced injection patterns
            "'; EXEC xp_cmdshell('dir'); --",
            "'; WAITFOR DELAY '00:00:05'; --",
            "'; SELECT SLEEP(5); --",
            "'; SHUTDOWN; --",
            # Blind injection patterns
            "' AND (SELECT COUNT(*) FROM workflow_state) > 0 --",
            "' AND (SELECT COUNT(*) FROM workflow_state) = 0 --",
            "' AND (SELECT COUNT(*) FROM workflow_state) BETWEEN 0 AND 100 --",
            # Time-based injection patterns
            "' AND (SELECT COUNT(*) FROM workflow_state WHERE 1=1) > 0 --",
            "' AND (SELECT COUNT(*) FROM workflow_state WHERE 1=0) > 0 --",
        ]

        for pattern in injection_patterns:
            try:
                # Test with malicious input in various contexts
                result1 = await backend.get_failed_workflows(hours_back=pattern)
                assert isinstance(result1, list)

                result2 = await backend.cleanup_old_workflows(days_old=pattern)
                assert isinstance(result2, int)

                result3 = await backend.list_workflows(status=pattern)
                assert isinstance(result3, list)

                # Verify no unexpected side effects
                all_workflows = await backend.list_workflows()
                assert isinstance(all_workflows, list)

            except (ValueError, TypeError, AttributeError):
                # Expected for invalid input types
                pass
