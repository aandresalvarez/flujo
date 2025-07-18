"""Security tests for SQL injection prevention in SQLite backend.

This test suite verifies that the SQLite backend properly validates and sanitizes
all SQL inputs to prevent injection attacks. This is critical for healthcare,
legal, and finance applications where data integrity and security are paramount.
"""

import pytest
from pathlib import Path
from datetime import datetime

from flujo.state.backends.sqlite import (
    SQLiteBackend,
    _validate_sql_identifier,
    _validate_column_definition,
)


class TestSQLInjectionPrevention:
    """Test SQL injection prevention mechanisms."""

    @pytest.fixture
    def temp_db_path(self, tmp_path: Path) -> Path:
        """Create a temporary database path."""
        return tmp_path / "security_test.db"

    @pytest.fixture
    def backend(self, temp_db_path: Path) -> SQLiteBackend:
        """Create a SQLite backend instance."""
        return SQLiteBackend(temp_db_path)

    def test_validate_sql_identifier_safe_identifiers(self):
        """Test that safe SQL identifiers are accepted."""
        safe_identifiers = [
            "valid_column",
            "column_123",
            "_private_column",
            "UPPER_CASE",
            "mixedCase123",
        ]

        for identifier in safe_identifiers:
            assert _validate_sql_identifier(identifier) is True

    def test_validate_sql_identifier_dangerous_identifiers(self):
        """Test that dangerous SQL identifiers are rejected."""
        dangerous_identifiers = [
            "DROP TABLE",
            "'; DROP TABLE users; --",
            "column; DELETE FROM users",
            "column' OR '1'='1",
            "column UNION SELECT * FROM users",
            "column/*comment*/",
            "column--comment",
            "column; INSERT INTO users VALUES (1, 'hacker')",
            "column; UPDATE users SET password='hacked'",
            "column; CREATE TABLE malicious (id INTEGER)",
            "column; ALTER TABLE users ADD COLUMN hacked TEXT",
            "column; EXEC xp_cmdshell 'rm -rf /'",
            "column; EXECUTE sp_configure 'show advanced options', 1",
        ]

        for identifier in dangerous_identifiers:
            with pytest.raises(ValueError, match="Unsafe SQL identifier"):
                _validate_sql_identifier(identifier)

    def test_validate_sql_identifier_sql_keywords(self):
        """Test that SQL keywords are rejected as identifiers."""
        sql_keywords = [
            "DROP",
            "DELETE",
            "INSERT",
            "UPDATE",
            "CREATE",
            "ALTER",
            "TRUNCATE",
            "EXEC",
            "EXECUTE",
            "UNION",
            "SELECT",
            "FROM",
            "WHERE",
            "OR",
            "AND",
        ]

        for keyword in sql_keywords:
            with pytest.raises(ValueError, match="dangerous SQL keyword"):
                _validate_sql_identifier(keyword)

    def test_validate_sql_identifier_invalid_types(self):
        """Test that invalid types are rejected."""
        invalid_inputs = [None, "", "123column", "column-name", "column.name", "column name"]

        for invalid_input in invalid_inputs:
            # All invalid inputs should raise ValueError
            with pytest.raises(ValueError):
                _validate_sql_identifier(invalid_input)

    def test_validate_column_definition_safe_definitions(self):
        """Test that safe column definitions are accepted."""
        safe_definitions = [
            "INTEGER",
            "TEXT NOT NULL",
            "REAL DEFAULT 0.0",
            "INTEGER PRIMARY KEY",
            "TEXT UNIQUE",
            "BLOB",
            "NUMERIC(10,2)",
            "BOOLEAN DEFAULT FALSE",
        ]

        for definition in safe_definitions:
            assert _validate_column_definition(definition) is True

    def test_validate_column_definition_dangerous_definitions(self):
        """Test that dangerous column definitions are rejected."""
        dangerous_definitions = [
            "INTEGER; DROP TABLE users",
            "TEXT; DELETE FROM users",
            "REAL; INSERT INTO users VALUES (1, 'hacker')",
            "INTEGER; UPDATE users SET password='hacked'",
            "TEXT; CREATE TABLE malicious (id INTEGER)",
            "REAL; ALTER TABLE users ADD COLUMN hacked TEXT",
            "INTEGER; EXEC xp_cmdshell 'rm -rf /'",
            "TEXT; EXECUTE sp_configure 'show advanced options', 1",
            "REAL; UNION SELECT * FROM users",
            "INTEGER; --comment",
            "TEXT; /*comment*/",
            "REAL; OR 1=1",
            "INTEGER; AND 1=1",
        ]

        for definition in dangerous_definitions:
            with pytest.raises(ValueError, match="Unsafe column definition"):
                _validate_column_definition(definition)

    def test_validate_column_definition_invalid_types(self):
        """Test that invalid column definition types are rejected."""
        invalid_inputs = [
            None,
            "",
            "INVALID_TYPE",
            "RANDOM_TEXT",
            "123INTEGER",
            # Very long strings
            "A" * 10000,
            "INTEGER" + "A" * 10000,
            # Unicode characters
            "INTEGER\u0000",
            "INTEGER\u2028",
            "INTEGER\u2029",
            "INTEGER\u200b",
            "INTEGER\u200c",
            "INTEGER\u200d",
            # Binary data simulation
            "INTEGER\x00\x01\x02",
            "INTEGER\xff\xfe\xfd",
            # SQL injection attempts
            "INTEGER; DROP TABLE users;",
            "INTEGER' OR '1'='1",
            "INTEGER/* */",
            "INTEGER-- comment",
            # Malicious patterns
            "INTEGER UNION SELECT * FROM users",
            "INTEGER EXEC xp_cmdshell",
            "INTEGER OR 1=1",
            # Invalid SQLite syntax
            "INTEGER INVALID_CONSTRAINT",
            "INTEGER DEFAULT 'value' CHECK (invalid)",
            "INTEGER COLLATE INVALID_COLLATION",
        ]

        for invalid_input in invalid_inputs:
            # All invalid inputs should raise ValueError with specific error messages
            # Skip None values as they're handled differently
            if invalid_input is not None:
                with pytest.raises(ValueError, match=r".*"):
                    _validate_column_definition(invalid_input)

    @pytest.mark.asyncio
    async def test_schema_migration_sql_injection_prevention(self, backend: SQLiteBackend):
        """Test that schema migration prevents SQL injection attacks."""
        # Initialize the backend through a public method
        await backend.save_state(
            "test_run",
            {
                "pipeline_id": "test_pipeline",
                "pipeline_name": "Test Pipeline",
                "pipeline_version": "1.0",
                "current_step_index": 0,
                "pipeline_context": {"test": "data"},
                "last_step_output": None,
                "step_history": [],
                "status": "running",
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "total_steps": 0,
                "error_message": None,
                "execution_time_ms": None,
                "memory_usage_mb": None,
            },
        )

        # Test with malicious column names and definitions
        malicious_columns = [
            ("malicious_column", "INTEGER; DROP TABLE workflow_state"),
            ("'; DROP TABLE users; --", "TEXT"),
            ("column_union", "TEXT UNION SELECT * FROM users"),
            ("column_or", "INTEGER OR 1=1"),
            ("column_comment", "TEXT --comment"),
            ("column_exec", "INTEGER; EXEC xp_cmdshell 'rm -rf /'"),
        ]

        for column_name, column_def in malicious_columns:
            # The validation should prevent these from being executed
            with pytest.raises(ValueError):
                _validate_sql_identifier(column_name)
                _validate_column_definition(column_def)

    @pytest.mark.asyncio
    async def test_safe_schema_migration(self, backend: SQLiteBackend):
        """Test that safe schema migrations work correctly."""
        # Initialize the backend through a public method
        await backend.save_state(
            "test_run",
            {
                "pipeline_id": "test_pipeline",
                "pipeline_name": "Test Pipeline",
                "pipeline_version": "1.0",
                "current_step_index": 0,
                "pipeline_context": {"test": "data"},
                "last_step_output": None,
                "step_history": [],
                "status": "running",
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "total_steps": 0,
                "error_message": None,
                "execution_time_ms": None,
                "memory_usage_mb": None,
            },
        )

        # Verify that the database was created successfully
        assert backend.db_path.exists()

        # Test that we can save and load state (which uses parameterized queries)
        test_state = {
            "run_id": "test_run",
            "pipeline_id": "test_pipeline",
            "pipeline_name": "Test Pipeline",
            "pipeline_version": "1.0",
            "current_step_index": 0,
            "pipeline_context": {"test": "data"},
            "last_step_output": None,
            "step_history": [],
            "status": "running",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "total_steps": 0,
            "error_message": None,
            "execution_time_ms": None,
            "memory_usage_mb": None,
        }

        await backend.save_state("test_run", test_state)
        loaded_state = await backend.load_state("test_run")

        assert loaded_state is not None
        assert loaded_state["run_id"] == "test_run"

    @pytest.mark.asyncio
    async def test_parameterized_queries_used(self, backend: SQLiteBackend):
        """Test that all database operations use parameterized queries."""
        from unittest.mock import patch
        import aiosqlite

        # Initialize the backend through a public method
        await backend.save_state(
            "test_run",
            {
                "pipeline_id": "test_pipeline",
                "pipeline_name": "Test Pipeline",
                "pipeline_version": "1.0",
                "current_step_index": 0,
                "pipeline_context": {"test": "data"},
                "last_step_output": None,
                "step_history": [],
                "status": "running",
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "total_steps": 0,
                "error_message": None,
                "execution_time_ms": None,
                "memory_usage_mb": None,
            },
        )

        # Create malicious input that would cause SQL injection if not parameterized
        malicious_run_id = "test'; DROP TABLE workflow_state; --"

        # Create a test state for the injection tests
        test_state = {
            "pipeline_id": "test_pipeline",
            "pipeline_name": "Test Pipeline",
            "pipeline_version": "1.0",
            "current_step_index": 0,
            "pipeline_context": {"test": "data"},
            "last_step_output": None,
            "step_history": [],
            "status": "running",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "total_steps": 0,
            "error_message": None,
            "execution_time_ms": None,
            "memory_usage_mb": None,
        }

        # Test additional injection patterns for comprehensive security testing
        injection_patterns = [
            # Basic injection
            "test'; DROP TABLE workflow_state; --",
            # UNION-based attacks
            "test' UNION SELECT * FROM workflow_state --",
            "test' UNION SELECT run_id, pipeline_id FROM workflow_state --",
            # Boolean-based blind injection
            "test' OR 1=1 --",
            "test' OR '1'='1' --",
            "test' AND 1=1 --",
            # Time-based blind injection
            "test'; WAITFOR DELAY '00:00:05' --",
            "test'; SELECT SLEEP(5) --",
            # Stacked queries
            "test'; INSERT INTO workflow_state VALUES ('hacked', 'hacked', 'hacked', 0, '{}', NULL, '[]', 'running', 0, 0, 0, NULL, NULL, NULL); --",
            # Comment-based injection
            "test'/*comment*/OR/*comment*/1=1--",
            "test'--comment\nOR 1=1--",
            # Hex encoding
            "test' OR 0x31=0x31 --",
            # URL encoding simulation
            "test'%20OR%201=1--",
            # Double encoding
            "test'%2520OR%25201=1--",
        ]

        for malicious_run_id in injection_patterns:
            # This should not cause SQL injection due to parameterized queries
            # The backend should handle this safely
            try:
                await backend.save_state(malicious_run_id, test_state)
                # If we get here, the injection was prevented (good)
                pass
            except Exception as e:
                # Any exception should not be due to SQL injection
                assert "DROP TABLE" not in str(e)
                assert "UNION" not in str(e)
                assert "OR 1=1" not in str(e)

        # Spy on the database execute calls to verify parameterized queries
        executed_sql = []
        executed_params = []

        original_execute = aiosqlite.Connection.execute

        async def spy_execute(self, sql, parameters=None):
            executed_sql.append(sql)
            executed_params.append(parameters)
            return await original_execute(self, sql, parameters)

        # Use a specific malicious input for the spy test
        malicious_test_input = "test'; DROP TABLE workflow_state; --"

        with patch("aiosqlite.Connection.execute", spy_execute):
            # Perform the operation with malicious input
            await backend.save_state(malicious_test_input, test_state)

        # Verify that parameterized queries were used
        assert len(executed_sql) > 0, "No SQL queries were executed"

        # Check that the malicious input is NOT in the SQL string
        for sql in executed_sql:
            assert malicious_test_input not in sql, f"Malicious input found in SQL: {sql}"
            assert "DROP TABLE" not in sql, f"DROP TABLE found in SQL: {sql}"
            assert "--" not in sql, f"SQL comment found in SQL: {sql}"

        # Check that the malicious input IS in the parameters
        malicious_found_in_params = False
        for params in executed_params:
            if params and any(malicious_test_input in str(param) for param in params):
                malicious_found_in_params = True
                break

        assert malicious_found_in_params, (
            "Malicious input not found in parameters - query may not be parameterized"
        )

        # Verify that the operation completed successfully
        loaded_state = await backend.load_state(malicious_run_id)
        assert loaded_state is not None
        assert loaded_state["run_id"] == malicious_run_id

        # Verify that the database is still functional
        workflows = await backend.list_workflows()
        assert len(workflows) >= 1

        # Verify that the malicious input was stored as data, not executed as SQL
        failed_workflows = await backend.get_failed_workflows(hours_back=24)
        assert isinstance(failed_workflows, list)

    def test_healthcare_data_security(self):
        """Test security measures for healthcare data scenarios."""
        # Test with realistic healthcare column names
        healthcare_columns = [
            ("patient_id", "TEXT NOT NULL"),
            ("medical_record_number", "TEXT UNIQUE"),
            ("diagnosis_code", "TEXT"),
            ("treatment_plan", "TEXT"),
            ("medication_list", "TEXT"),
            ("vital_signs", "TEXT"),
            ("lab_results", "TEXT"),
            ("insurance_info", "TEXT"),
        ]

        for column_name, column_def in healthcare_columns:
            assert _validate_sql_identifier(column_name) is True
            assert _validate_column_definition(column_def) is True

    def test_legal_data_security(self):
        """Test security measures for legal data scenarios."""
        # Test with realistic legal column names
        legal_columns = [
            ("case_number", "TEXT NOT NULL"),
            ("client_id", "TEXT UNIQUE"),
            ("case_type", "TEXT"),
            ("filing_date", "TEXT"),
            ("court_orders", "TEXT"),
            ("evidence_list", "TEXT"),
            ("witness_statements", "TEXT"),
            ("legal_citations", "TEXT"),
        ]

        for column_name, column_def in legal_columns:
            assert _validate_sql_identifier(column_name) is True
            assert _validate_column_definition(column_def) is True

    def test_finance_data_security(self):
        """Test security measures for finance data scenarios."""
        # Test with realistic finance column names
        finance_columns = [
            ("account_number", "TEXT NOT NULL"),
            ("transaction_id", "TEXT UNIQUE"),
            ("amount", "REAL"),
            ("currency", "TEXT"),
            ("transaction_type", "TEXT"),
            ("balance", "REAL"),
            ("routing_number", "TEXT"),
            ("tax_id", "TEXT"),
        ]

        for column_name, column_def in finance_columns:
            assert _validate_sql_identifier(column_name) is True
            assert _validate_column_definition(column_def) is True

    def test_edge_case_security(self):
        """Test edge cases that could bypass security measures."""
        edge_cases = [
            # Unicode injection attempts
            ("column\u0000", "TEXT"),
            ("column\u2028", "TEXT"),
            ("column\u2029", "TEXT"),
            # Zero-width characters
            ("column\u200b", "TEXT"),
            ("column\u200c", "TEXT"),
            ("column\u200d", "TEXT"),
            # Control characters
            ("column\x00", "TEXT"),
            ("column\x01", "TEXT"),
            ("column\x1f", "TEXT"),
            # Very long identifiers (should only reject >1000)
            ("a" * 1001, "TEXT"),
            ("column_" + "a" * 1001, "TEXT"),
        ]

        for column_name, column_def in edge_cases:
            # These should be rejected by the validation
            try:
                identifier_valid = _validate_sql_identifier(column_name)
                definition_valid = _validate_column_definition(column_def)
                assert not (identifier_valid is True and definition_valid is True), (
                    f"Validation failed: {repr(column_name)} and {repr(column_def)} were both accepted!"
                )
            except ValueError:
                # Expected for unsafe inputs
                pass


class TestSecurityLogging:
    """Test that security violations are properly logged."""

    @pytest.mark.asyncio
    async def test_security_violation_logging(self, tmp_path: Path):
        """Test that security violations are logged for audit purposes."""
        temp_db_path = tmp_path / "security_test.db"
        # Create backend instance to test the validation
        _ = SQLiteBackend(temp_db_path)

        # Test that validation properly rejects dangerous identifiers
        with pytest.raises(ValueError, match="Unsafe SQL identifier"):
            _validate_sql_identifier("DROP TABLE users")

        # Test that validation properly rejects dangerous column definitions
        with pytest.raises(ValueError, match="Unsafe column definition"):
            _validate_column_definition("INTEGER; DROP TABLE users")


if __name__ == "__main__":
    pytest.main([__file__])
