"""Integration tests for SQLiteBackend filesystem edge cases."""

import os
import sqlite3
from pathlib import Path
from unittest.mock import patch
from datetime import datetime

import pytest

from flujo.state.backends.sqlite import SQLiteBackend

if getattr(os, "geteuid", lambda: -1)() == 0:
    pytest.skip(
        "permission-based SQLite tests skipped when running as root",
        allow_module_level=True,
    )

pytestmark = pytest.mark.serial


class TestSQLiteFilesystemEdgeCases:
    """Tests for SQLiteBackend filesystem edge cases."""

    def _get_complete_state(self) -> dict:
        """Create a complete state object that matches SQLite schema requirements."""
        now = datetime.utcnow().replace(microsecond=0)
        return {
            "pipeline_id": "test_pipeline",
            "pipeline_name": "test_pipeline",
            "pipeline_version": "1.0.0",
            "current_step_index": 0,
            "pipeline_context": {"test": "data"},
            "last_step_output": None,
            "step_history": [],
            "status": "running",
            "created_at": now,
            "updated_at": now,
            "total_steps": 0,
            "error_message": None,
            "execution_time_ms": None,
            "memory_usage_mb": None,
        }

    @pytest.mark.asyncio
    async def test_no_write_permissions(self, tmp_path: Path) -> None:
        """Test handling when directory has no write permissions."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        # Make directory read-only
        tmp_path.chmod(0o444)

        try:
            backend = SQLiteBackend(db_path)
            with pytest.raises((OSError, PermissionError, sqlite3.DatabaseError)):
                await backend.save_state("test", self._get_complete_state())
        finally:
            # Restore permissions
            tmp_path.chmod(0o755)

    @pytest.mark.asyncio
    async def test_disk_full_scenario(self, tmp_path: Path) -> None:
        """Test handling when disk is full during backup."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        backend = SQLiteBackend(db_path)

        # Mock disk full error during database initialization
        with patch("aiosqlite.connect", side_effect=OSError("[Errno 28] No space left on device")):
            with pytest.raises(OSError, match="No space left on device"):
                await backend.save_state("test", self._get_complete_state())

    @pytest.mark.asyncio
    async def test_readonly_directory_fallback(self, tmp_path: Path) -> None:
        """Test fallback behavior in readonly directory."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        backend = SQLiteBackend(db_path)

        # Mock rename to fail due to readonly directory
        with patch.object(Path, "rename", side_effect=OSError("[Errno 30] Read-only file system")):
            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test", self._get_complete_state())

        # Verify that the corrupted file was removed when backup creation failed
        assert not db_path.exists(), (
            "Corrupted file should have been removed when backup creation failed"
        )

        # Verify that no backup files were created (since rename failed)
        backup_files = list(tmp_path.glob("*.corrupt.*"))
        assert len(backup_files) == 0, "No backup files should be created when rename fails"

    @pytest.mark.asyncio
    async def test_race_condition_in_backup_creation(self, tmp_path: Path) -> None:
        """Test race condition handling in backup creation."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        backend = SQLiteBackend(db_path)

        # Mock exists to simulate race condition where backup files already exist
        def mock_exists(self):
            # Simulate file being created by another process
            if "corrupt.1234567890" in str(self):
                return True
            return Path.exists(self)

        with patch.object(Path, "exists", mock_exists):
            with patch("time.time", return_value=1234567890):
                with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                    await backend.save_state("test", self._get_complete_state())

        # Verify that the race condition handling attempted to create backup files
        # The system should have tried to create files with different suffixes
        backup_files = list(tmp_path.glob("*.corrupt.*"))

        # In a real race condition, the system might succeed in creating a backup
        # or it might fall back to removing the file. Both are valid outcomes.
        if len(backup_files) > 0:
            # If backup files were created, verify they follow the naming convention
            for backup_file in backup_files:
                assert "corrupt" in backup_file.name, (
                    f"Backup file {backup_file} doesn't follow naming convention"
                )
                # Use a flexible regex pattern to check backup file naming
                import re

                pattern = r".*\.db\.corrupt\.\d+\.\d+$"
                assert re.match(pattern, backup_file.name), (
                    f"Backup file {backup_file} doesn't have expected naming pattern"
                )
        else:
            # If no backup files were created, the corrupted file should have been removed
            assert not db_path.exists(), (
                "Corrupted file should have been removed when backup creation failed"
            )
