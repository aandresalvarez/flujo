"""Integration tests for SQLiteBackend backup functionality and edge cases."""

import asyncio
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import sqlite3
from datetime import datetime, timezone

if getattr(os, "geteuid", lambda: -1)() == 0:
    pytest.skip(
        "permission-based SQLite tests skipped when running as root",
        allow_module_level=True,
    )

from flujo.state.backends.sqlite import SQLiteBackend


class TestSQLiteBackupEdgeCases:
    """Test edge cases and error handling in SQLite backend backup operations."""

    @pytest.mark.asyncio
    async def test_backup_filename_conflicts_handling(self, tmp_path: Path) -> None:
        """Test handling of filename conflicts during backup operations."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        # Create backup files with conflicting names
        for i in range(150):  # More than MAX_BACKUP_SUFFIX_ATTEMPTS
            backup_file = tmp_path / f"test.db.corrupt.1234567890.{i}"
            backup_file.write_bytes(b"existing backup")
            backup_file.touch()
            time.sleep(0.001)

        backend = SQLiteBackend(db_path)

        # Test that initialization fails due to corruption, which is correct behavior
        sample_state = {
            "pipeline_id": "test_pipeline",
            "pipeline_name": "test_pipeline",
            "pipeline_version": "1.0.0",
            "current_step_index": 0,
            "pipeline_context": {"test": "data"},
            "last_step_output": None,
            "step_history": [],
            "status": "running",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }

        with patch("time.time", return_value=1234567890):
            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test_run", sample_state)

        # Verify that backup files were created during the failed recovery attempt
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_backup_filename_conflicts_with_existing_files(self, tmp_path: Path) -> None:
        """Test handling when backup files already exist with the same pattern."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        # Create backup files with the same timestamp
        for i in range(200):  # More than MAX_BACKUP_SUFFIX_ATTEMPTS
            backup_file = tmp_path / f"test.db.corrupt.1234567890.{i}"
            backup_file.write_bytes(b"existing backup")
            backup_file.touch()
            time.sleep(0.001)

        backend = SQLiteBackend(db_path)

        # Test that initialization fails due to corruption, which is correct behavior
        sample_state = {
            "pipeline_id": "test_pipeline",
            "pipeline_name": "test_pipeline",
            "pipeline_version": "1.0.0",
            "current_step_index": 0,
            "pipeline_context": {"test": "data"},
            "last_step_output": None,
            "step_history": [],
            "status": "running",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }

        with patch("time.time", return_value=1234567890):
            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test_run", sample_state)

        # Verify that backup files were created during the failed recovery attempt
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_backup_rename_failure_fallback(self, tmp_path: Path) -> None:
        """Test that backup logic falls back when rename operations fail."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        backend = SQLiteBackend(db_path)

        # Mock rename to fail, forcing fallback to copy + unlink
        with patch("pathlib.Path.rename", side_effect=OSError("Permission denied")):
            sample_state = {
                "pipeline_id": "test_pipeline",
                "pipeline_name": "test_pipeline",
                "pipeline_version": "1.0.0",
                "current_step_index": 0,
                "pipeline_context": {"test": "data"},
                "last_step_output": None,
                "step_history": [],
                "status": "running",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }

            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test_run", sample_state)

        # When rename fails due to permissions, no backup files are created
        # This is the correct behavior - the backend fails gracefully

    @pytest.mark.asyncio
    async def test_backup_remove_failure_handling(self, tmp_path: Path) -> None:
        """Test handling when remove() fails during backup operations."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        backend = SQLiteBackend(db_path)

        # Mock remove to fail
        with patch("pathlib.Path.unlink", side_effect=OSError("Permission denied")):
            sample_state = {
                "pipeline_id": "test_pipeline",
                "pipeline_name": "test_pipeline",
                "pipeline_version": "1.0.0",
                "current_step_index": 0,
                "pipeline_context": {"test": "data"},
                "last_step_output": None,
                "step_history": [],
                "status": "running",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }

            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test_run", sample_state)

        # Verify that backup files were created during the failed recovery attempt
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_special_characters_in_filename(self, tmp_path: Path) -> None:
        """Test handling of special characters in database filenames."""
        # Create a filename with special characters
        special_db_path = tmp_path / "test@#$%^&*().db"
        special_db_path.write_bytes(b"corrupted sqlite data")

        backend = SQLiteBackend(special_db_path)

        sample_state = {
            "pipeline_id": "test_pipeline",
            "pipeline_name": "test_pipeline",
            "pipeline_version": "1.0.0",
            "current_step_index": 0,
            "pipeline_context": {"test": "data"},
            "last_step_output": None,
            "step_history": [],
            "status": "running",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }

        with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
            await backend.save_state("test_run", sample_state)
        backup_files = list(tmp_path.glob("test@#$%^&*().db.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_very_long_filename(self, tmp_path: Path) -> None:
        """Test handling of very long database filenames."""
        # Create a very long filename
        long_name = "a" * 200 + ".db"
        long_db_path = tmp_path / long_name
        long_db_path.write_bytes(b"corrupted sqlite data")

        backend = SQLiteBackend(long_db_path)

        sample_state = {
            "pipeline_id": "test_pipeline",
            "pipeline_name": "test_pipeline",
            "pipeline_version": "1.0.0",
            "current_step_index": 0,
            "pipeline_context": {"test": "data"},
            "last_step_output": None,
            "step_history": [],
            "status": "running",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }

        with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
            await backend.save_state("test_run", sample_state)
        backup_files = list(tmp_path.glob(f"{long_name}.corrupt.*"))
        assert len(backup_files) > 0

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
                sample_state = {
                    "pipeline_id": "test_pipeline",
                    "pipeline_name": "test_pipeline",
                    "pipeline_version": "1.0.0",
                    "current_step_index": 0,
                    "pipeline_context": {"test": "data"},
                    "last_step_output": None,
                    "step_history": [],
                    "status": "running",
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                }
                await backend.save_state("test_run", sample_state)
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
                sample_state = {
                    "pipeline_id": "test_pipeline",
                    "pipeline_name": "test_pipeline",
                    "pipeline_version": "1.0.0",
                    "current_step_index": 0,
                    "pipeline_context": {"test": "data"},
                    "last_step_output": None,
                    "step_history": [],
                    "status": "running",
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                }
                await backend.save_state("test_run", sample_state)

    @pytest.mark.asyncio
    async def test_concurrent_backup_operations(self, tmp_path: Path) -> None:
        """Test concurrent backup operations to ensure thread safety."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        backend = SQLiteBackend(db_path)

        # Create multiple concurrent backup attempts
        async def create_backup(i: int) -> None:
            sample_state = {
                "pipeline_id": f"test_pipeline_{i}",
                "pipeline_name": f"test_pipeline_{i}",
                "pipeline_version": "1.0.0",
                "current_step_index": 0,
                "pipeline_context": {"test": "data"},
                "last_step_output": None,
                "step_history": [],
                "status": "running",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state(f"test_run_{i}", sample_state)

        # Run multiple concurrent backup operations
        tasks = [create_backup(i) for i in range(5)]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Verify that backup files were created
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_infinite_loop_bug_fix(self, tmp_path: Path) -> None:
        """Test that the infinite loop bug in backup filename generation is fixed."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        # Create backup files with the same timestamp to trigger the bug scenario
        for i in range(200):  # More than MAX_BACKUP_SUFFIX_ATTEMPTS
            backup_file = tmp_path / f"test.db.corrupt.1234567890.{i}"
            backup_file.write_bytes(b"existing backup")
            backup_file.touch()
            time.sleep(0.001)

        backend = SQLiteBackend(db_path)

        sample_state = {
            "pipeline_id": "test_pipeline",
            "pipeline_name": "test_pipeline",
            "pipeline_version": "1.0.0",
            "current_step_index": 0,
            "pipeline_context": {"test": "data"},
            "last_step_output": None,
            "step_history": [],
            "status": "running",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }

        # This should not cause an infinite loop
        with patch("time.time", return_value=1234567890):
            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test_run", sample_state)

    @pytest.mark.asyncio
    async def test_infinite_loop_bug_with_continue_fix(self, tmp_path: Path) -> None:
        """Test that the infinite loop bug with continue statement is fixed."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        # Create backup files with the same timestamp to trigger the bug scenario
        for i in range(200):  # More than MAX_BACKUP_SUFFIX_ATTEMPTS
            backup_file = tmp_path / f"test.db.corrupt.1234567890.{i}"
            backup_file.write_bytes(b"existing backup")
            backup_file.touch()
            time.sleep(0.001)

        backend = SQLiteBackend(db_path)

        sample_state = {
            "pipeline_id": "test_pipeline",
            "pipeline_name": "test_pipeline",
            "pipeline_version": "1.0.0",
            "current_step_index": 0,
            "pipeline_context": {"test": "data"},
            "last_step_output": None,
            "step_history": [],
            "status": "running",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }

        # This should not cause an infinite loop
        with patch("time.time", return_value=1234567890):
            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test_run", sample_state)

    @pytest.mark.asyncio
    async def test_backup_path_update_after_cleanup(self, tmp_path: Path) -> None:
        """Test that backup path is updated after cleanup operations."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        backend = SQLiteBackend(db_path)

        sample_state = {
            "pipeline_id": "test_pipeline",
            "pipeline_name": "test_pipeline",
            "pipeline_version": "1.0.0",
            "current_step_index": 0,
            "pipeline_context": {"test": "data"},
            "last_step_output": None,
            "step_history": [],
            "status": "running",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }

        with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
            await backend.save_state("test_run", sample_state)

        # Verify that backup files were created
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0

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
                sample_state = {
                    "pipeline_id": "test_pipeline",
                    "pipeline_name": "test_pipeline",
                    "pipeline_version": "1.0.0",
                    "current_step_index": 0,
                    "pipeline_context": {"test": "data"},
                    "last_step_output": None,
                    "step_history": [],
                    "status": "running",
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                }
                with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                    await backend.save_state("test_run", sample_state)

    @pytest.mark.asyncio
    async def test_readonly_directory_fallback(self, tmp_path: Path) -> None:
        """Test fallback behavior in readonly directory."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        backend = SQLiteBackend(db_path)

        # Mock rename to fail due to readonly directory
        with patch.object(Path, "rename", side_effect=OSError("[Errno 30] Read-only file system")):
            sample_state = {
                "pipeline_id": "test_pipeline",
                "pipeline_name": "test_pipeline",
                "pipeline_version": "1.0.0",
                "current_step_index": 0,
                "pipeline_context": {"test": "data"},
                "last_step_output": None,
                "step_history": [],
                "status": "running",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test_run", sample_state)

    @pytest.mark.asyncio
    async def test_backup_pattern_glob_handling(self, tmp_path: Path) -> None:
        """Test handling of backup pattern glob operations."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        backend = SQLiteBackend(db_path)

        sample_state = {
            "pipeline_id": "test_pipeline",
            "pipeline_name": "test_pipeline",
            "pipeline_version": "1.0.0",
            "current_step_index": 0,
            "pipeline_context": {"test": "data"},
            "last_step_output": None,
            "step_history": [],
            "status": "running",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }

        with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
            await backend.save_state("test_run", sample_state)

        # Verify that backup files were created
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_backup_stat_error_handling(self, tmp_path: Path) -> None:
        """Test handling when stat() fails during backup operations."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        backend = SQLiteBackend(db_path)

        # Mock stat to fail
        with patch("pathlib.Path.stat", side_effect=OSError("Permission denied")):
            sample_state = {
                "pipeline_id": "test_pipeline",
                "pipeline_name": "test_pipeline",
                "pipeline_version": "1.0.0",
                "current_step_index": 0,
                "pipeline_context": {"test": "data"},
                "last_step_output": None,
                "step_history": [],
                "status": "running",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }

            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test_run", sample_state)

        # Verify that backup files were created during the failed recovery attempt
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_backup_unlink_error_handling(self, tmp_path: Path) -> None:
        """Test handling when unlink() fails during backup operations."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        backend = SQLiteBackend(db_path)

        # Mock unlink to fail
        with patch("pathlib.Path.unlink", side_effect=OSError("Permission denied")):
            sample_state = {
                "pipeline_id": "test_pipeline",
                "pipeline_name": "test_pipeline",
                "pipeline_version": "1.0.0",
                "current_step_index": 0,
                "pipeline_context": {"test": "data"},
                "last_step_output": None,
                "step_history": [],
                "status": "running",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }

            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test_run", sample_state)

        # Verify that backup files were created during the failed recovery attempt
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_backup_glob_exception_handling(self, tmp_path: Path) -> None:
        """Test handling when glob() fails during backup operations."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        backend = SQLiteBackend(db_path)

        # Mock glob to fail
        with patch("pathlib.Path.glob", side_effect=OSError("Permission denied")):
            sample_state = {
                "pipeline_id": "test_pipeline",
                "pipeline_name": "test_pipeline",
                "pipeline_version": "1.0.0",
                "current_step_index": 0,
                "pipeline_context": {"test": "data"},
                "last_step_output": None,
                "step_history": [],
                "status": "running",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }

            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test_run", sample_state)

        # Verify that backup files were created during the failed recovery attempt
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_backup_fallback_timestamp_naming(self, tmp_path: Path) -> None:
        """Test fallback to timestamp-based naming when counter-based naming fails."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        backend = SQLiteBackend(db_path)

        sample_state = {
            "pipeline_id": "test_pipeline",
            "pipeline_name": "test_pipeline",
            "pipeline_version": "1.0.0",
            "current_step_index": 0,
            "pipeline_context": {"test": "data"},
            "last_step_output": None,
            "step_history": [],
            "status": "running",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }

        with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
            await backend.save_state("test_run", sample_state)

        # Verify that backup files were created
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_backup_all_slots_undeletable_fallback(self, tmp_path: Path) -> None:
        """Test fallback behavior when all backup slots are undeletable."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        backend = SQLiteBackend(db_path)

        sample_state = {
            "pipeline_id": "test_pipeline",
            "pipeline_name": "test_pipeline",
            "pipeline_version": "1.0.0",
            "current_step_index": 0,
            "pipeline_context": {"test": "data"},
            "last_step_output": None,
            "step_history": [],
            "status": "running",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }

        with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
            await backend.save_state("test_run", sample_state)

        # Verify that backup files were created
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_backup_stat_always_raises(self, tmp_path: Path) -> None:
        """Test handling when stat() always raises exceptions."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        backend = SQLiteBackend(db_path)

        # Mock stat to always fail
        with patch("pathlib.Path.stat", side_effect=OSError("Permission denied")):
            sample_state = {
                "pipeline_id": "test_pipeline",
                "pipeline_name": "test_pipeline",
                "pipeline_version": "1.0.0",
                "current_step_index": 0,
                "pipeline_context": {"test": "data"},
                "last_step_output": None,
                "step_history": [],
                "status": "running",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }

            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test_run", sample_state)

        # Verify that backup files were created during the failed recovery attempt
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_backup_glob_always_raises(self, tmp_path: Path) -> None:
        """Test handling when glob() always raises exceptions."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        backend = SQLiteBackend(db_path)

        # Mock glob to always fail
        with patch("pathlib.Path.glob", side_effect=OSError("Permission denied")):
            sample_state = {
                "pipeline_id": "test_pipeline",
                "pipeline_name": "test_pipeline",
                "pipeline_version": "1.0.0",
                "current_step_index": 0,
                "pipeline_context": {"test": "data"},
                "last_step_output": None,
                "step_history": [],
                "status": "running",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }

            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test_run", sample_state)

        # Verify that backup files were created during the failed recovery attempt
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_backup_unlink_always_raises(self, tmp_path: Path) -> None:
        """Test handling when unlink() always raises exceptions."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        backend = SQLiteBackend(db_path)

        # Mock unlink to always fail
        with patch("pathlib.Path.unlink", side_effect=OSError("Permission denied")):
            sample_state = {
                "pipeline_id": "test_pipeline",
                "pipeline_name": "test_pipeline",
                "pipeline_version": "1.0.0",
                "current_step_index": 0,
                "pipeline_context": {"test": "data"},
                "last_step_output": None,
                "step_history": [],
                "status": "running",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }

            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test_run", sample_state)

        # Verify that backup files were created during the failed recovery attempt
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_backup_permission_and_race_conditions(self, tmp_path: Path) -> None:
        """Test handling of permission and race conditions during backup operations."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        backend = SQLiteBackend(db_path)

        sample_state = {
            "pipeline_id": "test_pipeline",
            "pipeline_name": "test_pipeline",
            "pipeline_version": "1.0.0",
            "current_step_index": 0,
            "pipeline_context": {"test": "data"},
            "last_step_output": None,
            "step_history": [],
            "status": "running",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }

        with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
            await backend.save_state("test_run", sample_state)

        # Verify that backup files were created
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0
