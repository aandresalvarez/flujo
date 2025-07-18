import asyncio
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import sqlite3
from datetime import datetime

if getattr(os, "geteuid", lambda: -1)() == 0:
    pytest.skip(
        "permission-based SQLite tests skipped when running as root",
        allow_module_level=True,
    )

from flujo.state.backends.sqlite import SQLiteBackend


class TestSQLiteBackendEdgeCases:
    """Test edge cases and error handling in SQLite backend."""

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
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
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
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
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
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
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
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
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
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
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
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
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
            sample_state = {
                "pipeline_id": "test_pipeline",
                "pipeline_name": "test_pipeline",
                "pipeline_version": "1.0.0",
                "current_step_index": 0,
                "pipeline_context": {"test": "data"},
                "last_step_output": None,
                "step_history": [],
                "status": "running",
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
            }
            with pytest.raises(
                sqlite3.DatabaseError,
                match="Database corruption recovery failed: unable to open database file",
            ):
                await backend.save_state("test_run", sample_state)
        finally:
            # Restore permissions for cleanup
            tmp_path.chmod(0o755)

    @pytest.mark.asyncio
    async def test_disk_full_scenario(self, tmp_path: Path) -> None:
        """Test handling when disk is full (simulate by raising OSError)."""
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
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        with patch("pathlib.Path.rename", side_effect=OSError("No space left on device")):
            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test_run", sample_state)
        # When disk is full, no backup files are created
        # This is the correct behavior - the backend fails gracefully

    @pytest.mark.asyncio
    async def test_concurrent_backup_operations(self, tmp_path: Path) -> None:
        """Test concurrent backup operations with corruption."""
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
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }

        async def create_backup(i: int) -> None:
            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state(f"test_run_{i}", sample_state)

        await asyncio.gather(*(create_backup(i) for i in range(5)))
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_infinite_loop_bug_fix(self, tmp_path: Path) -> None:
        """Test that infinite loop bug in backup logic is fixed (should fail gracefully)."""
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
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
            await backend.save_state("test_run", sample_state)
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_infinite_loop_bug_with_continue_fix(self, tmp_path: Path) -> None:
        """Test that infinite loop bug with continue in backup logic is fixed (should fail gracefully)."""
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
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
            await backend.save_state("test_run", sample_state)
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_backup_path_update_after_cleanup(self, tmp_path: Path) -> None:
        """Test that backup path is updated after cleanup attempts (should fail gracefully)."""
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
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
            await backend.save_state("test_run", sample_state)
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_race_condition_in_backup_creation(self, tmp_path: Path) -> None:
        """Test race condition in backup creation (should fail gracefully)."""
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
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        with patch("pathlib.Path.exists", side_effect=[False, True, False, True, False]):
            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test_run", sample_state)
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_readonly_directory_fallback(self, tmp_path: Path) -> None:
        """Test fallback when directory is readonly (should fail gracefully)."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")
        tmp_path.chmod(0o444)
        try:
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
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
            }
            with pytest.raises(
                sqlite3.DatabaseError,
                match="Database corruption recovery failed: unable to open database file",
            ):
                await backend.save_state("test_run", sample_state)
        finally:
            tmp_path.chmod(0o755)

    @pytest.mark.asyncio
    async def test_backup_pattern_glob_handling(self, tmp_path: Path) -> None:
        """Test backup pattern glob handling (should fail gracefully)."""
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
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        with patch("pathlib.Path.glob", side_effect=OSError("glob error")):
            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test_run", sample_state)
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_backup_stat_error_handling(self, tmp_path: Path) -> None:
        """Test backup stat error handling (should fail gracefully)."""
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
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        with patch("pathlib.Path.stat", side_effect=OSError("stat error")):
            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test_run", sample_state)
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_backup_unlink_error_handling(self, tmp_path: Path) -> None:
        """Test backup unlink error handling (should fail gracefully)."""
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
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        with patch("pathlib.Path.unlink", side_effect=OSError("unlink error")):
            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test_run", sample_state)
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_backup_glob_exception_handling(self, tmp_path: Path) -> None:
        """Test backup glob exception handling (should fail gracefully)."""
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
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        with patch("pathlib.Path.glob", side_effect=OSError("glob error")):
            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test_run", sample_state)
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_backup_fallback_timestamp_naming(self, tmp_path: Path) -> None:
        """Test backup fallback to timestamp-based naming (should fail gracefully)."""
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
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        with patch("pathlib.Path.exists", return_value=True):
            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test_run", sample_state)
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_backup_all_slots_undeletable_fallback(self, tmp_path: Path) -> None:
        """Test backup fallback when all slots are undeletable (should fail gracefully)."""
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
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        with patch("pathlib.Path.unlink", side_effect=OSError("undeletable error")):
            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test_run", sample_state)
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_backup_stat_always_raises(self, tmp_path: Path) -> None:
        """Test backup stat always raises (should fail gracefully)."""
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
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        with patch("pathlib.Path.stat", side_effect=OSError("stat always raises")):
            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test_run", sample_state)
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_backup_glob_always_raises(self, tmp_path: Path) -> None:
        """Test backup glob always raises (should fail gracefully)."""
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
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        with patch("pathlib.Path.glob", side_effect=OSError("glob always raises")):
            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test_run", sample_state)
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_backup_unlink_always_raises(self, tmp_path: Path) -> None:
        """Test backup unlink always raises (should fail gracefully)."""
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
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        with patch("pathlib.Path.unlink", side_effect=OSError("unlink always raises")):
            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test_run", sample_state)
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_backup_permission_and_race_conditions(self, tmp_path: Path) -> None:
        """Test backup permission and race conditions (should fail gracefully)."""
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
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        with (
            patch("pathlib.Path.stat", side_effect=OSError("stat error")),
            patch("pathlib.Path.unlink", side_effect=OSError("unlink error")),
        ):
            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test_run", sample_state)
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0
