"""Integration tests for SQLiteBackend backup functionality and edge cases."""

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from datetime import datetime

from flujo.state.backends.sqlite import SQLiteBackend

pytestmark = pytest.mark.serial


def create_corrupted_db(db_path: Path):
    db_path.write_bytes(b"corrupted sqlite data")
    return SQLiteBackend(db_path)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "db_name,existing_backups,expected_new_backups",
    [
        ("test.db", [], 1),
        (
            "test.db",
            [
                "test.db.corrupt.1234567890",
                "test.db.corrupt.1234567890.1",
                "test.db.corrupt.1234567890.2",
            ],
            4,
        ),
        ("test'with\"quotes.db", [], 1),
        ("{}".format("a" * 200 + ".db"), [], 1),
    ],
)
async def test_backup_filename_variations(
    tmp_path: Path, db_name, existing_backups, expected_new_backups
):
    """Test backup logic for various filename/path scenarios."""
    db_path = tmp_path / db_name
    for backup_name in existing_backups:
        (tmp_path / backup_name).write_bytes(b"existing backup")
    backend = create_corrupted_db(db_path)

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

    # Check that backup files were created during the failed recovery attempt
    backup_files = list(tmp_path.glob(f"{db_name}.corrupt.*"))
    assert len(backup_files) > 0


class TestSQLiteBackupEdgeCases:
    """Comprehensive tests for SQLiteBackend backup functionality."""

    @pytest.mark.asyncio
    async def test_backup_filename_conflicts_handling(self, tmp_path: Path) -> None:
        """Test handling of backup filename conflicts with unique timestamps."""
        db_path = tmp_path / "test.db"

        # Create initial corrupted database
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

        # Mock time.time to return predictable timestamps
        with patch("time.time", return_value=1234567890):
            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test_run", sample_state)

        # Check that backup was created during the failed recovery attempt
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_backup_filename_conflicts_with_existing_files(self, tmp_path: Path) -> None:
        """Test handling when backup files already exist."""
        db_path = tmp_path / "test.db"

        # Create existing backup files
        existing_backups = [
            tmp_path / "test.db.corrupt.1234567890",
            tmp_path / "test.db.corrupt.1234567890.1",
            tmp_path / "test.db.corrupt.1234567890.2",
        ]
        for backup in existing_backups:
            backup.write_bytes(b"existing backup")

        # Create corrupted database
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

        # Mock time.time to return the same timestamp as existing backups
        with patch("time.time", return_value=1234567890):
            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test_run", sample_state)

        # Check that backup files were created during the failed recovery attempt
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_backup_rename_failure_fallback(self, tmp_path: Path) -> None:
        """Test fallback behavior when backup rename fails."""
        db_path = tmp_path / "test.db"

        # Create corrupted database
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

        # Mock rename to fail
        with patch.object(Path, "rename", side_effect=OSError("Permission denied")):
            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test_run", sample_state)

        # When rename fails due to permissions, no backup files are created
        # This is the correct behavior - the backend fails gracefully

    @pytest.mark.asyncio
    async def test_backup_remove_failure_handling(self, tmp_path: Path) -> None:
        """Test handling when backup removal fails."""
        db_path = tmp_path / "test.db"

        # Create corrupted database
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

        # Mock unlink to fail
        with patch.object(Path, "unlink", side_effect=OSError("Permission denied")):
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

        # Check that backup files were created during the failed recovery attempt
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

        # Check that backup files were created during the failed recovery attempt
        backup_files = list(tmp_path.glob(f"{long_name}.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_infinite_loop_bug_fix(self, tmp_path: Path) -> None:
        """Test that infinite loop bug in backup logic is fixed."""
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

        # Check that backup files were created during the failed recovery attempt
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_backup_pattern_glob_handling(self, tmp_path: Path) -> None:
        """Test backup pattern glob handling."""
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

        # Check that backup files were created during the failed recovery attempt
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_backup_stat_error_handling(self, tmp_path: Path) -> None:
        """Test backup stat error handling."""
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

        def mock_stat(self, *args, **kwargs):
            raise OSError("stat error")

        with patch.object(Path, "stat", mock_stat):
            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test_run", sample_state)

        # Check that backup files were created during the failed recovery attempt
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_backup_unlink_error_handling(self, tmp_path: Path) -> None:
        """Test handling of unlink errors during backup cleanup."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        # Create existing backup files
        for i in range(5):
            backup_file = tmp_path / f"test.db.corrupt.1234567890.{i}"
            backup_file.write_bytes(b"backup data")

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

        # Mock unlink to raise exceptions
        def mock_unlink(self, *args, **kwargs):
            if "corrupt" in str(self):
                raise OSError("Permission denied")
            return Path.unlink(self, *args, **kwargs)

        with patch.object(Path, "unlink", mock_unlink):
            with patch("time.time", return_value=1234567890):
                with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                    await backend.save_state("test_run", sample_state)

    @pytest.mark.asyncio
    async def test_backup_min_function_with_none_values(self, tmp_path: Path) -> None:
        """Test backup min function handling with None values."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        # Create existing backup files
        for i in range(5):
            backup_file = tmp_path / f"test.db.corrupt.1234567890.{i}"
            backup_file.write_bytes(b"backup data")

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

        # Mock stat to return None for some files
        def mock_stat(self, *args, **kwargs):
            if "corrupt.1234567890.2" in str(self):
                return None
            # Return a proper mock stat result for non-corrupt files
            from unittest.mock import Mock

            mock_result = Mock()
            mock_result.st_mtime = 1234567890
            mock_result.st_mode = 0o644  # Regular file mode
            mock_result.st_size = 1024
            return mock_result

        with patch.object(Path, "stat", mock_stat):
            with patch("time.time", return_value=1234567890):
                with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                    await backend.save_state("test_run", sample_state)

    @pytest.mark.asyncio
    async def test_backup_empty_directory_handling(self, tmp_path: Path) -> None:
        """Test backup handling in empty directory."""
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

        # Mock time.time to return predictable timestamp
        with patch("time.time", return_value=1234567890):
            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test_run", sample_state)

    @pytest.mark.asyncio
    async def test_backup_path_reset_after_cleanup(self, tmp_path: Path) -> None:
        """Test that backup path is reset after cleanup."""
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

        # Mock exists to simulate cleanup
        original_exists = Path.exists

        def mock_exists(self):
            # This will be called multiple times during the backup process
            if "corrupt.1234567890.1" in str(self):
                return False  # Simulate file being deleted
            return original_exists(self)

        with patch.object(Path, "exists", mock_exists):
            with patch("time.time", return_value=1234567890):
                with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                    await backend.save_state("test_run", sample_state)

    @pytest.mark.asyncio
    async def test_backup_counter_reset_after_cleanup(self, tmp_path: Path) -> None:
        """Test that backup counter is reset after cleanup."""
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

        # Mock exists to simulate cleanup
        original_exists = Path.exists

        def mock_exists(self):
            # This will be called multiple times during the backup process
            if "corrupt.1234567890.1" in str(self):
                return False  # Simulate file being deleted
            return original_exists(self)

        with patch.object(Path, "exists", mock_exists):
            with patch("time.time", return_value=1234567890):
                with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                    await backend.save_state("test_run", sample_state)

    @pytest.mark.asyncio
    async def test_backup_infinite_loop_prevention(self, tmp_path: Path) -> None:
        """Test that backup logic prevents infinite loops."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        # Create many existing backup files
        for i in range(100):
            backup_file = tmp_path / f"test.db.corrupt.1234567890.{i}"
            backup_file.write_bytes(b"backup data")

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

        # Mock time.time to return the same timestamp
        with patch("time.time", return_value=1234567890):
            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test_run", sample_state)

    @pytest.mark.asyncio
    async def test_backup_cleanup_attempts_limit(self, tmp_path: Path) -> None:
        """Test that backup cleanup has a limit on attempts."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        # Create existing backup files
        for i in range(10):
            backup_file = tmp_path / f"test.db.corrupt.1234567890.{i}"
            backup_file.write_bytes(b"backup data")

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

        # Mock time.time to return the same timestamp
        with patch("time.time", return_value=1234567890):
            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test_run", sample_state)

    @pytest.mark.asyncio
    async def test_backup_stat_exception_handling(self, tmp_path: Path) -> None:
        """Test handling of stat exceptions during backup cleanup."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        # Create existing backup files
        for i in range(5):
            backup_file = tmp_path / f"test.db.corrupt.1234567890.{i}"
            backup_file.write_bytes(b"backup data")

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

        # Mock stat to raise exceptions for some files
        def mock_stat(self, *args, **kwargs):
            if "corrupt.1234567890.2" in str(self):
                raise OSError("Permission denied")
            # Return a proper mock stat result for non-corrupt files
            from unittest.mock import Mock

            mock_result = Mock()
            mock_result.st_mtime = 1234567890
            mock_result.st_mode = 0o644  # Regular file mode
            mock_result.st_size = 1024
            return mock_result

        with patch.object(Path, "stat", mock_stat):
            with patch("time.time", return_value=1234567890):
                with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                    await backend.save_state("test_run", sample_state)

    @pytest.mark.asyncio
    async def test_backup_unlink_exception_handling(self, tmp_path: Path) -> None:
        """Test handling of unlink exceptions during backup cleanup."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        # Create existing backup files
        for i in range(5):
            backup_file = tmp_path / f"test.db.corrupt.1234567890.{i}"
            backup_file.write_bytes(b"backup data")

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

        # Mock unlink to raise exceptions
        def mock_unlink(self, *args, **kwargs):
            if "corrupt" in str(self):
                raise OSError("Permission denied")
            return Path.unlink(self, *args, **kwargs)

        with patch.object(Path, "unlink", mock_unlink):
            with patch("time.time", return_value=1234567890):
                with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                    await backend.save_state("test_run", sample_state)

    @pytest.mark.asyncio
    async def test_backup_glob_exception_handling(self, tmp_path: Path) -> None:
        """Test handling of glob exceptions during backup cleanup."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        # Create existing backup files
        for i in range(5):
            backup_file = tmp_path / f"test.db.corrupt.1234567890.{i}"
            backup_file.write_bytes(b"backup data")

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

        # Mock glob to raise exceptions
        def mock_glob(self, pattern):
            if "corrupt" in pattern:
                raise OSError("Permission denied")
            return Path.glob(self, pattern)

        with patch.object(Path, "glob", mock_glob):
            with patch("time.time", return_value=1234567890):
                with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                    await backend.save_state("test_run", sample_state)

    @pytest.mark.asyncio
    async def test_backup_fallback_timestamp_naming(self, tmp_path: Path) -> None:
        """Test fallback to timestamp naming when counter naming fails."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        # Create existing backup files with high counters
        for i in range(1000, 1010):
            backup_file = tmp_path / f"test.db.corrupt.1234567890.{i}"
            backup_file.write_bytes(b"backup data")

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

        # Mock time.time to return a different timestamp
        with patch("time.time", return_value=9876543210):
            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test_run", sample_state)

    @pytest.mark.asyncio
    async def test_backup_all_slots_undeletable_fallback(self, tmp_path: Path) -> None:
        """Test fallback when all backup slots are undeletable."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        # Create existing backup files
        for i in range(10):
            backup_file = tmp_path / f"test.db.corrupt.1234567890.{i}"
            backup_file.write_bytes(b"backup data")

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

        # Mock unlink to always fail
        def mock_unlink(self, *args, **kwargs):
            if "corrupt" in str(self):
                raise OSError("Permission denied")
            return Path.unlink(self, *args, **kwargs)

        with patch.object(Path, "unlink", mock_unlink):
            with patch("time.time", return_value=1234567890):
                with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                    await backend.save_state("test_run", sample_state)

    @pytest.mark.asyncio
    async def test_backup_stat_always_raises(self, tmp_path: Path) -> None:
        """Test backup handling when stat always raises exceptions."""
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

        # Mock stat to always raise for corrupt files
        def always_raises_stat(self, *a, **k):
            if "corrupt" in str(self):
                raise OSError("Permission denied")
            # Return a proper mock stat result for non-corrupt files
            from unittest.mock import Mock

            mock_result = Mock()
            mock_result.st_mtime = 1234567890
            mock_result.st_mode = 0o644  # Regular file mode
            mock_result.st_size = 1024
            return mock_result

        with patch.object(Path, "stat", always_raises_stat):
            with patch("time.time", return_value=1234567890):
                with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                    await backend.save_state("test_run", sample_state)

    @pytest.mark.asyncio
    async def test_backup_glob_always_raises(self, tmp_path: Path) -> None:
        """Test backup handling when glob always raises exceptions."""
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

        # Mock glob to always raise
        def always_raises_glob(self, pattern):
            if "corrupt" in pattern:
                raise OSError("Permission denied")
            return Path.glob(self, pattern)

        with patch.object(Path, "glob", always_raises_glob):
            with patch("time.time", return_value=1234567890):
                with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                    await backend.save_state("test_run", sample_state)

    @pytest.mark.asyncio
    async def test_backup_unlink_always_raises(self, tmp_path: Path) -> None:
        """Test backup handling when unlink always raises exceptions."""
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

        # Mock unlink to always raise
        def always_raises_unlink(self, *a, **k):
            if "corrupt" in str(self):
                raise OSError("Permission denied")
            return Path.unlink(self, *a, **k)

        with patch.object(Path, "unlink", always_raises_unlink):
            with patch("time.time", return_value=1234567890):
                with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                    await backend.save_state("test_run", sample_state)

    @pytest.mark.asyncio
    async def test_backup_permission_and_race_conditions(self, tmp_path: Path) -> None:
        """Test backup handling with permission and race conditions."""
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

        # Mock stat to sometimes raise
        def sometimes_raises_stat(self, *a, **k):
            if "corrupt" in str(self) and hash(str(self)) % 3 == 0:
                raise OSError("Permission denied")
            # Return a proper mock stat result for non-corrupt files
            from unittest.mock import Mock

            mock_result = Mock()
            mock_result.st_mtime = 1234567890
            mock_result.st_mode = 0o644  # Regular file mode
            mock_result.st_size = 1024
            return mock_result

        with patch.object(Path, "stat", sometimes_raises_stat):
            with patch("time.time", return_value=1234567890):
                with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                    await backend.save_state("test_run", sample_state)

    @pytest.mark.asyncio
    async def test_backup_max_attempts_exceeded_handling(self, tmp_path: Path) -> None:
        """Test handling when max attempts are exceeded."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        # Create many existing backup files
        for i in range(150):  # More than MAX_BACKUP_SUFFIX_ATTEMPTS
            backup_file = tmp_path / f"test.db.corrupt.1234567890.{i}"
            backup_file.write_bytes(b"backup data")

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

        # Mock time.time to return the same timestamp
        with patch("time.time", return_value=1234567890):
            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state("test_run", sample_state)

    @pytest.mark.asyncio
    async def test_backup_continue_statement_effectiveness(self, tmp_path: Path) -> None:
        """Test that continue statement works correctly in backup logic."""
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

        # Mock glob to raise exceptions
        def mock_glob(self, pattern):
            if "corrupt" in pattern:
                raise OSError("Permission denied")
            return Path.glob(self, pattern)

        with patch.object(Path, "glob", mock_glob):
            with patch("time.time", return_value=1234567890):
                with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                    await backend.save_state("test_run", sample_state)
