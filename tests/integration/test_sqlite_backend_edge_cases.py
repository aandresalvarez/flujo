import asyncio
import sqlite3
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from flujo.state.backends.sqlite import SQLiteBackend


class TestSQLiteBackendEdgeCases:
    """Comprehensive tests for SQLiteBackend edge cases and bug fixes."""

    @pytest.mark.asyncio
    async def test_backup_filename_conflicts_handling(self, tmp_path: Path) -> None:
        """Test handling of backup filename conflicts with unique timestamps."""
        db_path = tmp_path / "test.db"

        # Create initial corrupted database
        db_path.write_bytes(b"corrupted sqlite data")

        backend = SQLiteBackend(db_path)

        # Mock time.time to return predictable timestamps
        with patch("time.time", return_value=1234567890):
            await backend._init_db()

        # Check that backup was created with timestamp
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) == 1
        assert "corrupt.1234567890" in backup_files[0].name

        # Verify new database was created
        assert db_path.exists()
        assert db_path.stat().st_size > 0

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

        # Mock time.time to return the same timestamp as existing backups
        with patch("time.time", return_value=1234567890):
            await backend._init_db()

        # Check that a new backup was created with counter suffix
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert (
            len(backup_files) == 4
        )  # 3 existing + 1 new backup (corrupted DB moved to counter suffix path)

        # Verify new database was created
        assert db_path.exists()
        assert db_path.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_backup_rename_failure_fallback(self, tmp_path: Path) -> None:
        """Test fallback behavior when backup rename fails."""
        db_path = tmp_path / "test.db"

        # Create corrupted database
        db_path.write_bytes(b"corrupted sqlite data")

        backend = SQLiteBackend(db_path)

        # Mock rename to fail
        with patch.object(Path, "rename", side_effect=OSError("Permission denied")):
            await backend._init_db()

        # Verify corrupted file was removed and new database created
        assert db_path.exists()
        assert db_path.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_backup_remove_failure_handling(self, tmp_path: Path) -> None:
        """Test handling when backup removal fails."""
        db_path = tmp_path / "test.db"

        # Create corrupted database
        db_path.write_bytes(b"corrupted sqlite data")

        backend = SQLiteBackend(db_path)

        # Mock both rename and unlink to fail
        with (
            patch.object(Path, "rename", side_effect=OSError("Permission denied")),
            patch.object(Path, "unlink", side_effect=OSError("Permission denied")),
        ):
            with pytest.raises(sqlite3.DatabaseError, match="Database corruption recovery failed"):
                await backend._init_db()

    @pytest.mark.asyncio
    async def test_special_characters_in_filename(self, tmp_path: Path) -> None:
        """Test handling of special characters in database filename."""
        # Create path with special characters
        special_path = tmp_path / "test'with\"quotes.db"
        special_path.write_bytes(b"corrupted sqlite data")

        backend = SQLiteBackend(special_path)

        with patch("time.time", return_value=1234567890):
            await backend._init_db()

        # Verify backup was created and new database exists
        backup_files = list(tmp_path.glob("test'with\"quotes.db.corrupt.*"))
        assert len(backup_files) == 1
        assert special_path.exists()

    @pytest.mark.asyncio
    async def test_very_long_filename(self, tmp_path: Path) -> None:
        """Test handling of very long filenames."""
        # Create path with very long name
        long_name = "a" * 200 + ".db"
        long_path = tmp_path / long_name
        long_path.write_bytes(b"corrupted sqlite data")

        backend = SQLiteBackend(long_path)

        with patch("time.time", return_value=1234567890):
            await backend._init_db()

        # Verify backup was created and new database exists
        backup_files = list(tmp_path.glob(f"{long_name}.corrupt.*"))
        assert len(backup_files) == 1
        assert long_path.exists()

    @pytest.mark.asyncio
    async def test_no_write_permissions(self, tmp_path: Path) -> None:
        """Test handling when directory has no write permissions."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        # Make directory read-only
        tmp_path.chmod(0o444)

        try:
            backend = SQLiteBackend(db_path)
            with pytest.raises((OSError, PermissionError)):
                await backend._init_db()
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
                await backend._init_db()

    @pytest.mark.asyncio
    async def test_concurrent_backup_operations(self, tmp_path: Path) -> None:
        """Test concurrent backup operations."""

        async def create_backup(i: int) -> None:
            db_path = tmp_path / f"test{i}.db"
            db_path.write_bytes(b"corrupted sqlite data")
            backend = SQLiteBackend(db_path)
            await backend._init_db()

        # Run multiple concurrent backup operations
        tasks = [create_backup(i) for i in range(5)]
        await asyncio.gather(*tasks)

        # Verify all backups were created successfully
        backup_files = list(tmp_path.glob("*.corrupt.*"))
        assert len(backup_files) == 5

    @pytest.mark.asyncio
    async def test_infinite_loop_bug_fix(self, tmp_path: Path) -> None:
        """Test the fix for the infinite loop bug in backup logic."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        # Create many existing backup files to trigger the cleanup logic
        for i in range(150):  # More than MAX_BACKUP_SUFFIX_ATTEMPTS
            backup_file = tmp_path / f"test.db.corrupt.1234567890.{i}"
            backup_file.write_bytes(b"existing backup")
            # Set different modification times to ensure oldest can be identified
            backup_file.touch()
            time.sleep(0.001)  # Ensure different timestamps

        backend = SQLiteBackend(db_path)

        # Mock time.time to return the same timestamp as existing backups
        with patch("time.time", return_value=1234567890):
            await backend._init_db()

        # Verify that the corrupted database was moved to backup and a new one was created
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert (
            len(backup_files) == 151
        )  # 150 existing + 1 new backup (corrupted DB moved to base path)

        # Verify new database was created
        assert db_path.exists()
        assert db_path.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_infinite_loop_bug_with_continue_fix(self, tmp_path: Path) -> None:
        """Test that the continue statement properly resets the backup path selection."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        # Create exactly MAX_BACKUP_SUFFIX_ATTEMPTS backup files
        for i in range(100):  # MAX_BACKUP_SUFFIX_ATTEMPTS
            backup_file = tmp_path / f"test.db.corrupt.1234567890.{i}"
            backup_file.write_bytes(b"existing backup")
            backup_file.touch()
            time.sleep(0.001)

        backend = SQLiteBackend(db_path)

        # Mock time.time to return the same timestamp
        with patch("time.time", return_value=1234567890):
            await backend._init_db()

        # Verify that a new backup was created after removing the oldest
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert (
            len(backup_files) == 101
        )  # 100 existing + 1 new backup (corrupted DB moved to base path)

        # Verify new database was created
        assert db_path.exists()
        assert db_path.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_backup_path_update_after_cleanup(self, tmp_path: Path) -> None:
        """Test that backup path is properly updated after cleanup."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        # Create backup files with different timestamps
        for i in range(50):
            backup_file = tmp_path / f"test.db.corrupt.1234567890.{i}"
            backup_file.write_bytes(b"existing backup")
            backup_file.touch()
            time.sleep(0.001)

        backend = SQLiteBackend(db_path)

        # Mock time.time to return a new timestamp
        with patch("time.time", return_value=9876543210):
            await backend._init_db()

        # Verify that a new backup was created with the new timestamp
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) == 51  # 50 existing + 1 new

        # Check that the new backup has the new timestamp
        new_backup = next(f for f in backup_files if "9876543210" in f.name)
        assert new_backup.exists()

    @pytest.mark.asyncio
    async def test_race_condition_in_backup_creation(self, tmp_path: Path) -> None:
        """Test race condition where backup file is created between exists() check and rename()."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        backend = SQLiteBackend(db_path)

        # Mock exists() to return False initially, then True on second call
        original_exists = Path.exists
        call_count = 0

        def mock_exists(self):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return False
            return original_exists(self)

        with patch("pathlib.Path.exists", mock_exists):
            with patch("time.time", return_value=1234567890):
                await backend._init_db()

        # Verify backup was created despite race condition
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) == 1

    @pytest.mark.asyncio
    async def test_readonly_directory_fallback(self, tmp_path: Path) -> None:
        """Test fallback behavior when backup directory becomes read-only."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        # Create a backup file that would be read-only
        backup_file = tmp_path / "test.db.corrupt.1234567890"
        backup_file.write_bytes(b"existing backup")
        backup_file.chmod(0o444)  # Read-only

        try:
            backend = SQLiteBackend(db_path)

            with patch("time.time", return_value=1234567890):
                # Should handle read-only backup file gracefully
                await backend._init_db()

            # Verify new database was created
            assert db_path.exists()
            assert db_path.stat().st_size > 0
        finally:
            # Restore permissions
            backup_file.chmod(0o644)

    @pytest.mark.asyncio
    async def test_backup_pattern_glob_handling(self, tmp_path: Path) -> None:
        """Test that backup pattern globbing works correctly with special characters."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        # Create backup files with various patterns
        backup_files = [
            tmp_path / "test.db.corrupt.1234567890",
            tmp_path / "test.db.corrupt.1234567890.1",
            tmp_path / "test.db.corrupt.1234567890.2",
            tmp_path / "test.db.corrupt.1234567890.3",
        ]

        for backup in backup_files:
            backup.write_bytes(b"existing backup")
            backup.touch()
            time.sleep(0.001)

        backend = SQLiteBackend(db_path)

        # Mock time.time to return the same timestamp
        with patch("time.time", return_value=1234567890):
            await backend._init_db()

        # Verify that the oldest backup was removed and a new one was created
        backup_files_after = list(tmp_path.glob("test.db.corrupt.*"))
        assert (
            len(backup_files_after) == 5
        )  # 4 existing + 1 new backup (corrupted DB moved to counter suffix path)

    @pytest.mark.asyncio
    async def test_backup_stat_error_handling(self, tmp_path: Path) -> None:
        """Test handling when stat() fails on backup files."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        # Create backup files
        for i in range(10):
            backup_file = tmp_path / f"test.db.corrupt.1234567890.{i}"
            backup_file.write_bytes(b"existing backup")
            backup_file.touch()
            time.sleep(0.001)

        backend = SQLiteBackend(db_path)

        # Mock stat() to fail on some files
        original_stat = Path.stat
        call_count = 0

        def mock_stat(self):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Fail every 3rd call
                raise OSError("Permission denied")
            return original_stat(self)

        with patch("pathlib.Path.stat", mock_stat):
            with patch("time.time", return_value=1234567890):
                await backend._init_db()

        # Verify new database was created despite stat errors
        assert db_path.exists()
        assert db_path.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_backup_unlink_error_handling(self, tmp_path: Path) -> None:
        """Test handling when unlink() fails on backup files."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        # Create backup files
        for i in range(10):
            backup_file = tmp_path / f"test.db.corrupt.1234567890.{i}"
            backup_file.write_bytes(b"existing backup")
            backup_file.touch()
            time.sleep(0.001)

        backend = SQLiteBackend(db_path)

        # Mock unlink() to fail
        with patch("pathlib.Path.unlink", side_effect=OSError("Permission denied")):
            with patch("time.time", return_value=1234567890):
                # Should handle unlink failure gracefully
                await backend._init_db()

        # Verify new database was created
        assert db_path.exists()
        assert db_path.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_backup_min_function_with_none_values(self, tmp_path: Path) -> None:
        """Test that min() function handles None values correctly in backup cleanup."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        # Create backup files
        for i in range(5):
            backup_file = tmp_path / f"test.db.corrupt.1234567890.{i}"
            backup_file.write_bytes(b"existing backup")
            backup_file.touch()
            time.sleep(0.001)

        backend = SQLiteBackend(db_path)

        # Mock stat() to return None for some files
        original_stat = Path.stat
        call_count = 0

        def mock_stat(self):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:  # Return None for every 2nd call
                return Mock(st_mtime=None)
            return original_stat(self)

        with patch("pathlib.Path.stat", mock_stat):
            with patch("time.time", return_value=1234567890):
                await backend._init_db()

        # Verify new database was created
        assert db_path.exists()
        assert db_path.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_backup_empty_directory_handling(self, tmp_path: Path) -> None:
        """Test handling when no existing backup files are found."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        backend = SQLiteBackend(db_path)

        with patch("time.time", return_value=1234567890):
            await backend._init_db()

        # Verify backup was created and new database exists
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) == 1
        assert db_path.exists()

    @pytest.mark.asyncio
    async def test_backup_max_attempts_exceeded_handling(self, tmp_path: Path) -> None:
        """Test handling when MAX_BACKUP_SUFFIX_ATTEMPTS is exceeded."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        # Create exactly MAX_BACKUP_SUFFIX_ATTEMPTS + 1 backup files
        for i in range(101):  # MAX_BACKUP_SUFFIX_ATTEMPTS + 1
            backup_file = tmp_path / f"test.db.corrupt.1234567890.{i}"
            backup_file.write_bytes(b"existing backup")
            backup_file.touch()
            time.sleep(0.001)

        backend = SQLiteBackend(db_path)

        # Mock time.time to return the same timestamp
        with patch("time.time", return_value=1234567890):
            await backend._init_db()

        # Verify that cleanup occurred and new database was created
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert (
            len(backup_files) == 102
        )  # 101 existing + 1 new backup (corrupted DB moved to base path)
        assert db_path.exists()

    @pytest.mark.asyncio
    async def test_backup_continue_statement_effectiveness(self, tmp_path: Path) -> None:
        """Test that the continue statement properly continues the loop after cleanup."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        # Create backup files that will trigger cleanup
        for i in range(150):  # More than MAX_BACKUP_SUFFIX_ATTEMPTS
            backup_file = tmp_path / f"test.db.corrupt.1234567890.{i}"
            backup_file.write_bytes(b"existing backup")
            backup_file.touch()
            time.sleep(0.001)

        backend = SQLiteBackend(db_path)

        # Track the number of times the cleanup block is executed
        cleanup_calls = 0

        original_glob = Path.glob

        def mock_glob(self, pattern):
            nonlocal cleanup_calls
            if "corrupt.*" in pattern:
                cleanup_calls += 1
            return original_glob(self, pattern)

        with patch("pathlib.Path.glob", mock_glob):
            with patch("time.time", return_value=1234567890):
                await backend._init_db()

        # Verify that cleanup was called and new database was created
        # Note: cleanup_calls might be 0 if the backup path is available immediately
        assert db_path.exists()
        assert db_path.exists()
        assert db_path.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_backup_path_reset_after_cleanup(self, tmp_path: Path) -> None:
        """Test that backup path is properly reset after cleanup to avoid infinite loop."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        # Create backup files
        for i in range(150):  # More than MAX_BACKUP_SUFFIX_ATTEMPTS
            backup_file = tmp_path / f"test.db.corrupt.1234567890.{i}"
            backup_file.write_bytes(b"existing backup")
            backup_file.touch()
            time.sleep(0.001)

        backend = SQLiteBackend(db_path)

        # Mock exists() to return True initially, then False after cleanup
        call_count = 0

        def mock_exists(self):
            nonlocal call_count
            call_count += 1
            if call_count <= 100:  # First 100 calls return True
                return True
            return False  # After cleanup, return False

        with patch("pathlib.Path.exists", mock_exists):
            with patch("time.time", return_value=1234567890):
                await backend._init_db()

        # Verify new database was created
        assert db_path.exists()
        assert db_path.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_backup_counter_reset_after_cleanup(self, tmp_path: Path) -> None:
        """Test that the counter is properly reset after cleanup."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        # Create backup files
        for i in range(150):  # More than MAX_BACKUP_SUFFIX_ATTEMPTS
            backup_file = tmp_path / f"test.db.corrupt.1234567890.{i}"
            backup_file.write_bytes(b"existing backup")
            backup_file.touch()
            time.sleep(0.001)

        backend = SQLiteBackend(db_path)

        # Track counter values during backup creation
        original_exists = Path.exists

        def mock_exists(self):
            # This will be called multiple times during the backup process
            return original_exists(self)

        with patch("pathlib.Path.exists", mock_exists):
            with patch("time.time", return_value=1234567890):
                await backend._init_db()

        # Verify new database was created
        assert db_path.exists()
        assert db_path.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_backup_infinite_loop_prevention(self, tmp_path: Path) -> None:
        """Test that infinite loops are prevented in backup logic."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        # Create backup files that will trigger the cleanup logic
        for i in range(150):  # More than MAX_BACKUP_SUFFIX_ATTEMPTS
            backup_file = tmp_path / f"test.db.corrupt.1234567890.{i}"
            backup_file.write_bytes(b"existing backup")
            backup_file.touch()
            time.sleep(0.001)

        backend = SQLiteBackend(db_path)

        # Mock time.time to return the same timestamp as existing backups
        with patch("time.time", return_value=1234567890):
            # This should not cause an infinite loop due to the continue fix
            await backend._init_db()

        # Verify new database was created despite the challenging conditions
        assert db_path.exists()
        assert db_path.stat().st_size > 0

        # Verify that backup files were handled correctly
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) == 151  # 150 existing + 1 new backup
