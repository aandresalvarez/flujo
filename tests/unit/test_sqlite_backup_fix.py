"""Tests for SQLiteBackend backup fix to handle platform-specific issues."""

import os
import sqlite3
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from flujo.state.backends.sqlite import SQLiteBackend

if getattr(os, "geteuid", lambda: -1)() == 0:
    pytest.skip(
        "permission-based SQLite tests skipped when running as root",
        allow_module_level=True,
    )

# Mark all tests in this module for serial execution to prevent SQLite concurrency issues
pytestmark = pytest.mark.serial


class TestSQLiteBackupFix:
    """Test the backup fix for platform-specific issues."""

    @pytest.mark.asyncio
    async def test_backup_creates_unique_filenames(self, tmp_path: Path) -> None:
        """Test that backup creates unique filenames with timestamps."""
        db_path = tmp_path / "test.db"
        backend = SQLiteBackend(db_path)

        # Create a corrupted database file
        db_path.write_text("corrupted database content")

        # Mock the database connection to raise corruption error
        with patch("aiosqlite.connect", side_effect=sqlite3.DatabaseError("Corrupted database")):
            try:
                await backend._init_db()
            except sqlite3.DatabaseError:
                pass  # Expected to fail

        # Check that backup file was created with timestamp
        backup_files = list(tmp_path.glob("*.db.corrupt.*"))
        assert len(backup_files) == 1
        backup_file = backup_files[0]
        assert backup_file.name.startswith("test.db.corrupt.")
        # Use regex to match timestamp pattern instead of exact time
        import re

        assert re.match(r"test\.db\.corrupt\.\d+$", backup_file.name), (
            f"Unexpected backup filename: {backup_file.name}"
        )

    @pytest.mark.asyncio
    async def test_backup_handles_existing_files(self, tmp_path: Path) -> None:
        """Test that backup handles existing backup files gracefully."""
        db_path = tmp_path / "test.db"
        backend = SQLiteBackend(db_path)

        # Create existing backup files
        timestamp = int(time.time())
        existing_backup1 = tmp_path / f"test.db.corrupt.{timestamp}"
        existing_backup2 = tmp_path / f"test.db.corrupt.{timestamp}.1"
        existing_backup1.write_text("existing backup 1")
        existing_backup2.write_text("existing backup 2")

        # Create a corrupted database file
        db_path.write_text("corrupted database content")

        # Mock the database connection to raise corruption error
        with patch("aiosqlite.connect", side_effect=sqlite3.DatabaseError("Corrupted database")):
            try:
                await backend._init_db()
            except sqlite3.DatabaseError:
                pass  # Expected to fail

        # Check that new backup file was created with counter
        backup_files = list(tmp_path.glob("*.db.corrupt.*"))
        assert len(backup_files) == 3  # 2 existing + 1 new

        # The new backup should have a counter suffix
        new_backups = [f for f in backup_files if f.name.endswith(".2")]
        assert len(new_backups) == 1

    @pytest.mark.asyncio
    async def test_backup_handles_rename_failure(self, tmp_path: Path) -> None:
        """Test that backup handles rename failures gracefully."""
        db_path = tmp_path / "test.db"
        backend = SQLiteBackend(db_path)

        # Create a corrupted database file
        db_path.write_text("corrupted database content")

        # Mock the rename to fail (simulating Windows FileExistsError)
        with patch("pathlib.Path.rename", side_effect=FileExistsError("File exists")):
            with patch(
                "aiosqlite.connect", side_effect=sqlite3.DatabaseError("Corrupted database")
            ):
                try:
                    await backend._init_db()
                except sqlite3.DatabaseError:
                    pass  # Expected to fail

        # Check that the corrupted file was removed as fallback
        assert not db_path.exists()

    @pytest.mark.asyncio
    async def test_backup_prevents_infinite_loop(self, tmp_path: Path) -> None:
        """Test that backup prevents infinite loop with too many existing files."""
        db_path = tmp_path / "test.db"
        backend = SQLiteBackend(db_path)

        # Create many existing backup files
        timestamp = int(time.time())
        for i in range(105):  # More than the 100 limit
            backup_file = tmp_path / f"test.db.corrupt.{timestamp}.{i}"
            backup_file.write_text(f"existing backup {i}")

        # Create a corrupted database file
        db_path.write_text("corrupted database content")

        # Mock the database connection to raise corruption error
        with patch("aiosqlite.connect", side_effect=sqlite3.DatabaseError("Corrupted database")):
            try:
                await backend._init_db()
            except sqlite3.DatabaseError:
                pass  # Expected to fail

        # Check that the corrupted file was removed
        assert not db_path.exists()

    @pytest.mark.asyncio
    async def test_backup_preserves_debugging_information(self, tmp_path: Path) -> None:
        """Test that backup preserves debugging information across platforms."""
        db_path = tmp_path / "test.db"
        backend = SQLiteBackend(db_path)

        # Create multiple corrupted databases to test preservation
        for i in range(3):
            # Create a corrupted database file
            db_path.write_text(f"corrupted database content {i}")

            # Mock the database connection to raise corruption error
            with patch(
                "aiosqlite.connect", side_effect=sqlite3.DatabaseError(f"Corrupted database {i}")
            ):
                try:
                    await backend._init_db()
                except sqlite3.DatabaseError:
                    pass  # Expected to fail

        # Check that all backup files are preserved
        backup_files = list(tmp_path.glob("*.db.corrupt.*"))
        assert len(backup_files) == 3

        # Verify each backup has unique content
        backup_contents = [f.read_text() for f in backup_files]
        assert "corrupted database content 0" in backup_contents
        assert "corrupted database content 1" in backup_contents
        assert "corrupted database content 2" in backup_contents

    @pytest.mark.asyncio
    async def test_backup_handles_many_existing_files_correctly(self, tmp_path: Path) -> None:
        """Test that backup logic correctly handles many existing backup files."""
        db_path = tmp_path / "test.db"
        backend = SQLiteBackend(db_path)

        # Create many existing backup files to trigger the cleanup logic
        timestamp = int(time.time())
        for i in range(105):  # More than MAX_BACKUP_SUFFIX_ATTEMPTS
            backup_file = tmp_path / f"test.db.corrupt.{timestamp}.{i}"
            backup_file.write_text(f"backup content {i}")
            # Set different modification times to ensure oldest detection works
            os.utime(backup_file, (timestamp - i, timestamp - i))

        # Create a corrupted database file
        db_path.write_text("corrupted database content")

        # Mock the database connection to raise corruption error
        with patch("aiosqlite.connect", side_effect=sqlite3.DatabaseError("Corrupted database")):
            try:
                await backend._init_db()
            except sqlite3.DatabaseError:
                pass  # Expected to fail

        # Verify that the corrupted database was successfully backed up
        # The oldest backup should have been removed and the corrupted DB moved to that location
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) >= 1, "Should have at least one backup file"

        # Check that the corrupted database file no longer exists
        assert not db_path.exists(), "Corrupted database should have been moved"

        # Verify that one of the backup files contains the corrupted content
        backup_with_corrupted_content = None
        for backup_file in backup_files:
            if backup_file.read_text() == "corrupted database content":
                backup_with_corrupted_content = backup_file
                break

        assert backup_with_corrupted_content is not None, (
            "Should find backup with corrupted content"
        )

    @pytest.mark.asyncio
    async def test_backup_regression_removes_oldest_and_moves_db(self, tmp_path: Path) -> None:
        """Regression: When too many backups exist, oldest is removed and corrupted DB is moved to that path."""
        db_path = tmp_path / "test.db"
        backend = SQLiteBackend(db_path)
        timestamp = int(time.time())
        max_attempts = 100
        # Fill up with max_attempts backup files
        for i in range(1, max_attempts + 1):
            backup_file = tmp_path / f"test.db.corrupt.{timestamp}.{i}"
            backup_file.write_text(f"backup content {i}")
            os.utime(backup_file, (timestamp - i, timestamp - i))
        # The oldest is .1
        db_path.write_text("corrupted database content")
        with patch("aiosqlite.connect", side_effect=sqlite3.DatabaseError("Corrupted database")):
            try:
                await backend._init_db()
            except sqlite3.DatabaseError:
                pass
        # The corrupted DB should be moved to one of the backup files
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) >= 1, "Should have at least one backup file"

        # Find the backup file that contains the corrupted content
        backup_with_corrupted_content = None
        for backup_file in backup_files:
            if backup_file.read_text() == "corrupted database content":
                backup_with_corrupted_content = backup_file
                break

        assert backup_with_corrupted_content is not None, (
            "Should find backup with corrupted content"
        )
        # The original DB should be gone
        assert not db_path.exists()

    @pytest.mark.asyncio
    async def test_backup_path_is_unique_after_cleanup(self, tmp_path: Path) -> None:
        """Backup path is always unique and available after cleanup."""
        db_path = tmp_path / "test.db"
        backend = SQLiteBackend(db_path)
        timestamp = int(time.time())
        # Fill up with backups, but leave .1 missing
        for i in range(2, 102):
            backup_file = tmp_path / f"test.db.corrupt.{timestamp}.{i}"
            backup_file.write_text(f"backup content {i}")
        db_path.write_text("corrupted database content")
        with patch("aiosqlite.connect", side_effect=sqlite3.DatabaseError("Corrupted database")):
            try:
                await backend._init_db()
            except sqlite3.DatabaseError:
                pass
        # The base backup file should now exist and contain the corrupted DB
        base_backup = tmp_path / f"test.db.corrupt.{timestamp}"
        assert base_backup.read_text() == "corrupted database content"

    @pytest.mark.asyncio
    async def test_backup_race_condition_file_deleted_between_check_and_rename(
        self, tmp_path: Path
    ) -> None:
        """Backup logic works if a file is deleted between check and rename (race condition)."""
        db_path = tmp_path / "test.db"
        backend = SQLiteBackend(db_path)
        timestamp = int(time.time())
        backup_file = tmp_path / f"test.db.corrupt.{timestamp}.1"
        backup_file.write_text("old backup")
        db_path.write_text("corrupted database content")
        # Patch Path.rename to delete the backup file just before rename
        orig_rename = Path.rename

        def race_rename(self, target):
            if target.exists():
                target.unlink()
            return orig_rename(self, target)

        with (
            patch("aiosqlite.connect", side_effect=sqlite3.DatabaseError("Corrupted database")),
            patch.object(Path, "rename", race_rename),
        ):
            try:
                await backend._init_db()
            except sqlite3.DatabaseError:
                pass
        # The base backup file should now contain the corrupted DB
        base_backup = tmp_path / f"test.db.corrupt.{timestamp}"
        assert base_backup.read_text() == "corrupted database content"

    @pytest.mark.asyncio
    async def test_backup_read_only_directory_fallback(self, tmp_path: Path) -> None:
        """Backup logic falls back to deletion if directory is read-only."""
        db_path = tmp_path / "test.db"
        backend = SQLiteBackend(db_path)
        db_path.write_text("corrupted database content")
        # Patch Path.rename to always raise PermissionError
        with (
            patch("aiosqlite.connect", side_effect=sqlite3.DatabaseError("Corrupted database")),
            patch.object(Path, "rename", side_effect=PermissionError("Read-only file system")),
        ):
            try:
                await backend._init_db()
            except sqlite3.DatabaseError:
                pass
        # The corrupted DB should be deleted
        assert not db_path.exists()
        # No exception should be raised beyond DatabaseError
