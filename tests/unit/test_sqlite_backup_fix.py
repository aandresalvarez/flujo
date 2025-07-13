"""Tests for SQLiteBackend backup fix to handle platform-specific issues."""

import sqlite3
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from flujo.state.backends.sqlite import SQLiteBackend


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
        assert backup_file.name.endswith(str(int(time.time())))

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
