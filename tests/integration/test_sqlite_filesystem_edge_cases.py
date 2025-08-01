"""Integration tests for SQLiteBackend filesystem edge cases."""

import os
import sqlite3
from pathlib import Path
from unittest.mock import patch

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
    async def test_readonly_directory_fallback(self, tmp_path: Path) -> None:
        """Test fallback behavior in readonly directory."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        backend = SQLiteBackend(db_path)

        # Mock rename to fail due to readonly directory
        with patch.object(Path, "rename", side_effect=OSError("[Errno 30] Read-only file system")):
            await backend._init_db()

        # Verify corrupted file was removed and new database created
        assert db_path.exists()
        assert db_path.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_race_condition_in_backup_creation(self, tmp_path: Path) -> None:
        """Test race condition handling in backup creation."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"corrupted sqlite data")

        backend = SQLiteBackend(db_path)

        # Mock exists to simulate race condition - return False initially, then True
        original_exists = Path.exists
        call_count = 0

        def mock_exists(self):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return False
            return original_exists(self)

        with patch.object(Path, "exists", mock_exists):
            with patch("time.time", return_value=1234567890):
                await backend._init_db()

        # Verify backup was created and new database exists
        backup_files = list(tmp_path.glob("test.db.corrupt.*"))
        assert len(backup_files) >= 1  # At least one backup should exist
        assert db_path.exists()
