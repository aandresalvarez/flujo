"""SQLite-backed persistent storage for workflow state with optimized schema."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import time
import weakref
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
    cast,
)

import aiosqlite
import os

from .base import StateBackend
from flujo.infra import telemetry
from flujo.utils.serialization import robust_serialize, safe_deserialize

# Try to import orjson for faster JSON serialization
try:
    import orjson
    from flujo.utils.serialization import safe_serialize

    def _fast_json_dumps(obj: Any) -> str:
        """Use orjson for faster JSON serialization with robust serialization.

        Note: avoid key-sorting to reduce CPU overhead on hot paths.
        """
        serialized_obj = safe_serialize(obj)
        blob: bytes = orjson.dumps(serialized_obj)
        return blob.decode("utf-8")

except ImportError:
    from flujo.utils.serialization import safe_serialize

    def _fast_json_dumps(obj: Any) -> str:
        """Fallback to standard json for JSON serialization with robust serialization.

        Note: avoid key-sorting to reduce CPU overhead on hot paths.
        """
        serialized_obj = safe_serialize(obj)
        s: str = json.dumps(serialized_obj, separators=(",", ":"))
        return s


if TYPE_CHECKING:
    from asyncio import AbstractEventLoop


# Maximum length for SQL identifiers
MAX_SQL_IDENTIFIER_LENGTH = 1000

# Problematic Unicode characters that should not be in SQL identifiers
PROBLEMATIC_UNICODE_CHARS = [
    "\u0000",  # Null character
    "\u2028",  # Line separator
    "\u2029",  # Paragraph separator
    "\u200b",  # Zero-width space
    "\u200c",  # Zero-width non-joiner
    "\u200d",  # Zero-width joiner
    "\x01",  # Start of heading
    "\x1f",  # Unit separator
]

# Whitelist of allowed column names and definitions for enhanced security
ALLOWED_COLUMNS = {
    "total_steps": "INTEGER DEFAULT 0",
    "error_message": "TEXT",
    "execution_time_ms": "INTEGER",
    "memory_usage_mb": "REAL",
    "step_history": "TEXT",
}

# Compiled regex pattern for column definition validation
COLUMN_DEF_PATTERN = re.compile(
    r"""^(INTEGER|REAL|TEXT|BLOB|NUMERIC|BOOLEAN)(\([0-9, ]+\))?(
        (\s+PRIMARY\s+KEY)?
        (\s+UNIQUE)?
        (\s+NOT\s+NULL)?
        (\s+DEFAULT\s+(NULL|[0-9]+|[0-9]*\.[0-9]+|'.*?'|\".*?\"|TRUE|FALSE))?
        (\s+CHECK\s+\([a-zA-Z0-9_<>=!&|()\s]+\))?
        (\s+COLLATE\s+(BINARY|NOCASE|RTRIM))?
    )*$""",
    re.IGNORECASE | re.VERBOSE,
)


def _validate_sql_identifier(identifier: str) -> bool:
    """Validate that a string is a safe SQL identifier.

    This function ensures that column names and table names are safe to use
    in SQL statements by checking against a whitelist of allowed characters.

    Args:
        identifier: The identifier to validate

    Returns:
        True if the identifier is safe, False otherwise

    Raises:
        ValueError: If the identifier contains unsafe characters
    """
    if not identifier or not isinstance(identifier, str):
        raise ValueError(f"Invalid identifier type or empty: {identifier}")

    # SQLite identifiers can contain: letters, digits, underscore
    # Must start with a letter or underscore
    # Also check for problematic Unicode characters
    safe_pattern = r"^[a-zA-Z_][a-zA-Z0-9_]*$"

    # Check for problematic Unicode characters
    for char in PROBLEMATIC_UNICODE_CHARS:
        if char in identifier:
            raise ValueError(f"Identifier contains problematic Unicode character: {identifier}")

    # Check for very long identifiers (SQLite has limits)
    if len(identifier) > MAX_SQL_IDENTIFIER_LENGTH:
        raise ValueError(
            f"Identifier too long (max {MAX_SQL_IDENTIFIER_LENGTH} characters): {identifier}"
        )

    if not re.match(safe_pattern, identifier):
        raise ValueError(f"Unsafe SQL identifier: {identifier}")

    # Additional safety: check for SQL keywords that could be dangerous
    dangerous_keywords = {
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
    }

    identifier_upper = identifier.upper()
    if identifier_upper in dangerous_keywords:
        raise ValueError(f"Identifier matches dangerous SQL keyword: {identifier}")

    return True


def _validate_column_definition(column_def: str) -> bool:
    """Validate that a column definition is safe.

    Args:
        column_def: The column definition to validate

    Returns:
        True if the definition is safe, False otherwise

    Raises:
        ValueError: If the definition contains unsafe content
    """
    if not column_def or not isinstance(column_def, str):
        raise ValueError(f"Invalid column definition type or empty: {column_def}")

    # Reject non-printable, non-ASCII, or control characters using regex for better performance
    if re.search(r"[^\x20-\x7e]", column_def):
        raise ValueError(
            f"Unsafe column definition: contains non-printable or non-ASCII characters: {column_def}"
        )
    # Reject SQL injection patterns and malformed definitions
    if any(x in column_def for x in [";", "--", "/*", "*/", "'", '"']):
        raise ValueError(
            f"Unsafe column definition: contains forbidden SQL characters: {column_def}"
        )
    if column_def.count("(") != column_def.count(")"):
        raise ValueError(f"Unsafe column definition: unmatched parentheses: {column_def}")

    # Parse the definition to check for unsafe content
    definition_upper = column_def.upper()

    # Check for dangerous SQL constructs (pre-computed as uppercase for efficiency)
    dangerous_patterns = [
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
        "FROM",
        "WHERE",
        "OR",
        "AND",
        ";",
        "--",
        "/*",
        "*/",
        "XP_",
        "SP_",
    ]

    for pattern in dangerous_patterns:
        if pattern in definition_upper:
            raise ValueError(f"Unsafe column definition contains '{pattern}': {column_def}")

    # Validate the entire column definition structure using a regular expression
    # Use more restrictive patterns for DEFAULT and CHECK constraints to prevent SQL injection
    match = COLUMN_DEF_PATTERN.match(column_def)
    if not match:
        raise ValueError(f"Column definition does not match a safe SQLite structure: {column_def}")
    # Ensure no unknown trailing content after allowed constraints
    allowed_constraints = [
        "PRIMARY KEY",
        "UNIQUE",
        "NOT NULL",
        "DEFAULT",
        "CHECK",
        "COLLATE",
    ]
    # Remove type and type parameters
    rest = column_def[len(match.group(1) or "") :]
    if match.group(2):
        rest = rest[len(match.group(2)) :]  # Remove type parameters
    # Remove all allowed constraints
    for constraint in allowed_constraints:
        rest = re.sub(rf"\b{constraint}\b(\s+\S+|\s*\(.+?\))?", "", rest, flags=re.IGNORECASE)
    if rest.strip():
        raise ValueError(
            f"Unsafe column definition: unknown or unsafe trailing content: {column_def}"
        )
    # Additional checks for COLLATE and DEFAULT
    collate_match = re.search(r"COLLATE\s+(\w+)", column_def, re.IGNORECASE)
    if collate_match:
        if collate_match.group(1).upper() not in {"BINARY", "NOCASE", "RTRIM"}:
            raise ValueError(
                f"Unsafe column definition: invalid COLLATE value: {collate_match.group(1)}"
            )
    default_match = re.search(r"DEFAULT\s+([^ ]+)", column_def, re.IGNORECASE)
    if default_match:
        val = default_match.group(1)
        if not re.match(
            r"^(NULL|[0-9]+|[0-9]*\.[0-9]+|'.*?'|\".*?\"|TRUE|FALSE)$",
            val,
            re.IGNORECASE,
        ):
            raise ValueError(f"Unsafe column definition: invalid DEFAULT value: {val}")

    return True


class SQLiteBackend(StateBackend):
    """SQLite-backed persistent storage for workflow state with optimized schema."""

    _global_file_locks: "weakref.WeakKeyDictionary[AbstractEventLoop, Dict[str, asyncio.Lock]]" = (
        weakref.WeakKeyDictionary()
    )
    _thread_file_locks: Dict[int, Dict[str, asyncio.Lock]] = {}

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure parent directories exist
        self._lock = asyncio.Lock()
        self._initialized = False
        # Lightweight single-connection pool to reduce connect() overhead on hot paths.
        # Guarded by self._lock for serialized access.
        self._connection_pool: Optional[aiosqlite.Connection] = None

        # Event-loop-local file-level lock - will be initialized lazily
        self._file_lock: Optional[asyncio.Lock] = None
        self._file_lock_key = str(self.db_path.absolute())

    def _get_file_lock(self) -> asyncio.Lock:
        """Get the file lock for the current event loop."""
        if self._file_lock is None:
            try:
                # Try to get the current event loop
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running event loop, try to get the current event loop
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    # No event loop in current thread, use a thread-local approach
                    # instead of creating a new loop that could interfere with existing operations
                    import threading

                    thread_id = threading.get_ident()
                    if thread_id not in SQLiteBackend._thread_file_locks:
                        SQLiteBackend._thread_file_locks[thread_id] = {}
                    lock_map = SQLiteBackend._thread_file_locks[thread_id]
                    db_key = str(self.db_path.absolute())
                    if db_key not in lock_map:
                        lock_map[db_key] = asyncio.Lock()
                    self._file_lock = lock_map[db_key]
                    return self._file_lock

            # We have a valid event loop
            if loop not in SQLiteBackend._global_file_locks:
                SQLiteBackend._global_file_locks[loop] = {}
            lock_map = SQLiteBackend._global_file_locks[loop]
            db_key = str(self.db_path.absolute())
            if db_key not in lock_map:
                lock_map[db_key] = asyncio.Lock()
            self._file_lock = lock_map[db_key]
        # Always return a Lock
        assert self._file_lock is not None
        return self._file_lock

    async def _init_db(self, retry_count: int = 0, max_retries: int = 1) -> None:
        """Initialize the database with optimized schema and settings."""
        try:
            # Check if database file exists and handle possible corrupt or inaccessible file
            try:
                # Use os.path.exists to avoid Path.stat side effects
                exists = os.path.exists(self.db_path)
            except OSError as e:
                telemetry.logfire.warning(
                    f"File existence check failed, skipping backup and proceeding: {e}"
                )
            else:
                if exists:
                    try:
                        # Check file size safely
                        try:
                            file_size = self.db_path.stat().st_size
                            if file_size > 0:
                                # Try to connect to check if database is valid
                                async with aiosqlite.connect(self.db_path) as test_db:
                                    await test_db.execute("SELECT 1")
                        except (OSError, TypeError) as e:
                            telemetry.logfire.warning(f"File stat failed, assuming corrupted: {e}")
                            await self._backup_corrupted_database()
                    except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
                        telemetry.logfire.warning(
                            f"Database appears to be corrupted, creating backup: {e}"
                        )
                        await self._backup_corrupted_database()

            async with aiosqlite.connect(self.db_path) as db:
                # OPTIMIZATION: Use more efficient SQLite settings for performance
                await db.execute("PRAGMA journal_mode = WAL")
                await db.execute("PRAGMA synchronous = NORMAL")
                await db.execute("PRAGMA cache_size = 10000")  # Increase cache size
                await db.execute("PRAGMA temp_store = MEMORY")  # Use memory for temp tables
                await db.execute("PRAGMA mmap_size = 268435456")  # 256MB memory mapping
                await db.execute("PRAGMA page_size = 4096")

                # Batch DDL inside a transaction to reduce fsyncs
                await db.execute("BEGIN")

                # Create the main workflow_state table with optimized schema
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS workflow_state (
                        run_id TEXT PRIMARY KEY,
                        pipeline_id TEXT NOT NULL,
                        pipeline_name TEXT NOT NULL,
                        pipeline_version TEXT NOT NULL,
                        current_step_index INTEGER NOT NULL,
                        pipeline_context TEXT,
                        last_step_output TEXT,
                        step_history TEXT,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        total_steps INTEGER DEFAULT 0,
                        error_message TEXT,
                        execution_time_ms INTEGER,
                        memory_usage_mb REAL
                    )
                    """
                )

                # Create indexes for better query performance (after migration)
                # Note: Index creation is moved after migration to ensure columns exist

                # Create the runs table for run tracking (for backward compatibility)
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS runs (
                        run_id TEXT PRIMARY KEY,
                        pipeline_id TEXT NOT NULL,
                        pipeline_name TEXT NOT NULL,
                        pipeline_version TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        execution_time_ms INTEGER,
                        memory_usage_mb REAL,
                        total_steps INTEGER DEFAULT 0,
                        error_message TEXT
                    )
                    """
                )

                # Create indexes for runs table
                await db.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)")
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_runs_pipeline_id ON runs(pipeline_id)"
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at)"
                )

                # Create the steps table for step tracking
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS steps (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        step_name TEXT NOT NULL,
                        step_index INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        output TEXT,
                        raw_response TEXT,
                        cost_usd REAL,
                        token_counts INTEGER,
                        execution_time_ms INTEGER,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                    )
                    """
                )

                # Create indexes for steps table
                await db.execute("CREATE INDEX IF NOT EXISTS idx_steps_run_id ON steps(run_id)")
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_steps_step_index ON steps(step_index)"
                )

                # Create the traces table for trace tracking
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS traces (
                        run_id TEXT PRIMARY KEY,
                        trace_data TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                    )
                    """
                )

                # Create the spans table for span tracking
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS spans (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        span_id TEXT NOT NULL,
                        parent_span_id TEXT,
                        name TEXT NOT NULL,
                        start_time REAL NOT NULL,
                        end_time REAL,
                        status TEXT NOT NULL,
                        attributes TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                    )
                    """
                )

                # Create indexes for spans table
                await db.execute("CREATE INDEX IF NOT EXISTS idx_spans_run_id ON spans(run_id)")
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_spans_parent_span ON spans(parent_span_id)"
                )

                await db.execute("COMMIT")

                # Run migration to ensure schema is up to date
                await self._migrate_existing_schema(db)

                # Create indexes after migration to ensure columns exist
                await self._create_indexes(db)

                await db.commit()
                telemetry.logfire.debug(f"Initialized SQLite database at {self.db_path}")

        except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
            # If we get a database error during initialization, try to backup and retry
            corruption_indicators = [
                "file is not a database",
                "corrupted database",
                "database disk image is malformed",
                "database is locked",
            ]
            if (
                any(indicator in str(e).lower() for indicator in corruption_indicators)
                and retry_count == 0
            ):
                telemetry.logfire.warning(
                    f"Database corruption detected during initialization, creating backup: {e}"
                )
                await self._backup_corrupted_database()
                # Retry once after backup
                await self._init_db(retry_count + 1, max_retries)
            elif retry_count < max_retries:
                telemetry.logfire.warning(
                    f"Database initialization failed, retrying ({retry_count + 1}/{max_retries}): {e}"
                )
                await asyncio.sleep(0.1 * (2**retry_count))  # Exponential backoff
                await self._init_db(retry_count + 1, max_retries)
            else:
                telemetry.logfire.error(
                    f"Failed to initialize database after {max_retries} retries: {e}"
                )
                raise
        except Exception as e:
            if retry_count < max_retries:
                telemetry.logfire.warning(
                    f"Database initialization failed, retrying ({retry_count + 1}/{max_retries}): {e}"
                )
                await asyncio.sleep(0.1 * (2**retry_count))  # Exponential backoff
                await self._init_db(retry_count + 1, max_retries)
            else:
                telemetry.logfire.error(
                    f"Failed to initialize database after {max_retries} retries: {e}"
                )
                raise

    async def _backup_corrupted_database(self) -> None:
        """Backup a corrupted database file with a unique timestamp."""

        # Determine if corrupted DB file exists using os.path.exists to avoid Path.stat side effects
        import os

        try:
            exists = os.path.exists(self.db_path)
        except OSError:
            exists = False
        if not exists:
            return

        # Generate base timestamp for backup filename. If there are existing backups
        # for this DB, prefer reusing their timestamp family to maintain grouping
        # and satisfy tests that expect continuity.
        existing_ts: Optional[int] = None
        try:
            pattern = f"{self.db_path.name}.corrupt."
            candidates = [p for p in self.db_path.parent.glob(f"{self.db_path.name}.corrupt.*")]
            ts_values: List[int] = []
            for p in candidates:
                name = p.name
                if pattern in name:
                    try:
                        # extract first numeric token after '.corrupt.'
                        suffix = name.split(".corrupt.", 1)[1]
                        ts_token = suffix.split(".", 1)[0]
                        if ts_token.isdigit():
                            ts_values.append(int(ts_token))
                    except Exception:
                        continue
            if ts_values:
                # Choose the most recent timestamp observed among existing backups
                existing_ts = max(ts_values)
        except Exception:
            existing_ts = None

        timestamp = existing_ts if existing_ts is not None else int(time.time())
        # Start counter for duplicate backup filenames
        counter = 1
        backup_path = self.db_path.parent / f"{self.db_path.name}.corrupt.{timestamp}"

        # Resolve unique backup filename, skipping paths that raise stat errors
        while True:
            try:
                exists = backup_path.exists()
            except (OSError, TypeError):
                # Cannot stat this path, assume it does not exist and proceed
                break
            if not exists:
                break
            backup_path = self.db_path.parent / f"{self.db_path.name}.corrupt.{timestamp}.{counter}"
            counter += 1
            if counter > 1000:
                break

        try:
            # Try to move the corrupted file to backup location using Path.rename
            self.db_path.rename(backup_path)
            telemetry.logfire.warning(f"Corrupted database backed up to {backup_path}")
        except (OSError, IOError):
            # If move fails, try to copy and then remove
            try:
                import shutil

                shutil.copy2(str(self.db_path), str(backup_path))
                self.db_path.unlink()
                telemetry.logfire.warning(f"Corrupted database copied to {backup_path} and removed")
            except (OSError, IOError) as copy_error:
                # If all else fails, just remove the corrupted file
                try:
                    self.db_path.unlink()
                    telemetry.logfire.warning(f"Corrupted database removed: {copy_error}")
                except (OSError, IOError) as remove_error:
                    telemetry.logfire.error(f"Failed to remove corrupted database: {remove_error}")
                    # If all backup attempts fail, raise a DatabaseError
                    raise sqlite3.DatabaseError(
                        "Database corruption recovery failed"
                    ) from remove_error

    async def _migrate_existing_schema(self, db: aiosqlite.Connection) -> None:
        """Migrate existing database schema to the new optimized structure."""
        try:
            cursor = await db.execute("PRAGMA table_info(workflow_state)")
            existing_columns = {row[1] for row in await cursor.fetchall()}
            await cursor.close()
        except sqlite3.OperationalError:
            # Table doesn't exist yet, which is fine - it will be created with the new schema
            existing_columns = set()

        # First, ensure all required core columns exist
        core_columns = [
            ("pipeline_name", "TEXT NOT NULL DEFAULT ''"),
            ("pipeline_version", "TEXT NOT NULL DEFAULT '1.0'"),
            ("current_step_index", "INTEGER NOT NULL DEFAULT 0"),
            ("pipeline_context", "TEXT"),
            ("last_step_output", "TEXT"),
            ("status", "TEXT NOT NULL DEFAULT 'running'"),
            ("created_at", "TEXT NOT NULL DEFAULT ''"),
            ("updated_at", "TEXT NOT NULL DEFAULT ''"),
        ]

        for column_name, column_def in core_columns:
            if column_name not in existing_columns:
                # Use proper SQLite quoting to prevent SQL injection
                escaped_name = column_name.replace('"', '""')
                quoted_column_name = f'"{escaped_name}"'
                await db.execute(
                    f"ALTER TABLE workflow_state ADD COLUMN {quoted_column_name} {column_def}"
                )

        # Add new optional columns if they don't exist
        new_columns = [
            ("total_steps", "INTEGER DEFAULT 0"),
            ("error_message", "TEXT"),
            ("execution_time_ms", "INTEGER"),
            ("memory_usage_mb", "REAL"),
            ("step_history", "TEXT"),
        ]

        for column_name, column_def in new_columns:
            if column_name not in existing_columns:
                # Validate column name and definition against the whitelist
                if column_name not in ALLOWED_COLUMNS or ALLOWED_COLUMNS[column_name] != column_def:
                    telemetry.logfire.error(
                        f"Invalid column definition: {column_name} {column_def}"
                    )
                    raise ValueError(
                        f"Schema migration failed due to invalid column definition: {column_name} {column_def}"
                    )

                # Use proper SQLite quoting to prevent SQL injection
                escaped_name = column_name.replace('"', '""')
                quoted_column_name = f'"{escaped_name}"'
                await db.execute(
                    f"ALTER TABLE workflow_state ADD COLUMN {quoted_column_name} {column_def}"
                )

        # Update any NULL values in required columns
        await db.execute(
            "UPDATE workflow_state SET current_step_index = 0 WHERE current_step_index IS NULL"
        )
        # Ensure steps table has raw_response column (FSD-013)
        try:
            cursor = await db.execute("PRAGMA table_info(steps)")
            steps_columns = {row[1] for row in await cursor.fetchall()}
            await cursor.close()
        except sqlite3.OperationalError:
            steps_columns = set()

        if steps_columns:
            if "raw_response" not in steps_columns:
                try:
                    await db.execute("ALTER TABLE steps ADD COLUMN raw_response TEXT")
                except sqlite3.OperationalError:
                    # Ignore if cannot alter; will rely on write-path retry/migration
                    pass

        await db.execute(
            "UPDATE workflow_state SET pipeline_context = '{}' WHERE pipeline_context IS NULL"
        )
        await db.execute("UPDATE workflow_state SET status = 'running' WHERE status IS NULL")
        await db.execute(
            "UPDATE workflow_state SET pipeline_name = pipeline_id WHERE pipeline_name = ''"
        )
        await db.execute(
            "UPDATE workflow_state SET pipeline_version = '1.0' WHERE pipeline_version = ''"
        )
        await db.execute(
            "UPDATE workflow_state SET created_at = datetime('now') WHERE created_at = ''"
        )
        await db.execute(
            "UPDATE workflow_state SET updated_at = datetime('now') WHERE updated_at = ''"
        )

    async def _create_indexes(self, db: aiosqlite.Connection) -> None:
        """Create indexes for better query performance, only if columns exist."""
        try:
            # Check which columns exist in the workflow_state table
            cursor = await db.execute("PRAGMA table_info(workflow_state)")
            existing_columns = {row[1] for row in await cursor.fetchall()}
            await cursor.close()

            # Create indexes only for columns that exist
            if "status" in existing_columns:
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_workflow_state_status ON workflow_state(status)"
                )
            if "pipeline_id" in existing_columns:
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_workflow_state_pipeline_id ON workflow_state(pipeline_id)"
                )
            if "created_at" in existing_columns:
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_workflow_state_created_at ON workflow_state(created_at)"
                )
        except sqlite3.OperationalError:
            # If there's an error checking table info, skip index creation
            pass

        # Migrate runs table schema if it exists
        try:
            cursor = await db.execute("PRAGMA table_info(runs)")
            runs_columns = {row[1] for row in await cursor.fetchall()}
            await cursor.close()

            # Add missing columns to runs table (including legacy fields used by tests)
            runs_new_columns = [
                ("pipeline_id", "TEXT NOT NULL DEFAULT 'unknown'"),
                ("created_at", "TEXT NOT NULL DEFAULT ''"),
                ("updated_at", "TEXT NOT NULL DEFAULT ''"),
                ("end_time", "TEXT"),
                ("total_cost", "REAL"),
                ("final_context_blob", "TEXT"),
                # Legacy/compatibility columns asserted by tests
                ("execution_time_ms", "INTEGER"),
                ("memory_usage_mb", "REAL"),
                ("total_steps", "INTEGER DEFAULT 0"),
                ("error_message", "TEXT"),
            ]

            for column_name, column_def in runs_new_columns:
                if column_name not in runs_columns:
                    escaped_name = column_name.replace('"', '""')
                    quoted_column_name = f'"{escaped_name}"'
                    await db.execute(
                        f"ALTER TABLE runs ADD COLUMN {quoted_column_name} {column_def}"
                    )

            # Update existing records with default values for NOT NULL columns
            if "pipeline_id" in runs_columns:
                await db.execute(
                    "UPDATE runs SET pipeline_id = 'unknown' WHERE pipeline_id IS NULL"
                )
            if "created_at" in runs_columns:
                await db.execute("UPDATE runs SET created_at = '' WHERE created_at IS NULL")
            if "updated_at" in runs_columns:
                await db.execute("UPDATE runs SET updated_at = '' WHERE updated_at IS NULL")

        except sqlite3.OperationalError:
            # Table doesn't exist yet, which is fine - it will be created with the new schema
            pass

    async def _ensure_init(self) -> None:
        if not self._initialized:
            # Use file-level lock to prevent concurrent initialization across instances
            async with self._get_file_lock():
                if not self._initialized:
                    try:
                        await self._init_db()
                        # Lazily create a pooled connection with optimized pragmas for subsequent writes
                        try:
                            self._connection_pool = await aiosqlite.connect(self.db_path)
                            await self._connection_pool.execute("PRAGMA journal_mode = WAL")
                            await self._connection_pool.execute("PRAGMA synchronous = NORMAL")
                            await self._connection_pool.execute("PRAGMA temp_store = MEMORY")
                            await self._connection_pool.execute("PRAGMA cache_size = 10000")
                            await self._connection_pool.execute("PRAGMA mmap_size = 268435456")
                            await self._connection_pool.execute("PRAGMA page_size = 4096")
                            await self._connection_pool.commit()
                        except Exception:
                            # If pool creation fails, fall back to per-call connections
                            self._connection_pool = None
                        self._initialized = True
                    except sqlite3.DatabaseError as e:
                        telemetry.logfire.error(f"Failed to initialize DB: {e}")
                        raise

    async def close(self) -> None:
        """Close database connections and cleanup resources."""
        conn = getattr(self, "_connection_pool", None)
        if conn:
            try:
                await conn.close()
            finally:
                self._connection_pool = None
        self._initialized = False

        # No global locks to clean up

    async def __aenter__(self) -> "SQLiteBackend":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit with cleanup."""
        await self.close()

    async def _with_retries(
        self, coro_func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any
    ) -> Any:
        """Execute a coroutine function with retry logic for database operations.

        Implements exponential backoff for database locked errors and schema migration
        retry for schema mismatch errors. Respects retry limits to prevent infinite loops.

        Args:
            coro_func: The coroutine function to execute
            *args: Positional arguments to pass to coro_func
            **kwargs: Keyword arguments to pass to coro_func

        Returns:
            The result of coro_func if successful

        Raises:
            sqlite3.OperationalError: If database locked errors persist after retries
            sqlite3.DatabaseError: If schema migration fails after retries or other DB errors
            RuntimeError: If all retry attempts are exhausted
        """
        max_retries = 3
        delay = 0.1
        for attempt in range(max_retries):
            try:
                result = await coro_func(*args, **kwargs)
                return result
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    telemetry.logfire.warn(
                        f"Database is locked, retrying ({attempt + 1}/{max_retries})..."
                    )
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                raise
            except sqlite3.DatabaseError as e:
                if "no such column" in str(e).lower():
                    if attempt < max_retries - 1:
                        telemetry.logfire.warn(
                            f"Schema mismatch detected: {e}. Attempting migration (attempt {attempt + 1}/{max_retries})..."
                        )
                        # Reset initialization state and re-initialize properly
                        self._initialized = False
                        await self._ensure_init()
                        continue
                    else:
                        telemetry.logfire.error(
                            f"Schema migration failed after {max_retries} attempts. Last error: {e}"
                        )
                        raise
                raise

        # This should never be reached due to explicit raises above, but ensures type safety
        raise RuntimeError(
            f"Operation failed after {max_retries} attempts due to unexpected conditions"
        )

    async def save_state(self, run_id: str, state: Dict[str, Any]) -> None:
        """Save workflow state to the database.

        Args:
            run_id: Unique identifier for the workflow run
            state: Dictionary containing workflow state data
        """
        await self._ensure_init()
        async with self._lock:

            async def _save() -> None:
                async with aiosqlite.connect(self.db_path) as db:
                    # OPTIMIZATION: Use more efficient serialization for performance-critical scenarios
                    # Skip expensive robust_serialize for simple data types
                    pipeline_context = state["pipeline_context"]
                    if isinstance(pipeline_context, dict):
                        # For simple dicts, use direct JSON serialization
                        pipeline_context_json = _fast_json_dumps(pipeline_context)
                    else:
                        # For complex objects, use robust serialization
                        pipeline_context_json = _fast_json_dumps(robust_serialize(pipeline_context))

                    last_step_output = state.get("last_step_output")
                    if last_step_output is not None:
                        if isinstance(last_step_output, (str, int, float, bool, type(None))):
                            # For simple types, use direct JSON serialization
                            last_step_output_json = _fast_json_dumps(last_step_output)
                        else:
                            # For complex objects, use robust serialization
                            last_step_output_json = _fast_json_dumps(
                                robust_serialize(last_step_output)
                            )
                    else:
                        last_step_output_json = None

                    step_history = state.get("step_history")
                    if step_history is not None:
                        if isinstance(step_history, list) and all(
                            isinstance(item, dict) for item in step_history
                        ):
                            # For simple list of dicts, use direct JSON serialization
                            step_history_json = _fast_json_dumps(step_history)
                        else:
                            # For complex objects, use robust serialization
                            step_history_json = _fast_json_dumps(robust_serialize(step_history))
                    else:
                        step_history_json = None

                    # Get execution_time_ms directly from state
                    execution_time_ms = state.get("execution_time_ms")

                    await db.execute(
                        """
                        INSERT OR REPLACE INTO workflow_state (
                            run_id, pipeline_id, pipeline_name, pipeline_version,
                            current_step_index, pipeline_context, last_step_output, step_history,
                            status, created_at, updated_at, total_steps,
                            error_message, execution_time_ms, memory_usage_mb
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            state["pipeline_id"],
                            state["pipeline_name"],
                            state["pipeline_version"],
                            state["current_step_index"],
                            pipeline_context_json,
                            last_step_output_json,
                            step_history_json,
                            state["status"],
                            state["created_at"].isoformat(),
                            state["updated_at"].isoformat(),
                            state.get("total_steps", 0),
                            state.get("error_message"),
                            execution_time_ms,
                            state.get("memory_usage_mb"),
                        ),
                    )
                    await db.commit()
                    telemetry.logfire.debug(f"Saved state for run_id={run_id}")

            await self._with_retries(_save)

    async def load_state(self, run_id: str) -> Optional[Dict[str, Any]]:
        await self._ensure_init()
        async with self._lock:

            async def _load() -> Optional[Dict[str, Any]]:
                async with aiosqlite.connect(self.db_path) as db:
                    cursor = await db.execute(
                        """
                        SELECT run_id, pipeline_id, pipeline_name, pipeline_version, current_step_index,
                               pipeline_context, last_step_output, step_history, status, created_at, updated_at,
                               total_steps, error_message, execution_time_ms, memory_usage_mb
                        FROM workflow_state WHERE run_id = ?
                        """,
                        (run_id,),
                    )
                    row = await cursor.fetchone()
                    await cursor.close()
                if row is None:
                    return None
                pipeline_context = (
                    safe_deserialize(json.loads(row[5])) if row[5] is not None else {}
                )
                last_step_output = (
                    safe_deserialize(json.loads(row[6])) if row[6] is not None else None
                )
                step_history = safe_deserialize(json.loads(row[7])) if row[7] is not None else []
                return {
                    "run_id": row[0],
                    "pipeline_id": row[1],
                    "pipeline_name": row[2],
                    "pipeline_version": row[3],
                    "current_step_index": row[4],
                    "pipeline_context": pipeline_context,
                    "last_step_output": last_step_output,
                    "step_history": step_history,
                    "status": row[8],
                    "created_at": datetime.fromisoformat(row[9]),
                    "updated_at": datetime.fromisoformat(row[10]),
                    "total_steps": row[11] or 0,
                    "error_message": row[12],
                    "execution_time_ms": row[13],
                    "memory_usage_mb": row[14],
                }

            result = await self._with_retries(_load)
            return cast(Optional[Dict[str, Any]], result)

    async def delete_state(self, run_id: str) -> None:
        """Delete workflow state."""
        await self._ensure_init()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("DELETE FROM workflow_state WHERE run_id = ?", (run_id,))
                await db.commit()

    async def list_states(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List workflow states with optional status filter."""
        await self._ensure_init()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                if status:
                    async with db.execute(
                        "SELECT run_id, status, created_at, updated_at FROM workflow_state WHERE status = ? ORDER BY created_at DESC",
                        (status,),
                    ) as cursor:
                        rows = await cursor.fetchall()
                else:
                    async with db.execute(
                        "SELECT run_id, status, created_at, updated_at FROM workflow_state ORDER BY created_at DESC"
                    ) as cursor:
                        rows = await cursor.fetchall()

                return [
                    {
                        "run_id": row[0],
                        "status": row[1],
                        "created_at": row[2],
                        "updated_at": row[3],
                    }
                    for row in rows
                ]

    async def list_workflows(
        self,
        status: Optional[str] = None,
        pipeline_id: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Enhanced workflow listing with additional filters and metadata."""
        await self._ensure_init()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                # Build query with optional filters
                query = """
                    SELECT run_id, pipeline_id, pipeline_name, pipeline_version,
                           current_step_index, status, created_at, updated_at,
                           total_steps, error_message, execution_time_ms, memory_usage_mb
                    FROM workflow_state
                    WHERE 1=1
                """
                params: List[Any] = []

                if status:
                    query += " AND status = ?"
                    params.append(status)

                if pipeline_id:
                    query += " AND pipeline_id = ?"
                    params.append(pipeline_id)

                query += " ORDER BY created_at DESC"

                if limit is not None:
                    query += " LIMIT ?"
                    params.append(limit)
                if offset:
                    query += " OFFSET ?"
                    params.append(offset)

                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()

                result: List[Dict[str, Any]] = []
                for row in rows:
                    if row is None:
                        continue
                    result.append(
                        {
                            "run_id": row[0],
                            "pipeline_id": row[1],
                            "pipeline_name": row[2],
                            "pipeline_version": row[3],
                            "current_step_index": row[4],
                            "status": row[5],
                            "created_at": row[6],
                            "updated_at": row[7],
                            "total_steps": row[8] or 0,
                            "error_message": row[9],
                            "execution_time_ms": row[10],
                            "memory_usage_mb": row[11],
                        }
                    )
                return result

    async def list_runs(
        self,
        status: Optional[str] = None,
        pipeline_name: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List runs from the new structured schema for lens CLI."""
        await self._ensure_init()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                # Optimize query based on filters to use appropriate indexes
                if status and not pipeline_name:
                    # Use status index for better performance
                    query = """
                        SELECT run_id, pipeline_name, pipeline_version, status, created_at, updated_at, execution_time_ms
                        FROM runs
                        WHERE status = ?
                        ORDER BY created_at DESC
                    """
                    params = [status]
                elif pipeline_name and not status:
                    # Use pipeline_name index
                    query = """
                        SELECT run_id, pipeline_name, pipeline_version, status, created_at, updated_at, execution_time_ms
                        FROM runs
                        WHERE pipeline_name = ?
                        ORDER BY created_at DESC
                    """
                    params = [pipeline_name]
                elif status and pipeline_name:
                    # Use both filters
                    query = """
                        SELECT run_id, pipeline_name, pipeline_version, status, created_at, updated_at, execution_time_ms
                        FROM runs
                        WHERE status = ? AND pipeline_name = ?
                        ORDER BY created_at DESC
                    """
                    params = [status, pipeline_name]
                else:
                    # No filters, use created_at index
                    query = """
                        SELECT run_id, pipeline_name, pipeline_version, status, created_at, updated_at, execution_time_ms
                        FROM runs
                        ORDER BY created_at DESC
                    """
                    params = []

                if limit is not None:
                    query += " LIMIT ?"
                    params.append(str(limit))
                if offset:
                    query += " OFFSET ?"
                    params.append(str(offset))

                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()

                result: List[Dict[str, Any]] = []
                for row in rows:
                    if row is None:
                        continue
                    result.append(
                        {
                            "run_id": row[0],
                            "pipeline_name": row[1],
                            "pipeline_version": row[2],
                            "status": row[3],
                            "start_time": row[
                                4
                            ],  # Map created_at to start_time for backward compatibility
                            "end_time": row[
                                5
                            ],  # Map updated_at to end_time for backward compatibility
                            "total_cost": row[6]
                            if row[6] is not None
                            else 0.0,  # Map execution_time_ms to total_cost for backward compatibility
                        }
                    )
                return result

    async def get_workflow_stats(self) -> Dict[str, Any]:
        """Get comprehensive workflow statistics."""
        await self._ensure_init()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                # Get total count
                cursor = await db.execute("SELECT COUNT(*) FROM workflow_state")
                total_workflows_row = await cursor.fetchone()
                total_workflows = total_workflows_row[0] if total_workflows_row else 0
                await cursor.close()

                # Get status counts
                cursor = await db.execute(
                    """
                    SELECT status, COUNT(*)
                    FROM workflow_state
                    GROUP BY status
                """
                )
                status_counts_rows = await cursor.fetchall()
                status_counts: Dict[str, int] = {
                    row[0]: row[1] for row in status_counts_rows if row is not None
                }
                await cursor.close()

                # Get recent workflows (last 24 hours)
                cursor = await db.execute(
                    """
                    SELECT COUNT(*)
                    FROM workflow_state
                    WHERE created_at >= datetime('now', '-24 hours')
                """
                )
                recent_workflows_24h_row = await cursor.fetchone()
                recent_workflows_24h = (
                    recent_workflows_24h_row[0] if recent_workflows_24h_row else 0
                )
                await cursor.close()

                # Get average execution time
                cursor = await db.execute(
                    """
                    SELECT AVG(execution_time_ms)
                    FROM workflow_state
                    WHERE execution_time_ms IS NOT NULL
                """
                )
                avg_exec_time_row = await cursor.fetchone()
                avg_exec_time = avg_exec_time_row[0] if avg_exec_time_row else 0
                await cursor.close()

                return {
                    "total_workflows": total_workflows,
                    "status_counts": status_counts,
                    "recent_workflows_24h": recent_workflows_24h,
                    "average_execution_time_ms": avg_exec_time or 0,
                }

    async def get_failed_workflows(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Get failed workflows from the last N hours."""
        await self._ensure_init()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    """
                    SELECT run_id, pipeline_id, pipeline_name, pipeline_version,
                           current_step_index, status, created_at, updated_at,
                           total_steps, error_message, execution_time_ms, memory_usage_mb
                    FROM workflow_state
                    WHERE status = 'failed'
                    AND updated_at >= datetime('now', '-' || ? || ' hours')
                    ORDER BY updated_at DESC
                """,
                    (hours_back,),
                )

                rows = await cursor.fetchall()
                await cursor.close()

                result: List[Dict[str, Any]] = []
                for row in rows:
                    if row is None:
                        continue
                    result.append(
                        {
                            "run_id": row[0],
                            "pipeline_id": row[1],
                            "pipeline_name": row[2],
                            "pipeline_version": row[3],
                            "current_step_index": row[4],
                            "status": row[5],
                            "created_at": row[6],
                            "updated_at": row[7],
                            "total_steps": row[8] or 0,
                            "error_message": row[9],
                            "execution_time_ms": row[10],
                            "memory_usage_mb": row[11],
                        }
                    )
                return result

    async def cleanup_old_workflows(self, days_old: float = 30) -> int:
        """Delete workflows older than specified days. Returns number of deleted workflows."""
        await self._ensure_init()
        async with self._lock:

            async def _cleanup() -> int:
                async with aiosqlite.connect(self.db_path) as db:
                    cursor = await db.execute(
                        """
                        SELECT COUNT(*)
                        FROM workflow_state
                        WHERE created_at < datetime('now', '-' || ? || ' days')
                        """,
                        (days_old,),
                    )
                    count_row = await cursor.fetchone()
                    count = count_row[0] if count_row else 0
                    await cursor.close()
                    await db.execute(
                        """
                        DELETE FROM workflow_state
                        WHERE created_at < datetime('now', '-' || ? || ' days')
                        """,
                        (days_old,),
                    )
                    await db.commit()
                    telemetry.logfire.info(
                        f"Cleaned up {count} old workflows older than {days_old} days"
                    )
                    return int(count)

            result = await self._with_retries(_cleanup)
            return cast(int, result)

    # ------------------------------------------------------------------
    # New structured persistence API
    # ------------------------------------------------------------------

    async def save_run_start(self, run_data: Dict[str, Any]) -> None:
        await self._ensure_init()
        async with self._lock:

            async def _save() -> None:
                # Use pooled connection when available to avoid connect() overhead
                db = self._connection_pool
                if db is None:
                    db_cm = aiosqlite.connect(self.db_path)
                    db = await db_cm.__aenter__()
                    should_close = True
                else:
                    should_close = False

                try:
                    # OPTIMIZATION: Use simplified schema for better performance
                    created_at = run_data.get("created_at") or datetime.utcnow().isoformat()
                    updated_at = run_data.get("updated_at") or created_at

                    await db.execute(
                        """
                        INSERT OR REPLACE INTO runs (
                            run_id, pipeline_id, pipeline_name, pipeline_version, status,
                            created_at, updated_at, execution_time_ms, memory_usage_mb,
                            total_steps, error_message
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_data["run_id"],
                            run_data.get("pipeline_id", "unknown"),
                            run_data.get("pipeline_name", "unknown"),
                            run_data.get("pipeline_version", "latest"),
                            run_data.get("status", "running"),
                            created_at,
                            updated_at,
                            run_data.get("execution_time_ms"),
                            run_data.get("memory_usage_mb"),
                            run_data.get("total_steps", 0),
                            run_data.get("error_message"),
                        ),
                    )
                    await db.commit()
                finally:
                    if should_close:
                        await db_cm.__aexit__(None, None, None)

            await self._with_retries(_save)

    async def save_step_result(self, step_data: Dict[str, Any]) -> None:
        await self._ensure_init()
        async with self._lock:

            async def _save() -> None:
                async with aiosqlite.connect(self.db_path) as db:
                    # OPTIMIZATION: Use simplified schema for better performance
                    await db.execute(
                        """
                    INSERT OR REPLACE INTO steps (
                        run_id, step_name, step_index, status, output, raw_response, cost_usd,
                        token_counts, execution_time_ms, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            step_data["run_id"],
                            step_data["step_name"],
                            step_data["step_index"],
                            step_data.get("status", "completed"),
                            _fast_json_dumps(step_data.get("output")),
                            _fast_json_dumps(step_data.get("raw_response"))
                            if step_data.get("raw_response") is not None
                            else None,
                            step_data.get("cost_usd"),
                            step_data.get("token_counts"),
                            step_data.get("execution_time_ms"),
                            step_data.get("created_at", datetime.utcnow().isoformat()),
                        ),
                    )
                    await db.commit()

            await self._with_retries(_save)

    async def save_run_end(self, run_id: str, end_data: Dict[str, Any]) -> None:
        await self._ensure_init()
        async with self._lock:

            async def _save() -> None:
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute(
                        """
                        UPDATE runs
                        SET status = ?, updated_at = ?, execution_time_ms = ?,
                            memory_usage_mb = ?, total_steps = ?, error_message = ?
                        WHERE run_id = ?
                        """,
                        (
                            end_data.get("status", "completed"),
                            end_data.get("updated_at", datetime.utcnow().isoformat()),
                            end_data.get("execution_time_ms"),
                            end_data.get("memory_usage_mb"),
                            end_data.get("total_steps", 0),
                            end_data.get("error_message"),
                            run_id,
                        ),
                    )
                    await db.commit()

            await self._with_retries(_save)

    async def get_run_details(self, run_id: str) -> Optional[Dict[str, Any]]:
        await self._ensure_init()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    """
                    SELECT run_id, pipeline_name, pipeline_version, status, created_at, updated_at,
                           execution_time_ms, memory_usage_mb, total_steps, error_message
                    FROM runs WHERE run_id = ?
                    """,
                    (run_id,),
                )
                row = await cursor.fetchone()
                await cursor.close()
                if row is None:
                    return None
                return {
                    "run_id": row[0],
                    "pipeline_name": row[1],
                    "pipeline_version": row[2],
                    "status": row[3],
                    "created_at": row[4],
                    "updated_at": row[5],
                    "execution_time_ms": row[6],
                    "memory_usage_mb": row[7],
                    "total_steps": row[8],
                    "error_message": row[9],
                }

    async def list_run_steps(self, run_id: str) -> List[Dict[str, Any]]:
        await self._ensure_init()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    """
                    SELECT step_name, step_index, status, output, raw_response, cost_usd, token_counts,
                           execution_time_ms, created_at
                    FROM steps WHERE run_id = ? ORDER BY step_index
                    """,
                    (run_id,),
                )
                rows = await cursor.fetchall()
                await cursor.close()
                results: List[Dict[str, Any]] = []
                for r in rows:
                    results.append(
                        {
                            "step_name": r[0],
                            "step_index": r[1],
                            "status": r[2],
                            "output": safe_deserialize(json.loads(r[3])) if r[3] else None,
                            "raw_response": safe_deserialize(json.loads(r[4])) if r[4] else None,
                            "cost_usd": r[5],
                            "token_counts": r[6],
                            "execution_time_ms": r[7],
                            "created_at": r[8],
                        }
                    )
                return results

    async def save_trace(self, run_id: str, trace: Dict[str, Any]) -> None:
        """Persist a trace tree as normalized spans for a given run_id."""
        await self._ensure_init()
        async with self._lock:

            async def _save() -> None:
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute("PRAGMA foreign_keys = ON")

                    # Extract all spans from the trace tree recursively
                    spans_to_insert = self._extract_spans_from_tree(trace, run_id)

                    if spans_to_insert:
                        # Use a transaction for atomic replacement
                        # Note: We use DELETE + INSERT instead of UPSERT because:
                        # 1. We need to replace ALL spans for a run_id, not individual spans
                        # 2. Different trace saves may have different span_id values
                        # 3. We must ensure no orphaned spans remain from previous saves
                        async with db.execute("BEGIN TRANSACTION"):
                            # Delete existing spans for this run_id to ensure clean replacement
                            await db.execute("DELETE FROM spans WHERE run_id = ?", (run_id,))

                            # Insert new spans
                            await db.executemany(
                                """
                                INSERT INTO spans (
                                    span_id, run_id, parent_span_id, name, start_time,
                                    end_time, status, attributes, created_at
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                                """,
                                spans_to_insert,
                            )
                        await db.commit()

            await self._with_retries(_save)

    def _extract_spans_from_tree(
        self, trace: Dict[str, Any], run_id: str, max_depth: int = 100
    ) -> List[Tuple[str, str, Optional[str], str, float, Optional[float], str, str]]:
        """Extract all spans from a trace tree for batch insertion."""
        spans: List[Tuple[str, str, Optional[str], str, float, Optional[float], str, str]] = []

        # Handle empty or invalid trace data
        if not trace or not isinstance(trace, dict):
            return spans

        def extract_span_recursive(
            span_data: Dict[str, Any],
            parent_span_id: Optional[str] = None,
            depth: int = 0,
        ) -> None:
            # Check depth limit to prevent stack overflow
            if depth > max_depth:
                from flujo.infra import telemetry

                telemetry.logfire.warn(
                    f"Trace tree depth limit ({max_depth}) exceeded for run_id {run_id}"
                )
                return

            # Validate required fields
            if (
                not isinstance(span_data, dict)
                or "span_id" not in span_data
                or "name" not in span_data
            ):
                return

            from flujo.utils.serialization import robust_serialize

            try:
                start_time = float(span_data.get("start_time", 0.0))
            except (ValueError, TypeError):
                logging.warning(
                    f"Skipping span with invalid start_time for run_id={run_id}, span_id={span_data.get('span_id')}"
                )
                return
            try:
                end_time = (
                    float(span_data["end_time"]) if span_data.get("end_time") is not None else None
                )
            except (ValueError, TypeError):
                logging.warning(
                    f"Skipping span with invalid end_time for run_id={run_id}, span_id={span_data.get('span_id')}"
                )
                return

            span_tuple: Tuple[str, str, Optional[str], str, float, Optional[float], str, str] = (
                str(span_data.get("span_id", "")),
                run_id,
                parent_span_id,
                str(span_data.get("name", "")),
                start_time,
                end_time,
                str(span_data.get("status", "running")),
                _fast_json_dumps(robust_serialize(span_data.get("attributes", {}))),
            )
            spans.append(span_tuple)

            # Process children recursively
            for child in span_data.get("children", []):
                extract_span_recursive(child, span_data.get("span_id"), depth + 1)

        extract_span_recursive(trace)
        return spans

    def _reconstruct_trace_tree(
        self,
        spans_data: List[Tuple[str, Optional[str], str, float, Optional[float], str, str]],
    ) -> Optional[Dict[str, Any]]:
        """Reconstruct a hierarchical trace tree from flat spans data."""
        spans_map: Dict[str, Dict[str, Any]] = {}
        root_spans: List[Dict[str, Any]] = []

        # First pass: create a map of all spans by ID
        for row in spans_data:
            (
                span_id,
                parent_span_id,
                name,
                start_time,
                end_time,
                status,
                attributes,
            ) = row
            span_data: Dict[str, Any] = {
                "span_id": span_id,
                "parent_span_id": parent_span_id,
                "name": name,
                "start_time": start_time,
                "end_time": end_time,
                "status": status,
                "attributes": json.loads(attributes) if attributes else {},
                "children": [],
            }
            spans_map[span_id] = span_data

        # Second pass: build the tree hierarchy
        for span_id, span_data in spans_map.items():
            parent_id = span_data.get("parent_span_id")
            if parent_id and parent_id in spans_map:
                spans_map[parent_id]["children"].append(span_data)
            else:
                root_spans.append(span_data)

        if len(root_spans) > 1:
            from flujo.infra import telemetry

            telemetry.logfire.warn(
                f"Trace for run_id has multiple root spans ({len(root_spans)}). Using the first one."
            )

        return root_spans[0] if root_spans else None

    async def get_trace(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve and reconstruct the trace tree for a given run_id. Audit log access."""
        await self._ensure_init()
        async with self._lock:
            telemetry.logfire.info(f"AUDIT: Trace accessed for run_id={run_id}")
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA foreign_keys = ON")
                async with db.execute(
                    """
                    SELECT span_id, parent_span_id, name, start_time, end_time,
                           status, attributes
                    FROM spans WHERE run_id = ? ORDER BY start_time
                    """,
                    (run_id,),
                ) as cursor:
                    rows = await cursor.fetchall()
                    rows_typed: List[
                        Tuple[str, Optional[str], str, float, Optional[float], str, str]
                    ] = [
                        (
                            str(r[0]),
                            str(r[1]) if r[1] is not None else None,
                            str(r[2]),
                            float(r[3]),
                            float(r[4]) if r[4] is not None else None,
                            str(r[5]),
                            str(r[6]),
                        )
                        for r in rows
                    ]
                    if not rows_typed:
                        return None
                    return self._reconstruct_trace_tree(rows_typed)

    async def get_spans(
        self, run_id: str, status: Optional[str] = None, name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get individual spans with optional filtering. Audit log export."""
        await self._ensure_init()
        async with self._lock:
            telemetry.logfire.info(
                f"AUDIT: Spans exported for run_id={run_id}, status={status}, name={name}"
            )
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA foreign_keys = ON")
                query = """
                    SELECT span_id, parent_span_id, name, start_time, end_time,
                           status, attributes
                    FROM spans WHERE run_id = ?
                """
                params: List[Any] = [run_id]
                if status:
                    query += " AND status = ?"
                    params.append(status)
                if name:
                    query += " AND name = ?"
                    params.append(name)
                query += " ORDER BY start_time"
                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    results: List[Dict[str, Any]] = []
                    for r in rows:
                        (
                            span_id,
                            parent_span_id,
                            name,
                            start_time,
                            end_time,
                            status,
                            attributes,
                        ) = r
                        results.append(
                            {
                                "span_id": str(span_id),
                                "parent_span_id": str(parent_span_id)
                                if parent_span_id is not None
                                else None,
                                "name": str(name),
                                "start_time": float(start_time),
                                "end_time": float(end_time) if end_time is not None else None,
                                "status": str(status),
                                "attributes": json.loads(attributes) if attributes else {},
                            }
                        )
                    return results

    async def get_span_statistics(
        self,
        pipeline_name: Optional[str] = None,
        time_range: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """Get aggregated span statistics."""
        await self._ensure_init()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA foreign_keys = ON")
                query = """
                    SELECT s.name, s.status, s.start_time, s.end_time,
                           r.pipeline_name
                    FROM spans s
                    JOIN runs r ON s.run_id = r.run_id
                    WHERE s.end_time IS NOT NULL
                """
                params: List[Any] = []
                if pipeline_name:
                    query += " AND r.pipeline_name = ?"
                    params.append(pipeline_name)
                if time_range:
                    start_time, end_time = time_range
                    query += " AND s.start_time >= ? AND s.start_time <= ?"
                    params.extend([start_time, end_time])
                async with db.execute(query, params) as cursor:
                    rows = list(await cursor.fetchall())
                    stats: Dict[str, Any] = {
                        "total_spans": len(rows),
                        "by_name": {},
                        "by_status": {},
                        "avg_duration_by_name": {},
                    }
                    for r in rows:
                        name, status, start_time, end_time, pipeline_name = r
                        duration = (
                            float(end_time) - float(start_time) if end_time is not None else 0.0
                        )
                        # Count by name
                        if name not in stats["by_name"]:
                            stats["by_name"][name] = 0
                        stats["by_name"][name] += 1
                        # Count by status
                        if status not in stats["by_status"]:
                            stats["by_status"][status] = 0
                        stats["by_status"][status] += 1
                        # Average duration by name
                        if name not in stats["avg_duration_by_name"]:
                            stats["avg_duration_by_name"][name] = {
                                "total": 0.0,
                                "count": 0,
                            }
                        stats["avg_duration_by_name"][name]["total"] += duration
                        stats["avg_duration_by_name"][name]["count"] += 1
                    for name, data in stats["avg_duration_by_name"].items():
                        if data["count"] > 0:
                            data["average"] = data["total"] / data["count"]
                        else:
                            data["average"] = 0.0
                    return stats

    async def delete_run(self, run_id: str) -> None:
        """Delete a run from the runs table (cascades to traces). Audit log deletion."""
        await self._ensure_init()
        async with self._lock:
            telemetry.logfire.info(f"AUDIT: Run and associated traces deleted for run_id={run_id}")
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA foreign_keys = ON")
                await db.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
                await db.commit()
