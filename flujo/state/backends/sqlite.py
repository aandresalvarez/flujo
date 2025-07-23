from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable, Any, Dict, List, Optional, cast, TYPE_CHECKING, Tuple

import aiosqlite
import sqlite3
import weakref

from .base import StateBackend
from ...utils.serialization import safe_deserialize, robust_serialize
from ...infra import telemetry

import re


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
    allowed_constraints = ["PRIMARY KEY", "UNIQUE", "NOT NULL", "DEFAULT", "CHECK", "COLLATE"]
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
            r"^(NULL|[0-9]+|[0-9]*\.[0-9]+|'.*?'|\".*?\"|TRUE|FALSE)$", val, re.IGNORECASE
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
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._initialized = False
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
        """Initialize the database with schema and indexes.

        Args:
            retry_count: Current retry attempt number
            max_retries: Maximum number of retry attempts for corruption recovery
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA foreign_keys = ON")
                await db.execute("PRAGMA journal_mode = WAL")
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS workflow_state (
                        run_id TEXT PRIMARY KEY,
                        pipeline_id TEXT NOT NULL,
                        pipeline_name TEXT NOT NULL,
                        pipeline_version TEXT NOT NULL,
                        current_step_index INTEGER NOT NULL DEFAULT 0,
                        pipeline_context TEXT NOT NULL,
                        last_step_output TEXT,
                        step_history TEXT,
                        status TEXT NOT NULL CHECK (status IN ('running', 'paused', 'completed', 'failed', 'cancelled')),
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        total_steps INTEGER DEFAULT 0,
                        error_message TEXT,
                        execution_time_ms INTEGER,
                        memory_usage_mb REAL
                    )
                    """
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_workflow_state_status ON workflow_state(status)"
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_workflow_state_created_at ON workflow_state(created_at)"
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_workflow_state_pipeline_id ON workflow_state(pipeline_id)"
                )
                # New structured tables for persistent run history
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
                        start_time TEXT,
                        end_time TEXT,
                        total_cost REAL,
                        final_context_blob TEXT
                    )
                    """
                )
                await db.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)")
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_runs_pipeline_name ON runs(pipeline_name)"
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_runs_start_time_desc ON runs(start_time DESC)"
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_runs_status_start_time ON runs(status, start_time DESC)"
                )
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS steps (
                        step_run_id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        step_name TEXT NOT NULL,
                        step_index INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        start_time TEXT NOT NULL,
                        end_time TEXT,
                        duration_ms INTEGER,
                        cost REAL,
                        tokens INTEGER,
                        input_blob TEXT,
                        output_blob TEXT,
                        error_blob TEXT,
                        FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                    )
                    """
                )
                await db.execute("CREATE INDEX IF NOT EXISTS idx_steps_run_id ON steps(run_id)")
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS spans (
                        span_id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        parent_span_id TEXT,
                        name TEXT NOT NULL,
                        start_time REAL NOT NULL,
                        end_time REAL,
                        status TEXT DEFAULT 'running',
                        attributes TEXT, -- JSON for flexible metadata
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
                        FOREIGN KEY (parent_span_id) REFERENCES spans(span_id) ON DELETE CASCADE
                    )
                    """
                )
                # Indexes for efficient querying
                await db.execute("CREATE INDEX IF NOT EXISTS idx_spans_run_id ON spans(run_id)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_spans_status ON spans(status)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_spans_name ON spans(name)")
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_spans_start_time ON spans(start_time)"
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_spans_parent_span_id ON spans(parent_span_id)"
                )
                await self._migrate_existing_schema(db)
                await db.commit()
            telemetry.logfire.info(f"Initialized SQLite database at {self.db_path}")
        except sqlite3.DatabaseError as e:
            if retry_count >= max_retries:
                telemetry.logfire.error(
                    f"Failed to initialize database after {max_retries} attempts: {e}"
                )
                raise
            telemetry.logfire.error(
                f"Database corruption detected: {e}. Reinitializing {self.db_path} (attempt {retry_count + 1}/{max_retries})."
            )
            # Create unique backup filename with timestamp to avoid conflicts
            timestamp = int(time.time())
            base_name = self.db_path.stem
            suffix = self.db_path.suffix

            # Find the first available backup path, handling gaps in sequence
            backup_path = None
            MAX_BACKUP_SUFFIX_ATTEMPTS = 100
            MAX_CLEANUP_ATTEMPTS = 10  # Prevent infinite cleanup loops
            cleanup_attempts = 0

            # First try the base path
            candidate_path = self.db_path.parent / f"{base_name}{suffix}.corrupt.{timestamp}"
            try:
                if not candidate_path.exists():
                    backup_path = candidate_path
            except OSError as stat_error:
                telemetry.logfire.warn(f"Could not stat backup path {candidate_path}: {stat_error}")

            # If base path is taken, find the first available gap in sequence
            if backup_path is None:
                for counter in range(1, MAX_BACKUP_SUFFIX_ATTEMPTS + 1):
                    candidate_path = (
                        self.db_path.parent / f"{base_name}{suffix}.corrupt.{timestamp}.{counter}"
                    )
                    try:
                        if not candidate_path.exists():
                            backup_path = candidate_path
                            break
                    except OSError as stat_error:
                        telemetry.logfire.warn(
                            f"Could not stat backup path {candidate_path}: {stat_error}"
                        )
                        continue

            # If no gaps found, we need to clean up old backups
            if backup_path is None:
                while cleanup_attempts < MAX_CLEANUP_ATTEMPTS:
                    cleanup_attempts += 1

                    # Find and remove the oldest backup file
                    backup_pattern = f"{base_name}{suffix}.corrupt.*"
                    try:
                        existing_backups = list(self.db_path.parent.glob(backup_pattern))
                        if existing_backups:
                            # Find oldest backup with proper exception handling
                            oldest_backup = None
                            oldest_time = float("inf")

                            for backup in existing_backups:
                                try:
                                    backup_time = backup.stat().st_mtime
                                    if backup_time < oldest_time:
                                        oldest_time = backup_time
                                        oldest_backup = backup
                                except OSError as stat_error:
                                    telemetry.logfire.warn(
                                        f"Could not stat backup file {backup}: {stat_error}"
                                    )
                                    continue

                            if oldest_backup:
                                telemetry.logfire.error(
                                    f"Too many backup files exist, removing oldest: {oldest_backup}"
                                )
                                try:
                                    oldest_backup.unlink(missing_ok=True)
                                    # After removing oldest, try the base path again
                                    candidate_path = (
                                        self.db_path.parent
                                        / f"{base_name}{suffix}.corrupt.{timestamp}"
                                    )
                                    try:
                                        if not candidate_path.exists():
                                            backup_path = candidate_path
                                            break
                                    except OSError as stat_error:
                                        telemetry.logfire.warn(
                                            f"Could not stat backup path {candidate_path} after cleanup: {stat_error}"
                                        )
                                except OSError as unlink_error:
                                    telemetry.logfire.error(
                                        f"Failed to remove oldest backup {oldest_backup}: {unlink_error}"
                                    )
                            else:
                                telemetry.logfire.warn("No valid backup files found to remove")
                    except OSError as glob_error:
                        telemetry.logfire.error(f"Failed to glob backup files: {glob_error}")

                # If still no path found after cleanup attempts, use a unique timestamp-based name
                if backup_path is None:
                    telemetry.logfire.error(
                        f"Failed to find available backup path after {MAX_CLEANUP_ATTEMPTS} cleanup attempts"
                    )
                    backup_path = (
                        self.db_path.parent
                        / f"{base_name}{suffix}.corrupt.{timestamp}.{int(time.time())}"
                    )

            try:
                self.db_path.rename(backup_path)
                telemetry.logfire.warn(f"Corrupted DB moved to {backup_path}")
            except (FileExistsError, OSError) as rename_error:
                telemetry.logfire.error(
                    f"Failed to rename corrupted DB to {backup_path}: {rename_error}"
                )
                # Fallback: try to remove the corrupted file
                try:
                    self.db_path.unlink(missing_ok=True)
                    telemetry.logfire.warn(f"Removed corrupted DB file: {self.db_path}")
                except OSError as unlink_error:
                    telemetry.logfire.error(f"Failed to remove corrupted DB: {unlink_error}")
                    raise sqlite3.DatabaseError(f"Database corruption recovery failed: {e}")

            await self._init_db(retry_count=retry_count + 1, max_retries=max_retries)

    async def _migrate_existing_schema(self, db: aiosqlite.Connection) -> None:
        """Migrate existing database schema to the new optimized structure."""
        cursor = await db.execute("PRAGMA table_info(workflow_state)")
        existing_columns = {row[1] for row in await cursor.fetchall()}
        await cursor.close()

        # Add new columns if they don't exist
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
                # Note: SQLite doesn't support parameterized DDL, so we use validation + proper quoting
                # The validation functions ensure safety before this point
                # Use proper SQLite identifier quoting for maximum security
                # SQLite doesn't have a built-in quote_identifier, so we use our own implementation
                escaped_name = column_name.replace('"', '""')
                quoted_column_name = f'"{escaped_name}"'
                await db.execute(
                    f"ALTER TABLE workflow_state ADD COLUMN {quoted_column_name} {column_def}"
                )

        # Ensure required columns exist with proper constraints
        if "pipeline_name" not in existing_columns:
            await db.execute(
                "ALTER TABLE workflow_state ADD COLUMN pipeline_name TEXT NOT NULL DEFAULT ''"
            )
            await db.execute(
                "UPDATE workflow_state SET pipeline_name = pipeline_id WHERE pipeline_name = ''"
            )

        # Update any NULL values in required columns
        await db.execute(
            "UPDATE workflow_state SET current_step_index = 0 WHERE current_step_index IS NULL"
        )
        await db.execute(
            "UPDATE workflow_state SET pipeline_context = '{}' WHERE pipeline_context IS NULL"
        )
        await db.execute("UPDATE workflow_state SET status = 'running' WHERE status IS NULL")

        # Migrate runs table schema if it exists
        try:
            cursor = await db.execute("PRAGMA table_info(runs)")
            runs_columns = {row[1] for row in await cursor.fetchall()}
            await cursor.close()

            # Add missing columns to runs table
            runs_new_columns = [
                ("pipeline_id", "TEXT NOT NULL DEFAULT 'unknown'"),
                ("created_at", "TEXT NOT NULL DEFAULT ''"),
                ("updated_at", "TEXT NOT NULL DEFAULT ''"),
                ("end_time", "TEXT"),
                ("total_cost", "REAL"),
                ("final_context_blob", "TEXT"),
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
                        self._initialized = True
                    except sqlite3.DatabaseError as e:
                        telemetry.logfire.error(f"Failed to initialize DB: {e}")
                        raise

    async def close(self) -> None:
        """Close database connections and cleanup resources."""
        if self._connection_pool:
            await self._connection_pool.close()
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
                    await db.execute("PRAGMA journal_mode=DELETE;")
                    # Get execution_time_ms directly from state
                    execution_time_ms = state.get("execution_time_ms")
                    pipeline_context_json = json.dumps(robust_serialize(state["pipeline_context"]))
                    last_step_output_json = (
                        json.dumps(robust_serialize(state["last_step_output"]))
                        if state.get("last_step_output") is not None
                        else None
                    )
                    step_history_json = (
                        json.dumps(robust_serialize(state["step_history"]))
                        if state.get("step_history") is not None
                        else None
                    )
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
                    await db.execute("VACUUM;")
                    telemetry.logfire.info(f"Saved state for run_id={run_id}")

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
        import sqlite3
        import sys

        # Use synchronous sqlite3 for diagnostics
        conn = sqlite3.connect(str(self.db_path))
        try:
            # Print all rows in workflow_state
            all_rows = conn.execute("SELECT * FROM workflow_state").fetchall()
            print(
                f"[SYNC DEBUG] ALL workflow_state rows in list_workflows: {all_rows}",
                file=sys.stderr,
            )
            # Build query with optional filters
            query = """
                SELECT run_id, pipeline_id, pipeline_name, pipeline_version,
                       current_step_index, status, created_at, updated_at,
                       total_steps, error_message, execution_time_ms, memory_usage_mb
                FROM workflow_state
                WHERE 1=1
            """
            params = []
            if status:
                query += " AND status = ?"
                params.append(status)
            if pipeline_id:
                query += " AND pipeline_id = ?"
                params.append(pipeline_id)
            query += " ORDER BY created_at DESC"
            if limit is not None:
                query += " LIMIT ?"
                params.append(str(limit))
            if offset:
                query += " OFFSET ?"
                params.append(str(offset))
            rows = conn.execute(query, params).fetchall()
            print(f"[SYNC DEBUG] list_workflows raw rows: {rows}", file=sys.stderr)
            result = []
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
            print(f"[SYNC DEBUG] list_workflows result: {result}", file=sys.stderr)
            return result
        finally:
            conn.close()

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
                        SELECT run_id, pipeline_name, pipeline_version, status, start_time, end_time, total_cost
                        FROM runs
                        WHERE status = ?
                        ORDER BY start_time DESC
                    """
                    params = [status]
                elif pipeline_name and not status:
                    # Use pipeline_name index
                    query = """
                        SELECT run_id, pipeline_name, pipeline_version, status, start_time, end_time, total_cost
                        FROM runs
                        WHERE pipeline_name = ?
                        ORDER BY start_time DESC
                    """
                    params = [pipeline_name]
                elif status and pipeline_name:
                    # Use both filters
                    query = """
                        SELECT run_id, pipeline_name, pipeline_version, status, start_time, end_time, total_cost
                        FROM runs
                        WHERE status = ? AND pipeline_name = ?
                        ORDER BY start_time DESC
                    """
                    params = [status, pipeline_name]
                else:
                    # No filters, use start_time index
                    query = """
                        SELECT run_id, pipeline_name, pipeline_version, status, start_time, end_time, total_cost
                        FROM runs
                        ORDER BY start_time DESC
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
                            "start_time": row[4],
                            "end_time": row[5],
                            "total_cost": row[6],
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
                cursor = await db.execute("""
                    SELECT status, COUNT(*)
                    FROM workflow_state
                    GROUP BY status
                """)
                status_counts_rows = await cursor.fetchall()
                status_counts: Dict[str, int] = {
                    row[0]: row[1] for row in status_counts_rows if row is not None
                }
                await cursor.close()

                # Get recent workflows (last 24 hours)
                cursor = await db.execute("""
                    SELECT COUNT(*)
                    FROM workflow_state
                    WHERE created_at >= datetime('now', '-24 hours')
                """)
                recent_workflows_24h_row = await cursor.fetchone()
                recent_workflows_24h = (
                    recent_workflows_24h_row[0] if recent_workflows_24h_row else 0
                )
                await cursor.close()

                # Get average execution time
                cursor = await db.execute("""
                    SELECT AVG(execution_time_ms)
                    FROM workflow_state
                    WHERE execution_time_ms IS NOT NULL
                """)
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
                async with aiosqlite.connect(self.db_path) as db:
                    base_timestamp = (
                        run_data.get("created_at")
                        or run_data.get("start_time")
                        or datetime.utcnow().isoformat()
                    )
                    created_at = base_timestamp
                    updated_at = run_data.get("updated_at") or base_timestamp
                    start_time = run_data.get("start_time") or created_at
                    end_time = run_data.get("end_time")
                    total_cost = run_data.get("total_cost")
                    final_context_blob = run_data.get("final_context_blob")
                    await db.execute(
                        """
                        INSERT OR REPLACE INTO runs (
                            run_id, pipeline_id, pipeline_name, pipeline_version, status, created_at, updated_at, start_time, end_time, total_cost, final_context_blob
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
                            start_time,
                            end_time,
                            total_cost,
                            final_context_blob,
                        ),
                    )
                    await db.commit()

            await self._with_retries(_save)

    async def save_step_result(self, step_data: Dict[str, Any]) -> None:
        await self._ensure_init()
        async with self._lock:

            async def _save() -> None:
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute(
                        """
                        INSERT OR REPLACE INTO steps (
                            step_run_id, run_id, step_name, step_index, status,
                            start_time, end_time, duration_ms, cost, tokens,
                            input_blob, output_blob, error_blob
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            step_data["step_run_id"],
                            step_data["run_id"],
                            step_data["step_name"],
                            step_data["step_index"],
                            step_data.get("status", "completed"),
                            (
                                (
                                    lambda v: v.isoformat()
                                    if isinstance(v, datetime)
                                    else (str(v) if v is not None else None)
                                )(step_data.get("start_time"))
                            ),
                            (
                                (
                                    lambda v: v.isoformat()
                                    if isinstance(v, datetime)
                                    else (str(v) if v is not None else None)
                                )(step_data.get("end_time"))
                            ),
                            step_data.get("duration_ms"),
                            step_data.get("cost"),
                            step_data.get("tokens"),
                            json.dumps(robust_serialize(step_data.get("input"))),
                            json.dumps(robust_serialize(step_data.get("output"))),
                            json.dumps(robust_serialize(step_data.get("error")))
                            if step_data.get("error") is not None
                            else None,
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
                        SET status = ?, end_time = ?, total_cost = ?, final_context_blob = ?
                        WHERE run_id = ?
                        """,
                        (
                            end_data.get("status", "completed"),
                            end_data.get("end_time", datetime.utcnow()).isoformat(),
                            end_data.get("total_cost"),
                            json.dumps(robust_serialize(end_data.get("final_context"))),
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
                    "SELECT run_id, pipeline_name, pipeline_version, status, start_time, end_time, total_cost, final_context_blob FROM runs WHERE run_id = ?",
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
                    "start_time": row[4],
                    "end_time": row[5],
                    "total_cost": row[6],
                    "final_context": safe_deserialize(json.loads(row[7])) if row[7] else None,
                }

    async def list_run_steps(self, run_id: str) -> List[Dict[str, Any]]:
        await self._ensure_init()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    """
                    SELECT step_name, step_index, status, start_time, end_time, duration_ms,
                           cost, tokens, input_blob, output_blob, error_blob
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
                            "start_time": r[3],
                            "end_time": r[4],
                            "duration_ms": r[5],
                            "cost": r[6],
                            "tokens": r[7],
                            "input": safe_deserialize(json.loads(r[8])) if r[8] else None,
                            "output": safe_deserialize(json.loads(r[9])) if r[9] else None,
                            "error": safe_deserialize(json.loads(r[10])) if r[10] else None,
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
            span_data: Dict[str, Any], parent_span_id: Optional[str] = None, depth: int = 0
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

            span_tuple: Tuple[str, str, Optional[str], str, float, Optional[float], str, str] = (
                str(span_data.get("span_id", "")),
                run_id,
                parent_span_id,
                str(span_data.get("name", "")),
                float(span_data.get("start_time", 0.0)),
                float(span_data["end_time"]) if span_data.get("end_time") is not None else None,
                str(span_data.get("status", "running")),
                json.dumps(robust_serialize(span_data.get("attributes", {}))),
            )
            spans.append(span_tuple)

            # Process children recursively
            for child in span_data.get("children", []):
                extract_span_recursive(child, span_data.get("span_id"), depth + 1)

        extract_span_recursive(trace)
        return spans

    def _reconstruct_trace_tree(
        self, spans_data: List[Tuple[str, Optional[str], str, float, Optional[float], str, str]]
    ) -> Optional[Dict[str, Any]]:
        """Reconstruct a hierarchical trace tree from flat spans data."""
        spans_map: Dict[str, Dict[str, Any]] = {}
        root_spans: List[Dict[str, Any]] = []

        # First pass: create a map of all spans by ID
        for row in spans_data:
            span_id, parent_span_id, name, start_time, end_time, status, attributes = row
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
                        span_id, parent_span_id, name, start_time, end_time, status, attributes = r
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
        self, pipeline_name: Optional[str] = None, time_range: Optional[Tuple[float, float]] = None
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
                            stats["avg_duration_by_name"][name] = {"total": 0.0, "count": 0}
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
