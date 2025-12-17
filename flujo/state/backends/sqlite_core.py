"""SQLite-backed persistent storage for workflow state with optimized schema."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import time
import threading
import weakref
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
    TYPE_CHECKING,
)

import aiosqlite
import os

from flujo.exceptions import InfiniteRedirectError, PausedException, PipelineAbortSignal
from .base import StateBackend, _serialize_for_json as _serialize_for_json_base
from flujo.infra import telemetry
from flujo.type_definitions.common import JSONObject
from flujo.utils.async_bridge import run_sync

logger = logging.getLogger(__name__)


# Serialize objects for persistence using the shared normalization logic.
def _serialize_for_json(obj: Any) -> Any:
    return _serialize_for_json_base(obj)


# Try to import orjson for faster JSON serialization
try:
    import orjson

    def _fast_json_dumps(obj: Any) -> str:
        """Use orjson for faster JSON serialization with typed model support."""
        blob: bytes = orjson.dumps(_serialize_for_json(obj))
        return blob.decode("utf-8")

except ImportError:

    def _fast_json_dumps(obj: Any) -> str:
        """Fallback to standard json for JSON serialization with typed model support."""
        s: str = json.dumps(_serialize_for_json(obj), separators=(",", ":"))
        return s


if TYPE_CHECKING:
    from asyncio import AbstractEventLoop


T = TypeVar("T")


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
    "metadata": "TEXT",
    "is_background_task": "INTEGER DEFAULT 0",
    "parent_run_id": "TEXT",
    "task_id": "TEXT",
    "background_error": "TEXT",
}

# Pre-compute allowed column definitions for fast membership checks
ALLOWED_COLUMN_DEFS = {f"{k} {v}" for k, v in ALLOWED_COLUMNS.items()}

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


def validate_column_definition_or_raise(column_def: str) -> None:
    """Validate and raise on invalid column definitions."""
    if not _validate_column_definition(column_def):
        raise ValueError(f"Invalid column definition: {column_def}")


class SQLiteBackendBase(StateBackend):
    """SQLite-backed persistent storage for workflow state with optimized schema."""

    _global_file_locks: "weakref.WeakKeyDictionary[AbstractEventLoop, Dict[str, asyncio.Lock]]" = (
        weakref.WeakKeyDictionary()
    )
    _thread_file_locks: Dict[int, Dict[str, asyncio.Lock]] = {}
    _instances: "weakref.WeakSet[SQLiteBackendBase]" = weakref.WeakSet()

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure parent directories exist
        # Locks must be loop-local to avoid cross-loop binding errors when the same backend
        # instance is used from multiple event loops (e.g., threads running asyncio.run()).
        # We keep _lock for backward compatibility but populate it with the current loop's
        # lock on demand via _get_lock().
        self._lock: Optional[asyncio.Lock] = None
        self._loop_locks: "weakref.WeakKeyDictionary[AbstractEventLoop, asyncio.Lock]" = (
            weakref.WeakKeyDictionary()
        )
        self._thread_locks: Dict[int, asyncio.Lock] = {}
        self._initialized = False
        # Backward-compat attribute (tests expect None when no pool). Actual pools are loop-local.
        self._connection_pool: Optional[aiosqlite.Connection] = None
        # Per-event-loop pooled connections to avoid cross-loop reuse issues in async apps.
        # Guarded by self._lock for serialized access.
        self._connection_pool_map: Dict[int, aiosqlite.Connection] = {}

        # Event-loop-local file-level lock - will be initialized lazily
        self._file_lock: Optional[asyncio.Lock] = None
        self._file_lock_key = str(self.db_path.absolute())

        # Register instance for process-exit cleanup to prevent lingering threads
        try:
            SQLiteBackendBase._instances.add(self)
        except Exception:
            pass

    def _get_lock(self) -> asyncio.Lock:
        """Return an event-loop-local lock to avoid cross-loop binding errors.

        asyncio.Lock instances are bound to the loop that creates them. When the same
        SQLiteBackend is used from different threads/loops (e.g., concurrent asyncio.run
        calls), sharing a single lock causes RuntimeError: Task attached to a different
        loop. We therefore allocate one lock per loop (and per thread when no loop is
        running) and reuse it on subsequent calls from that context.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            thread_id = threading.get_ident()
            lock = self._thread_locks.get(thread_id)
            if lock is None:
                lock = asyncio.Lock()
                self._thread_locks[thread_id] = lock
            self._lock = lock
            return lock

        lock = self._loop_locks.get(loop)
        if lock is None:
            lock = asyncio.Lock()
            self._loop_locks[loop] = lock
        self._lock = lock
        return lock

    def _current_loop_id(self) -> Optional[int]:
        try:
            return id(asyncio.get_running_loop())
        except RuntimeError:
            return None

    def _get_pooled_connection(self) -> Optional[aiosqlite.Connection]:
        loop_id = self._current_loop_id()
        if loop_id is None:
            return None
        return self._connection_pool_map.get(loop_id)

    def _set_pooled_connection(self, conn: aiosqlite.Connection) -> None:
        loop_id = self._current_loop_id()
        if loop_id is None:
            return
        self._connection_pool_map[loop_id] = conn

    async def _acquire_connection(self) -> tuple[aiosqlite.Connection, bool]:
        """Return (connection, close_after) respecting loop-local pooling."""
        pooled = self._get_pooled_connection()
        if pooled is not None:
            return pooled, False
        conn = await self._create_connection()
        return conn, True

    async def _create_connection(self) -> aiosqlite.Connection:
        """Return a new aiosqlite connection with daemonized worker thread."""
        conn = aiosqlite.connect(self.db_path)
        try:
            # aiosqlite.Connection is awaitable and may not expose Thread attrs in stubs;
            # use setattr to avoid mypy attr-defined errors while keeping best-effort behavior.
            setattr(conn, "daemon", True)
            setattr(conn, "name", f"flujo-sqlite-{id(self)}")
        except AttributeError:
            pass
        return await conn

    async def shutdown(self) -> None:
        """Close connection pool and release resources to avoid lingering threads."""
        lock = self._get_lock()
        acquired = False
        try:
            try:
                await asyncio.wait_for(lock.acquire(), timeout=2.0)
                acquired = True
            except TimeoutError as exc:
                logger.debug("Timed out acquiring shutdown lock: %s", exc)

            conns = list(self._connection_pool_map.values())
            self._connection_pool_map.clear()
        except Exception as exc:  # noqa: BLE001 - defensive shutdown
            logger.debug("Non-fatal shutdown error: %s", exc)
            conns = []
        finally:
            if acquired:
                try:
                    lock.release()
                except Exception:
                    pass

        # Close pooled async connections if present (best-effort; never hang).
        for conn in conns:
            try:
                await asyncio.wait_for(conn.close(), timeout=2.0)
            except TimeoutError as exc:
                logger.debug("Timed out closing pooled connection during shutdown: %s", exc)
            except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                logger.debug("Non-fatal error closing pooled connection during shutdown: %s", exc)

    @classmethod
    def shutdown_all(cls) -> None:
        """Best-effort synchronous shutdown of all live SQLiteBackend instances.

        This prevents pytest processes from lingering due to aiosqlite worker
        threads when connections remain open at interpreter shutdown.
        """
        try:
            instances = list(cls._instances)
        except Exception:
            instances = []
        for inst in instances:
            try:
                run_sync(inst.shutdown())
            except Exception:
                # Best-effort cleanup; ignore shutdown errors
                pass

    def _get_file_lock(self) -> asyncio.Lock:
        """Get the file lock for the current event loop with robust fallback handling."""
        if self._file_lock is None:
            try:
                # Try to get the current event loop
                loop = asyncio.get_running_loop()
                # We have a valid event loop
                if loop not in SQLiteBackendBase._global_file_locks:
                    SQLiteBackendBase._global_file_locks[loop] = {}
                lock_map = SQLiteBackendBase._global_file_locks[loop]
                db_key = str(self.db_path.absolute())
                if db_key not in lock_map:
                    lock_map[db_key] = asyncio.Lock()
                self._file_lock = lock_map[db_key]

            except RuntimeError:
                # No running event loop - this is the CLI context
                # Create and store a per-thread, per-db asyncio.Lock for deduplication
                import threading

                # Create a thread-local asyncio.Lock that's safe for CLI context
                thread_id = threading.get_ident()
                if thread_id not in SQLiteBackendBase._thread_file_locks:
                    SQLiteBackendBase._thread_file_locks[thread_id] = {}
                thread_lock_map = SQLiteBackendBase._thread_file_locks[thread_id]
                db_key = str(self.db_path.absolute())
                if db_key not in thread_lock_map:
                    # Create and store an asyncio.Lock for this db in this thread
                    thread_lock_map[db_key] = asyncio.Lock()

                # Use the stored asyncio.Lock for deduplication
                self._file_lock = thread_lock_map[db_key]

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
                                conn = await self._create_connection()
                                try:
                                    await conn.execute("SELECT 1")
                                finally:
                                    await conn.close()
                        except (OSError, TypeError) as e:
                            telemetry.logfire.warning(f"File stat failed, assuming corrupted: {e}")
                            await self._backup_corrupted_database()
                    except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
                        telemetry.logfire.warning(
                            f"Database appears to be corrupted, creating backup: {e}"
                        )
                        await self._backup_corrupted_database()

            conn = await self._create_connection()
            try:
                db = conn
                # OPTIMIZATION: Use more efficient SQLite settings for performance
                await db.execute("PRAGMA journal_mode = WAL")
                await db.execute("PRAGMA synchronous = NORMAL")
                await db.execute("PRAGMA cache_size = 10000")  # Increase cache size
                await db.execute("PRAGMA temp_store = MEMORY")  # Use memory for temp tables
                await db.execute("PRAGMA mmap_size = 268435456")  # 256MB memory mapping
                await db.execute("PRAGMA page_size = 4096")
                await db.execute("PRAGMA busy_timeout = 1000")

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
                        memory_usage_mb REAL,
                        metadata TEXT,
                        is_background_task INTEGER DEFAULT 0,
                        parent_run_id TEXT,
                        task_id TEXT,
                        background_error TEXT
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
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS system_state (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
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

                # Create evaluations table for shadow eval persistence
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS evaluations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        step_name TEXT,
                        score REAL NOT NULL,
                        feedback TEXT,
                        metadata TEXT,
                        created_at TEXT NOT NULL DEFAULT (DATETIME('now')),
                        FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                    )
                    """
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_evaluations_run_id ON evaluations(run_id)"
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_evaluations_created_at ON evaluations(created_at)"
                )

                await db.execute("COMMIT")

                # Apply versioned migrations (PRAGMA user_version)
                from .sqlite_migrations import apply_sqlite_migrations

                await apply_sqlite_migrations(db)

                # Create indexes after migration to ensure columns exist
                await self._create_indexes(db)

                await db.commit()
            finally:
                await conn.close()
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
            # Try to move the corrupted file to backup location using Path.rename (offloaded)
            await asyncio.to_thread(self.db_path.rename, backup_path)
            telemetry.logfire.warning(f"Corrupted database backed up to {backup_path}")
        except (OSError, IOError):
            # If move fails, try to copy and then remove (offloaded to avoid blocking loop)
            try:
                import shutil

                def _copy_and_remove() -> None:
                    shutil.copy2(str(self.db_path), str(backup_path))
                    wal = self.db_path.with_suffix(self.db_path.suffix + "-wal")
                    shm = self.db_path.with_suffix(self.db_path.suffix + "-shm")
                    if wal.exists():
                        shutil.copy2(str(wal), str(backup_path) + "-wal")
                    if shm.exists():
                        shutil.copy2(str(shm), str(backup_path) + "-shm")
                    self.db_path.unlink(missing_ok=True)
                    try:
                        wal.unlink(missing_ok=True)
                        shm.unlink(missing_ok=True)
                    except Exception as exc:
                        logger.debug("Could not remove WAL/SHM sidecars: %s", exc)

                await asyncio.to_thread(_copy_and_remove)
                telemetry.logfire.warning(f"Corrupted database copied to {backup_path} and removed")
            except (OSError, IOError) as copy_error:
                # If all else fails, just remove the corrupted file
                try:
                    await asyncio.to_thread(self.db_path.unlink)
                    telemetry.logfire.warning(f"Corrupted database removed: {copy_error}")
                except (OSError, IOError) as remove_error:
                    telemetry.logfire.error(f"Failed to remove corrupted database: {remove_error}")
                    # If all backup attempts fail, raise a DatabaseError
                    raise sqlite3.DatabaseError(
                        "Database corruption recovery failed"
                    ) from remove_error

    async def _create_indexes(self, db: aiosqlite.Connection) -> None:
        """Create indexes for better query performance."""
        try:
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_workflow_state_status ON workflow_state(status)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_workflow_state_pipeline_id ON workflow_state(pipeline_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_workflow_state_parent_run_id "
                "ON workflow_state(parent_run_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_workflow_state_created_at ON workflow_state(created_at)"
            )
        except sqlite3.OperationalError:
            # If the table doesn't exist yet, skip index creation.
            pass

        try:
            await db.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_runs_pipeline_id ON runs(pipeline_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at)")
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_runs_pipeline_name ON runs(pipeline_name)"
            )
        except sqlite3.OperationalError:
            # Table doesn't exist yet, which is fine - it will be created with the new schema
            pass

    async def _ensure_init(self) -> None:
        if not self._initialized:
            # Use file-level lock to prevent concurrent initialization across instances
            # Ensure the instance lock exists for callers that access self._lock directly
            self._get_lock()
            async with self._get_file_lock():
                if not self._initialized:
                    try:
                        await self._init_db()
                        # Lazily create a pooled connection with optimized pragmas for subsequent writes
                        try:
                            conn = await self._create_connection()
                            try:
                                await conn.execute("PRAGMA journal_mode = WAL")
                                await conn.execute("PRAGMA synchronous = NORMAL")
                                await conn.execute("PRAGMA temp_store = MEMORY")
                                await conn.execute("PRAGMA cache_size = 10000")
                                await conn.execute("PRAGMA mmap_size = 268435456")
                                await conn.execute("PRAGMA page_size = 4096")
                                await conn.execute("PRAGMA busy_timeout = 1000")
                                await conn.commit()
                                lock = self._get_lock()
                                async with lock:
                                    self._set_pooled_connection(conn)
                            except (sqlite3.Error, aiosqlite.Error):
                                try:
                                    await conn.close()
                                except Exception as close_exc:  # noqa: BLE001 - best-effort cleanup
                                    logger.debug(
                                        "Non-fatal error closing pooled connection: %s", close_exc
                                    )
                                # Fall back to per-call connections
                            except (PausedException, PipelineAbortSignal, InfiniteRedirectError):
                                try:
                                    await conn.close()
                                except Exception as close_exc:  # noqa: BLE001 - best-effort cleanup
                                    logger.debug(
                                        "Non-fatal error closing pooled connection: %s", close_exc
                                    )
                                raise
                            except Exception:
                                try:
                                    await conn.close()
                                except Exception as close_exc:  # noqa: BLE001 - best-effort cleanup
                                    logger.debug(
                                        "Non-fatal error closing pooled connection: %s", close_exc
                                    )
                                raise
                        except (PausedException, PipelineAbortSignal, InfiniteRedirectError):
                            raise
                        self._initialized = True
                    except sqlite3.DatabaseError as e:
                        telemetry.logfire.error(f"Failed to initialize DB: {e}")
                        raise

    async def close(self) -> None:
        """Close database connections and cleanup resources."""
        lock = self._get_lock()
        async with lock:
            # Close all loop-local pooled connections
            for conn in list(self._connection_pool_map.values()):
                try:
                    await conn.close()
                except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                    logger.debug("Non-fatal error closing pooled connection: %s", exc)
            self._connection_pool_map.clear()
            self._connection_pool = None
            self._initialized = False

        # No global locks to clean up

    def close_sync(self) -> None:
        """Synchronous version of close() that works from sync contexts.

        This method properly handles event loop contexts and prevents pytest teardown
        hangs when closing backends from synchronous test code. It uses the
        shared async bridge to run the async close() in the correct context.

        Use this method instead of asyncio.run(backend.close()) in sync tests.
        """
        run_sync(self.close())

    async def __aenter__(self) -> "SQLiteBackendBase":
        """Async context manager entry with eager lock/init."""
        # Ensure the instance lock exists before tests use backend._lock directly.
        self._get_lock()
        await self._ensure_init()
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

    async def persist_evaluation(
        self,
        run_id: str,
        score: float,
        feedback: str | None = None,
        step_name: str | None = None,
        metadata: JSONObject | None = None,
    ) -> None:
        """Persist a shadow evaluation result into SQLite."""

        async def _insert() -> None:
            conn, close_after = await self._acquire_connection()
            try:
                metadata_json = (
                    None if metadata is None else json.dumps(_serialize_for_json(metadata))
                )
                await conn.execute(
                    """
                    INSERT INTO evaluations (run_id, step_name, score, feedback, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, datetime('now'))
                    """,
                    (
                        run_id,
                        step_name,
                        score,
                        feedback,
                        metadata_json,
                    ),
                )
                await conn.commit()
            finally:
                if close_after:
                    await conn.close()

        await self._ensure_init()
        await self._with_retries(_insert)

    async def list_evaluations(
        self,
        limit: int = 20,
        run_id: str | None = None,
    ) -> list[JSONObject]:
        """Return recent evaluations."""

        async def _select() -> list[JSONObject]:
            conn, close_after = await self._acquire_connection()
            try:
                if run_id:
                    cursor = await conn.execute(
                        """
                        SELECT run_id, step_name, score, feedback, metadata, created_at
                        FROM evaluations
                        WHERE run_id = ?
                        ORDER BY datetime(created_at) DESC
                        LIMIT ?
                        """,
                        (run_id, limit),
                    )
                else:
                    cursor = await conn.execute(
                        """
                        SELECT run_id, step_name, score, feedback, metadata, created_at
                        FROM evaluations
                        ORDER BY datetime(created_at) DESC
                        LIMIT ?
                        """,
                        (limit,),
                    )
                rows = await cursor.fetchall()
                await cursor.close()
                records: list[JSONObject] = []
                for row in rows:
                    records.append(
                        {
                            "run_id": row[0],
                            "step_name": row[1],
                            "score": row[2],
                            "feedback": row[3],
                            "metadata": json.loads(row[4]) if row[4] else None,
                            "created_at": row[5],
                        }
                    )
                return records
            finally:
                if close_after:
                    await conn.close()

        await self._ensure_init()
        result = await self._with_retries(_select)
        return list(result) if isinstance(result, list) else []
