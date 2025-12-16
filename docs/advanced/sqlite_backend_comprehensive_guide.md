# SQLiteBackend Comprehensive Guide

## Overview

The `SQLiteBackend` is a production-ready state backend for Flujo that provides durable workflow persistence with advanced observability features. It's designed for high-performance, concurrent access patterns and includes comprehensive admin query capabilities.

## Key Features

- **Durable Persistence**: Workflows survive process restarts and system failures
- **Concurrent Access**: Thread-safe operations with proper locking
- **Admin Queries**: Built-in observability and monitoring capabilities
- **Automatic Migration**: Seamless schema upgrades for existing deployments
- **Performance Optimized**: Indexed queries and WAL mode for better concurrency
- **Enhanced Serialization**: Support for custom Python types and complex data structures

## Basic Usage

### Initialization

```python
from flujo.state.backends.sqlite import SQLiteBackend
from pathlib import Path

# Initialize with a database file path
backend = SQLiteBackend(Path("workflow_state.db"))

# The database will be automatically created and initialized on first use
```

### Core Operations

```python
from datetime import datetime, timezone

# Save workflow state
now = datetime.now(timezone.utc).replace(microsecond=0)
state = {
    "run_id": "run_123",
    "pipeline_id": "data_processing",
    "pipeline_name": "Data Processing Pipeline",
    "pipeline_version": "1.0.0",
    "current_step_index": 2,
    "pipeline_context": {"input_data": "sample.csv", "processed_rows": 1000},
    "last_step_output": {"status": "success", "rows_processed": 1000},
    "status": "running",
    "created_at": now,
    "updated_at": now,
    "total_steps": 5,
    "error_message": None,
    "execution_time_ms": 1500,
    "memory_usage_mb": 45.2
}

await backend.save_state("run_123", state)

# Load workflow state
loaded_state = await backend.load_state("run_123")
if loaded_state:
    print(f"Pipeline: {loaded_state['pipeline_name']}")
    print(f"Status: {loaded_state['status']}")
    print(f"Step: {loaded_state['current_step_index']}/{loaded_state['total_steps']}")

# Delete workflow state
await backend.delete_state("run_123")
```

## Admin Queries and Observability

### Listing Workflows

```python
# List all workflows
all_workflows = await backend.list_workflows()

# List with pagination
recent_workflows = await backend.list_workflows(limit=10, offset=0)

# Filter by status
running_workflows = await backend.list_workflows(status="running")
failed_workflows = await backend.list_workflows(status="failed")
completed_workflows = await backend.list_workflows(status="completed")

# Filter by pipeline
pipeline_workflows = await backend.list_workflows(pipeline_id="data_processing")

# Combined filtering
filtered_workflows = await backend.list_workflows(
    status="failed",
    pipeline_id="data_processing",
    limit=5
)
```

### Edge Cases for Listing

```python
# Empty database returns empty list
empty_workflows = await backend.list_workflows()
assert empty_workflows == []

# Non-existent status returns empty list
nonexistent_status = await backend.list_workflows(status="nonexistent")
assert nonexistent_status == []

# Non-existent pipeline_id returns empty list
nonexistent_pipeline = await backend.list_workflows(pipeline_id="nonexistent_pipeline")
assert nonexistent_pipeline == []

# Empty string pipeline_id returns all workflows (no filter applied)
all_with_empty_filter = await backend.list_workflows(pipeline_id="")
assert len(all_with_empty_filter) == len(await backend.list_workflows())
```

### Workflow Statistics

```python
# Get comprehensive statistics
stats = await backend.get_workflow_stats()

print(f"Total workflows: {stats['total_workflows']}")
print(f"Recent workflows (24h): {stats['recent_workflows_24h']}")
print(f"Status breakdown: {stats['status_counts']}")
print(f"Average execution time: {stats['average_execution_time_ms']}ms")

# Example output:
# Total workflows: 1250
# Recent workflows (24h): 45
# Status breakdown: {'completed': 1000, 'running': 15, 'failed': 35, 'cancelled': 5}
# Average execution time: 2340.5ms
```

### Edge Cases for Statistics

```python
# Empty database returns zero-filled stats
empty_stats = await backend.get_workflow_stats()
assert empty_stats["total_workflows"] == 0
assert empty_stats["status_counts"] == {}
assert empty_stats["recent_workflows_24h"] == 0
assert empty_stats["average_execution_time_ms"] == 0
```

### Failed Workflow Analysis

```python
# Get failed workflows from last 24 hours
failed_workflows = await backend.get_failed_workflows(hours_back=24)

for workflow in failed_workflows:
    print(f"Run ID: {workflow['run_id']}")
    print(f"Pipeline: {workflow['pipeline_name']}")
    print(f"Error: {workflow['error_message']}")
    print(f"Failed at: {workflow['updated_at']}")
    print("---")

# Get failed workflows from last week
recent_failures = await backend.get_failed_workflows(hours_back=168)

# Get failed workflows from last hour
recent_failures = await backend.get_failed_workflows(hours_back=1)
```

### Edge Cases for Failed Workflows

```python
# Empty database returns empty list
empty_failed = await backend.get_failed_workflows(hours_back=24)
assert empty_failed == []

# Different time ranges on empty database
for hours in [1, 6, 12, 24, 48, 168]:
    failed = await backend.get_failed_workflows(hours_back=hours)
    assert failed == []
```

### Cleanup Operations

```python
# Delete workflows older than 30 days
deleted_count = await backend.cleanup_old_workflows(days_old=30)
print(f"Deleted {deleted_count} old workflows")

# Delete workflows older than 7 days
weekly_cleanup = await backend.cleanup_old_workflows(days_old=7)

# Delete workflows older than 1 day
daily_cleanup = await backend.cleanup_old_workflows(days_old=1)
```

### Edge Cases for Cleanup

```python
# Empty database returns 0 deleted
deleted_empty = await backend.cleanup_old_workflows(days_old=1)
assert deleted_empty == 0

# Recent workflows are not deleted
now = datetime.now(timezone.utc).replace(microsecond=0)
recent_state = {
    "run_id": "recent_run",
    "pipeline_id": "test_pipeline",
    "pipeline_name": "Test Pipeline",
    "pipeline_version": "1.0",
    "current_step_index": 0,
    "pipeline_context": {"test": "data"},
    "last_step_output": "test_output",
    "status": "completed",
    "created_at": now,
    "updated_at": now,
    "total_steps": 5,
    "error_message": None,
    "execution_time_ms": 1000,
    "memory_usage_mb": 10.0,
}

await backend.save_state("recent_run", recent_state)

# Should not delete recent workflows
deleted_count = await backend.cleanup_old_workflows(days_old=1)
assert deleted_count == 0

# Verify workflow still exists
all_workflows = await backend.list_workflows()
assert len(all_workflows) == 1
assert all_workflows[0]["run_id"] == "recent_run"
```

## Advanced Features

### Custom Type Serialization

The SQLiteBackend supports custom Python types through enhanced serialization:

```python
from pydantic import BaseModel
from flujo.utils.serialization import register_custom_serializer

class CustomData(BaseModel):
    value: str
    metadata: dict

def serialize_custom_data(obj: CustomData) -> dict:
    return {"value": obj.value, "metadata": obj.metadata}

register_custom_serializer(CustomData, serialize_custom_data)

# Custom types are automatically handled
state = {
    "run_id": "custom_run",
    "pipeline_id": "test_pipeline",
    "pipeline_name": "Test Pipeline",
    "pipeline_version": "1.0",
    "current_step_index": 0,
    "pipeline_context": {"custom": CustomData(value="test", metadata={"key": "value"})},
    "last_step_output": CustomData(value="output", metadata={"status": "success"}),
    "status": "completed",
    "created_at": datetime.now(timezone.utc),
    "updated_at": datetime.now(timezone.utc),
    "total_steps": 5,
    "error_message": None,
    "execution_time_ms": 1000,
    "memory_usage_mb": 10.0,
}

await backend.save_state("custom_run", state)
loaded_state = await backend.load_state("custom_run")
# Custom types are properly deserialized
```

### Concurrent Access

The backend is designed for concurrent access patterns:

```python
import asyncio

async def concurrent_worker(backend: SQLiteBackend, worker_id: int):
    """Simulate concurrent workflow operations."""
    for i in range(10):
        state = {
            "run_id": f"worker_{worker_id}_run_{i}",
            "pipeline_id": f"pipeline_{worker_id}",
            "pipeline_name": f"Pipeline {worker_id}",
            "pipeline_version": "1.0",
            "current_step_index": i,
            "pipeline_context": {"worker": worker_id, "iteration": i},
            "last_step_output": f"output_{i}",
            "status": "completed",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "total_steps": 10,
            "error_message": None,
            "execution_time_ms": 1000 * i,
            "memory_usage_mb": 10.0 * i,
        }
        await backend.save_state(f"worker_{worker_id}_run_{i}", state)
        loaded = await backend.load_state(f"worker_{worker_id}_run_{i}")
        assert loaded is not None

# Run multiple workers concurrently
workers = [concurrent_worker(backend, i) for i in range(5)]
await asyncio.gather(*workers)

# Verify all workflows are present
all_workflows = await backend.list_workflows()
assert len(all_workflows) == 50  # 5 workers * 10 workflows each
```

## Performance Considerations

### Large Dataset Handling

The backend is optimized for large datasets:

```python
# Test with thousands of workflows
num_workflows = 5000
for i in range(num_workflows):
    state = {
        "run_id": f"large_run_{i}",
        "pipeline_id": f"pipeline_{i % 10}",
        "pipeline_name": f"Pipeline {i % 10}",
        "pipeline_version": "1.0",
        "current_step_index": i % 5,
        "pipeline_context": {"index": i},
        "last_step_output": f"output_{i}",
        "status": "completed" if i % 2 == 0 else "failed",
        "created_at": datetime.now(timezone.utc) - timedelta(minutes=i),
        "updated_at": datetime.now(timezone.utc) - timedelta(minutes=i),
        "total_steps": 5,
        "error_message": None,
        "execution_time_ms": 1000 * (i % 10),
        "memory_usage_mb": 10.0 * (i % 10),
    }
    await backend.save_state(f"large_run_{i}", state)

# Queries remain performant
import time
start_time = time.time()
all_workflows = await backend.list_workflows()
query_time = time.time() - start_time
print(f"Query time for {len(all_workflows)} workflows: {query_time:.2f}s")
assert query_time < 2.0  # Should be under 2 seconds for 5000 workflows
```

### Pagination Performance

```python
# Test pagination with large datasets
page_size = 25
seen_ids = set()

for offset in range(0, num_workflows, page_size):
    page = await backend.list_workflows(limit=page_size, offset=offset)
    assert len(page) <= page_size
    for workflow in page:
        seen_ids.add(workflow["run_id"])

assert len(seen_ids) == num_workflows
```

## Error Handling and Recovery

### Database Corruption Recovery

```python
# The backend handles corrupted database files gracefully
corrupted_db_path = Path("corrupted.db")
with open(corrupted_db_path, "w") as f:
    f.write("This is not a valid SQLite database")

corrupted_backend = SQLiteBackend(corrupted_db_path)

try:
    await corrupted_backend.save_state("test_run", {"test": "data"})
    # If it succeeds, the database was reinitialized
    loaded = await corrupted_backend.load_state("test_run")
    assert loaded is not None
except Exception as e:
    # If it fails, the error should be clear and informative
    assert "database" in str(e).lower() or "sqlite" in str(e).lower()
```

### Migration Safety

```python
# The backend automatically migrates existing databases
import aiosqlite

# Create an old database schema
old_db_path = Path("old_schema.db")
async with aiosqlite.connect(old_db_path) as conn:
    await conn.execute("""
        CREATE TABLE workflow_state (
            run_id TEXT PRIMARY KEY,
            pipeline_id TEXT,
            pipeline_version TEXT,
            current_step_index INTEGER,
            pipeline_context TEXT,
            last_step_output TEXT,
            status TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    """)
    await conn.commit()

# Add some data to the old schema
async with aiosqlite.connect(old_db_path) as conn:
    await conn.execute("""
        INSERT INTO workflow_state VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "old_run",
        "old_pipeline",
        "0.1",
        1,
        "{}",
        "old_output",
        "completed",
        "2023-01-01T00:00:00",
        "2023-01-01T00:00:00"
    ))
    await conn.commit()

# Create a backend that should migrate the old schema
migrated_backend = SQLiteBackend(old_db_path)

# The migration should succeed and preserve existing data
loaded = await migrated_backend.load_state("old_run")
assert loaded is not None
assert loaded["pipeline_id"] == "old_pipeline"

# Should be able to save new data with the new schema
new_state = {
    "run_id": "new_run",
    "pipeline_id": "new_pipeline",
    "pipeline_name": "New Pipeline",
    "pipeline_version": "1.0",
    "current_step_index": 0,
    "pipeline_context": {"new": "data"},
    "last_step_output": "new_output",
    "status": "running",
    "created_at": datetime.now(timezone.utc),
    "updated_at": datetime.now(timezone.utc),
    "total_steps": 5,
    "error_message": None,
    "execution_time_ms": 1000,
    "memory_usage_mb": 10.0,
}

await migrated_backend.save_state("new_run", new_state)
new_loaded = await migrated_backend.load_state("new_run")
assert new_loaded is not None
assert new_loaded["pipeline_name"] == "New Pipeline"
```

## Security Considerations

### SQL Injection Prevention

The backend uses parameterized queries to prevent SQL injection:

```python
# All user inputs are properly parameterized
malicious_inputs = [
    "'; DROP TABLE workflow_state; --",
    "' OR '1'='1",
    "' UNION SELECT * FROM workflow_state --",
    "'; INSERT INTO workflow_state VALUES ('hacked', 'hacked', 'hacked', 'hacked', 0, '{}', NULL, 'running', datetime('now'), datetime('now'), 0, NULL, NULL, NULL); --",
]

for malicious_input in malicious_inputs:
    try:
        # These should not cause SQL injection
        result = await backend.get_failed_workflows(hours_back=malicious_input)
        assert isinstance(result, list)

        result = await backend.cleanup_old_workflows(days_old=malicious_input)
        assert isinstance(result, int)

        result = await backend.list_workflows(status=malicious_input)
        assert isinstance(result, list)
    except (ValueError, TypeError):
        # Expected for invalid input types
        pass
```

## Best Practices

### 1. Regular Maintenance

```python
async def maintenance_routine():
    """Regular maintenance tasks."""
    # Clean up old workflows
    deleted = await backend.cleanup_old_workflows(days_old=30)
    print(f"Maintenance: deleted {deleted} old workflows")

    # Monitor database health
    stats = await backend.get_workflow_stats()
    print(f"Total workflows: {stats['total_workflows']}")
    print(f"Recent activity: {stats['recent_workflows_24h']} workflows in 24h")
```

### 2. Error Monitoring

```python
async def error_monitor():
    """Monitor for workflow failures."""
    while True:
        try:
            failures = await backend.get_failed_workflows(hours_back=1)
            if failures:
                print(f"⚠️  {len(failures)} failures in the last hour:")
                for wf in failures:
                    print(f"  - {wf['run_id']}: {wf['error_message']}")
        except Exception as e:
            print(f"Monitor error: {e}")

        # Check every 5 minutes
        await asyncio.sleep(5 * 60)
```

### 3. Performance Monitoring

```python
async def performance_monitor():
    """Monitor workflow performance."""
    stats = await backend.get_workflow_stats()

    avg_exec_time = stats['average_execution_time_ms']
    if avg_exec_time > 60000:  # 1 minute
        print(f"⚠️  Slow average execution: {avg_exec_time/1000:.1f}s")

    # Check for stuck workflows
    running = stats['status_counts'].get('running', 0)
    if running > 20:
        print(f"⚠️  Many running workflows: {running}")
```

### 4. Database Backup

```python
import shutil
from datetime import datetime, timezone

async def backup_database():
    """Create a backup of the database."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = f"workflow_state_backup_{timestamp}.db"

    # Simple file copy backup
    shutil.copy2(backend.db_path, backup_path)
    print(f"Backup created: {backup_path}")
```

## Troubleshooting

### Common Issues

1. **Permission Errors**: Ensure the database directory is writable
2. **Disk Space**: Monitor database size and implement cleanup routines
3. **Concurrent Access**: The backend handles concurrent access automatically
4. **Migration Issues**: Check database permissions and disk space

### Debug Queries

```sql
-- Check database schema
PRAGMA table_info(workflow_state);

-- Check index usage
SELECT * FROM sqlite_master WHERE type = 'index' AND tbl_name = 'workflow_state';

-- Check table size
SELECT COUNT(*) as total_rows FROM workflow_state;

-- Check for data integrity issues
SELECT run_id, status FROM workflow_state
WHERE status NOT IN ('running', 'paused', 'completed', 'failed', 'cancelled');

-- Check for stuck workflows
SELECT run_id, pipeline_name, updated_at
FROM workflow_state
WHERE status = 'running'
AND updated_at < datetime('now', '-1 hour');
```

### Performance Tuning

```sql
-- Enable WAL mode for better concurrency
PRAGMA journal_mode = WAL;

-- Check query performance
EXPLAIN QUERY PLAN SELECT * FROM workflow_state WHERE status = 'running';

-- Analyze table for better query planning
ANALYZE workflow_state;

-- Check database size
SELECT page_count * page_size as size_bytes
FROM pragma_page_count(), pragma_page_size();
```

## Migration Guide

### From Previous Versions

The SQLiteBackend automatically migrates existing databases. No manual intervention is required.

### Verification

```python
async def verify_migration(db_path: Path):
    """Verify that migration completed successfully."""
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("PRAGMA table_info(workflow_state)")
        columns = [row[1] for row in await cursor.fetchall()]

        required_columns = [
            'total_steps', 'error_message',
            'execution_time_ms', 'memory_usage_mb'
        ]

        missing_columns = [col for col in required_columns if col not in columns]

        if missing_columns:
            print(f"Missing columns: {missing_columns}")
        else:
            print("Migration completed successfully")
```

## API Reference

### SQLiteBackend

```python
class SQLiteBackend(StateBackend):
    """SQLite-backed persistent storage for workflow state with optimized schema."""

    def __init__(self, db_path: Path) -> None:
        """Initialize the backend with a database file path."""

    async def save_state(self, run_id: str, state: Dict[str, Any]) -> None:
        """Save workflow state with enhanced serialization."""

    async def load_state(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load workflow state with enhanced deserialization."""

    async def delete_state(self, run_id: str) -> None:
        """Delete workflow state."""

    async def list_workflows(
        self,
        status: Optional[str] = None,
        pipeline_id: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List workflows with optional filtering and pagination."""

    async def get_workflow_stats(self) -> Dict[str, Any]:
        """Get comprehensive workflow statistics."""

    async def get_failed_workflows(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Get failed workflows from the last N hours."""

    async def cleanup_old_workflows(self, days_old: float = 30) -> int:
        """Delete workflows older than specified days. Returns number of deleted workflows."""
```

## Testing

The SQLiteBackend includes comprehensive test coverage for:

- **Basic Operations**: Save, load, delete operations
- **Admin Queries**: All filtering and pagination scenarios
- **Edge Cases**: Empty databases, non-existent workflows, invalid inputs
- **Concurrency**: Multiple concurrent readers and writers
- **Performance**: Large dataset handling and query performance
- **Fault Tolerance**: Database corruption, migration failures, partial writes
- **Security**: SQL injection prevention and parameterized queries
- **Observability**: Logging, metrics, and error reporting

Run the tests with:

```bash
# Run all SQLiteBackend tests
pytest tests/unit/test_file_sqlite_backends.py -v

# Run performance tests
pytest tests/benchmarks/test_sqlite_performance.py -v

# Run fault tolerance tests
pytest tests/unit/test_sqlite_fault_tolerance.py -v

# Run observability tests
pytest tests/unit/test_sqlite_observability.py -v

# Run security tests
pytest tests/security/test_sql_injection_security.py -v
```
