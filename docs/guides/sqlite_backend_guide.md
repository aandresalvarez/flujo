# SQLite Backend Guide

## Overview

The `SQLiteBackend` is a production-ready state backend for Flujo that provides durable workflow persistence with advanced observability features. It's designed for high-performance, concurrent access patterns and includes comprehensive admin query capabilities.

### Key Features

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
print(f"Pipeline: {loaded_state['pipeline_name']}")
print(f"Status: {loaded_state['status']}")
print(f"Current step: {loaded_state['current_step_index']}")

# Delete workflow state
await backend.delete_state("run_123")
```

## Admin Queries and Observability

The SQLiteBackend provides powerful built-in observability features that allow you to monitor and analyze your workflows.

### List Workflows

```python
# List all workflows
all_workflows = await backend.list_workflows()
print(f"Total workflows: {len(all_workflows)}")

# Filter by status
running_workflows = await backend.list_workflows(status="running")
failed_workflows = await backend.list_workflows(status="failed")

# Filter by pipeline
pipeline_workflows = await backend.list_workflows(pipeline_id="data_processing")

# Pagination
recent_workflows = await backend.list_workflows(limit=10, offset=0)
```

### Workflow Statistics

```python
# Get comprehensive workflow statistics
stats = await backend.get_workflow_stats()

print(f"Total workflows: {stats['total_workflows']}")
print(f"Status breakdown: {stats['status_counts']}")
print(f"Recent workflows (24h): {stats['recent_workflows_24h']}")
print(f"Average execution time: {stats['average_execution_time_ms']}ms")

# Example output:
# Total workflows: 150
# Status breakdown: {'running': 5, 'completed': 120, 'failed': 15, 'paused': 10}
# Recent workflows (24h): 25
# Average execution time: 1250ms
```

### Failed Workflows Analysis

```python
# Get failed workflows from the last 24 hours
failed_24h = await backend.get_failed_workflows(hours_back=24)

for workflow in failed_24h:
    print(f"Failed workflow: {workflow['run_id']}")
    print(f"  Pipeline: {workflow['pipeline_name']}")
    print(f"  Error: {workflow['error_message']}")
    print(f"  Failed at: {workflow['updated_at']}")

# Get failed workflows from the last week
failed_week = await backend.get_failed_workflows(hours_back=168)
```

### Cleanup Operations

```python
# Clean up workflows older than 30 days
deleted_count = await backend.cleanup_old_workflows(days_old=30)
print(f"Deleted {deleted_count} old workflows")

# Clean up workflows older than 7 days
deleted_count = await backend.cleanup_old_workflows(days_old=7)
```

## Direct SQL Queries

For advanced operational tasks, you can directly query the SQLite database to gain deep insights into your workflow performance and behavior.

### Performance Monitoring Queries

```sql
-- Get average execution time by pipeline
SELECT
    pipeline_name,
    COUNT(*) as total_runs,
    AVG(execution_time_ms) as avg_execution_time,
    MAX(execution_time_ms) as max_execution_time,
    MIN(execution_time_ms) as min_execution_time
FROM workflow_state
WHERE execution_time_ms IS NOT NULL
GROUP BY pipeline_name
ORDER BY avg_execution_time DESC;

-- Get memory usage statistics
SELECT
    pipeline_name,
    AVG(memory_usage_mb) as avg_memory_usage,
    MAX(memory_usage_mb) as max_memory_usage,
    COUNT(*) as total_runs
FROM workflow_state
WHERE memory_usage_mb IS NOT NULL
GROUP BY pipeline_name;

-- Find slow workflows (execution time > 5 seconds)
SELECT
    run_id,
    pipeline_name,
    execution_time_ms,
    created_at
FROM workflow_state
WHERE execution_time_ms > 5000
ORDER BY execution_time_ms DESC;
```

### Error Analysis Queries

```sql
-- Get error patterns
SELECT
    error_message,
    COUNT(*) as error_count,
    pipeline_name
FROM workflow_state
WHERE error_message IS NOT NULL
GROUP BY error_message, pipeline_name
ORDER BY error_count DESC;

-- Find workflows that failed at specific steps
SELECT
    run_id,
    pipeline_name,
    current_step_index,
    error_message,
    updated_at
FROM workflow_state
WHERE status = 'failed'
ORDER BY updated_at DESC;

-- Get failure rate by pipeline
SELECT
    pipeline_name,
    COUNT(*) as total_runs,
    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_runs,
    ROUND(
        (SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)),
        2
    ) as failure_rate_percent
FROM workflow_state
GROUP BY pipeline_name
ORDER BY failure_rate_percent DESC;
```

### Activity Analysis Queries

```sql
-- Get workflow activity by hour
SELECT
    strftime('%Y-%m-%d %H:00:00', created_at) as hour,
    COUNT(*) as workflows_started,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
FROM workflow_state
WHERE created_at >= datetime('now', '-7 days')
GROUP BY hour
ORDER BY hour DESC;

-- Find most active pipelines
SELECT
    pipeline_name,
    COUNT(*) as total_runs,
    COUNT(DISTINCT DATE(created_at)) as active_days
FROM workflow_state
WHERE created_at >= datetime('now', '-30 days')
GROUP BY pipeline_name
ORDER BY total_runs DESC;

-- Get step completion statistics
SELECT
    pipeline_name,
    AVG(current_step_index) as avg_step_progress,
    MAX(current_step_index) as max_steps,
    COUNT(*) as total_workflows
FROM workflow_state
WHERE status = 'running'
GROUP BY pipeline_name;
```

## Performance Considerations

### WAL Mode and Concurrency

The SQLiteBackend uses Write-Ahead Logging (WAL) mode by default, which provides:

- **Better Concurrency**: Multiple readers can access the database simultaneously
- **Improved Performance**: Writes don't block reads
- **Crash Recovery**: Automatic recovery from unexpected shutdowns

```python
# The backend automatically enables WAL mode during initialization
backend = SQLiteBackend(Path("workflow_state.db"))
# WAL mode is enabled automatically - no additional configuration needed
```

### Indexed Queries

The backend creates strategic indexes for optimal query performance:

```sql
-- These indexes are created automatically
CREATE INDEX idx_workflow_state_status ON workflow_state(status);
CREATE INDEX idx_workflow_state_created_at ON workflow_state(created_at);
CREATE INDEX idx_workflow_state_pipeline_id ON workflow_state(pipeline_id);
```

### Large Dataset Optimization

For databases with thousands of workflows:

```python
# Use pagination for large result sets
workflows = await backend.list_workflows(limit=100, offset=0)

# Filter by specific criteria to reduce result size
recent_failures = await backend.list_workflows(
    status="failed",
    limit=50
)

# Use cleanup to prevent database bloat
await backend.cleanup_old_workflows(days_old=90)
```

## Fault Tolerance and Recovery

### Automatic Schema Migration

The backend automatically handles schema upgrades:

```python
# If you upgrade Flujo and the schema changes, the backend will:
# 1. Detect the schema mismatch
# 2. Automatically migrate existing data
# 3. Preserve all workflow state
# 4. Continue normal operation

backend = SQLiteBackend(Path("existing_workflow_state.db"))
# Migration happens automatically on first access
```

### Database Corruption Recovery

The backend includes robust error handling and recovery mechanisms:

```python
# The backend automatically retries operations on database locks
# and handles corruption scenarios gracefully

try:
    state = await backend.load_state("run_123")
except Exception as e:
    # The backend will attempt automatic recovery
    print(f"Recovery attempted: {e}")
```

### Connection Pooling and Retry Logic

```python
# The backend implements intelligent retry logic with exponential backoff
# for database locked errors and schema migration scenarios

# No additional configuration needed - handled automatically
backend = SQLiteBackend(Path("workflow_state.db"))
```

## Security Considerations

### SQL Injection Prevention

The SQLiteBackend uses parameterized queries throughout to prevent SQL injection:

```python
# All user inputs are properly parameterized
await backend.list_workflows(status="running")  # Safe
await backend.list_workflows(pipeline_id="user_input")  # Safe

# The backend never constructs SQL queries by string concatenation
# All queries use parameterized statements with proper escaping
```

### Input Validation

```python
# The backend validates all inputs before database operations
# SQL identifiers are validated for problematic characters
# Column definitions are validated against a whitelist

# Example of safe usage:
state = {
    "run_id": "safe_run_id_123",  # Validated
    "pipeline_id": "safe_pipeline_name",  # Validated
    # ... other fields
}
await backend.save_state("run_123", state)
```

## Best Practices

### Maintenance

```python
# Regular cleanup to prevent database bloat
import asyncio

async def maintenance_routine():
    backend = SQLiteBackend(Path("workflow_state.db"))

    # Clean up workflows older than 30 days
    deleted = await backend.cleanup_old_workflows(days_old=30)
    print(f"Cleaned up {deleted} old workflows")

    # Get statistics for monitoring
    stats = await backend.get_workflow_stats()
    print(f"Database contains {stats['total_workflows']} workflows")

# Run maintenance weekly
# asyncio.run(maintenance_routine())
```

### Monitoring

```python
# Set up monitoring for your workflows
from datetime import datetime, timezone

async def monitor_workflows():
    backend = SQLiteBackend(Path("workflow_state.db"))

    # Check for failed workflows
    failed = await backend.get_failed_workflows(hours_back=1)
    if failed:
        print(f"Alert: {len(failed)} workflows failed in the last hour")
        for wf in failed:
            print(f"  - {wf['run_id']}: {wf['error_message']}")

    # Monitor performance
    stats = await backend.get_workflow_stats()
    if stats['average_execution_time_ms'] > 10000:  # 10 seconds
        print("Alert: Average execution time is high")

    # Check for stuck workflows
    running = await backend.list_workflows(status="running")
    for wf in running:
        # Check if workflow has been running too long
        created = datetime.fromisoformat(wf['created_at'])
        if (datetime.now(timezone.utc) - created).total_seconds() > 3600:  # 1 hour
            print(f"Alert: Workflow {wf['run_id']} has been running for over 1 hour")
```

### Backups

```python
import shutil
from datetime import datetime, timezone
from pathlib import Path

def backup_database():
    """Create a backup of the workflow database."""
    db_path = Path("workflow_state.db")
    backup_path = Path(
        f"workflow_state_backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.db"
    )

    if db_path.exists():
        shutil.copy2(db_path, backup_path)
        print(f"Backup created: {backup_path}")

    # Also backup the WAL file if it exists
    wal_path = db_path.with_suffix('.db-wal')
    if wal_path.exists():
        wal_backup = backup_path.with_suffix('.db-wal')
        shutil.copy2(wal_path, wal_backup)
        print(f"WAL backup created: {wal_backup}")

# Run backups daily
# backup_database()
```

### Configuration

```python
# Use environment variables for configuration
import os
from pathlib import Path

# Set database path via environment variable
db_path = Path(os.getenv("FLUJO_DB_PATH", "flujo_ops.db"))
backend = SQLiteBackend(db_path)

# Or use the configuration system
from flujo.cli.config import load_backend_from_config
backend = load_backend_from_config()
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
        """Clean up workflows older than specified days. Returns number of deleted workflows."""
```

## Troubleshooting

### Common Issues

**Database is locked errors**
```python
# The backend automatically handles database locks with retry logic
# If you see these errors frequently, consider:
# 1. Reducing concurrent access
# 2. Using separate database files for different applications
# 3. Implementing proper connection pooling in your application
```

**High memory usage**
```python
# Monitor memory usage with the built-in tracking
stats = await backend.get_workflow_stats()
workflows = await backend.list_workflows()

for wf in workflows:
    if wf.get('memory_usage_mb', 0) > 100:  # 100MB threshold
        print(f"High memory usage: {wf['run_id']} - {wf['memory_usage_mb']}MB")
```

**Slow queries**
```python
# Use the provided indexes effectively
# Filter by status or pipeline_id when possible
# Use pagination for large result sets

# Good: Uses index
fast_query = await backend.list_workflows(status="running", limit=10)

# Avoid: No filtering, could be slow with large datasets
slow_query = await backend.list_workflows()  # No limit or filters
```

### Debug Queries

```sql
-- Check database size
SELECT
    name,
    page_count * page_size as size_bytes,
    page_count * page_size / 1024.0 / 1024.0 as size_mb
FROM pragma_page_count(), pragma_page_size()
WHERE name = 'workflow_state';

-- Check index usage
SELECT
    name,
    sql
FROM sqlite_master
WHERE type = 'index' AND tbl_name = 'workflow_state';

-- Check for database corruption
PRAGMA integrity_check;

-- Analyze query performance
EXPLAIN QUERY PLAN
SELECT * FROM workflow_state WHERE status = 'running';
```

### Performance Tuning

```python
# For high-throughput applications, consider:
# 1. Using separate database files for different pipeline types
# 2. Implementing application-level caching
# 3. Regular cleanup to maintain optimal performance

# Example: Separate databases by pipeline type
data_processing_backend = SQLiteBackend(Path("data_processing.db"))
ml_training_backend = SQLiteBackend(Path("ml_training.db"))
```

This comprehensive guide covers all aspects of the SQLiteBackend, from basic usage to advanced operational tasks. The backend is production-ready and provides the observability and reliability needed for serious workflow management.
