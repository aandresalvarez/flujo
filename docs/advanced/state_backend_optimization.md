# Optimized SQLite State Backend

## Overview

The SQLite state backend has been optimized for production use cases involving large numbers of durable workflows. This refactor significantly improves performance and utility through:

- **Performance at Scale**: Structured SQL schema with indexed columns for fast queries
- **Observability & Debugging**: Direct database queryability for operators
- **Advanced Features**: Foundation for UI dashboards and admin tools

## Key Improvements

### 1. Structured Schema
The new schema moves status, timestamps, and step indices into native SQL columns:

```sql
CREATE TABLE workflow_state (
    run_id TEXT PRIMARY KEY,
    pipeline_id TEXT NOT NULL,
    pipeline_name TEXT NOT NULL,
    pipeline_version TEXT NOT NULL,
    current_step_index INTEGER NOT NULL DEFAULT 0,
    pipeline_context TEXT NOT NULL,  -- JSON blob
    last_step_output TEXT,           -- JSON blob, nullable
    status TEXT NOT NULL CHECK (status IN ('running', 'paused', 'completed', 'failed', 'cancelled')),
    created_at TEXT NOT NULL,        -- ISO format datetime
    updated_at TEXT NOT NULL,        -- ISO format datetime

    -- Additional metadata for better observability
    total_steps INTEGER DEFAULT 0,
    error_message TEXT,
    execution_time_ms INTEGER,
    memory_usage_mb REAL
);
```

### 2. Performance Indexes
Strategic indexes for common query patterns:

```sql
CREATE INDEX idx_workflow_state_status ON workflow_state(status);
CREATE INDEX idx_workflow_state_pipeline_id ON workflow_state(pipeline_id);
CREATE INDEX idx_workflow_state_created_at ON workflow_state(created_at);
CREATE INDEX idx_workflow_state_updated_at ON workflow_state(updated_at);
CREATE INDEX idx_workflow_state_status_updated ON workflow_state(status, updated_at);
CREATE INDEX idx_workflow_state_pipeline_status ON workflow_state(pipeline_id, status);
```

### 3. Automatic Migration
Existing databases are automatically migrated to the new schema without data loss.

## Basic Usage

### Standard Operations

```python
from flujo.state import SQLiteBackend
from datetime import datetime, timezone
from pathlib import Path

# Initialize backend
backend = SQLiteBackend(Path("workflow_state.db"))

# Save workflow state
state = {
    "run_id": "run_123",
    "pipeline_id": "data_processing",
    "pipeline_name": "Data Processing Pipeline",
    "pipeline_version": "1.0.0",
    "current_step_index": 2,
    "pipeline_context": {"input_data": "sample.csv"},
    "last_step_output": {"processed_rows": 1000},
    "status": "running",
    "created_at": datetime.now(timezone.utc),
    "updated_at": datetime.now(timezone.utc),
    "total_steps": 5,
    "error_message": None,
    "execution_time_ms": 1500,
    "memory_usage_mb": 45.2
}

await backend.save_state("run_123", state)

# Load workflow state
loaded_state = await backend.load_state("run_123")

# Delete workflow state
await backend.delete_state("run_123")
```

## Admin Queries and Observability

### 1. List Workflows with Filtering

```python
# List all workflows
all_workflows = await backend.list_workflows()

# List only running workflows
running_workflows = await backend.list_workflows(status="running")

# List workflows for a specific pipeline
pipeline_workflows = await backend.list_workflows(pipeline_id="data_processing")

# List with pagination
recent_workflows = await backend.list_workflows(
    limit=10,
    offset=0
)

# Combined filtering
filtered_workflows = await backend.list_workflows(
    status="failed",
    pipeline_id="data_processing",
    limit=5
)
```

### 2. Get Workflow Statistics

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

### 3. Find Failed Workflows

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
```

### 4. Cleanup Old Workflows

```python
# Delete workflows older than 30 days
deleted_count = await backend.cleanup_old_workflows(days_old=30)
print(f"Deleted {deleted_count} old workflows")

# Delete workflows older than 7 days
weekly_cleanup = await backend.cleanup_old_workflows(days_old=7)
```

## Direct SQL Queries

For advanced operators, the database is directly queryable:

### Performance Monitoring

```sql
-- Find slow workflows (execution time > 5 minutes)
SELECT run_id, pipeline_name, execution_time_ms, created_at
FROM workflow_state
WHERE execution_time_ms > 300000
ORDER BY execution_time_ms DESC;

-- Find workflows with high memory usage
SELECT run_id, pipeline_name, memory_usage_mb, created_at
FROM workflow_state
WHERE memory_usage_mb > 100
ORDER BY memory_usage_mb DESC;
```

### Error Analysis

```sql
-- Most common error messages
SELECT error_message, COUNT(*) as error_count
FROM workflow_state
WHERE status = 'failed' AND error_message IS NOT NULL
GROUP BY error_message
ORDER BY error_count DESC
LIMIT 10;

-- Failed workflows by pipeline
SELECT pipeline_name, COUNT(*) as failure_count
FROM workflow_state
WHERE status = 'failed'
GROUP BY pipeline_name
ORDER BY failure_count DESC;
```

### Pipeline Performance

```sql
-- Average execution time by pipeline
SELECT
    pipeline_name,
    AVG(execution_time_ms) as avg_execution_time,
    COUNT(*) as total_runs,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful_runs
FROM workflow_state
WHERE execution_time_ms IS NOT NULL
GROUP BY pipeline_name
ORDER BY avg_execution_time DESC;
```

### Recent Activity

```sql
-- Workflows created in the last hour
SELECT run_id, pipeline_name, status, created_at
FROM workflow_state
WHERE created_at > datetime('now', '-1 hour')
ORDER BY created_at DESC;

-- Status changes in the last 24 hours
SELECT run_id, pipeline_name, status, updated_at
FROM workflow_state
WHERE updated_at > datetime('now', '-1 day')
ORDER BY updated_at DESC;
```

## Migration Strategy

### Automatic Migration
The backend automatically migrates existing databases:

1. **Schema Detection**: Checks for existing columns
2. **Column Addition**: Adds new columns with appropriate defaults
3. **Data Migration**: Updates existing data to meet new constraints
4. **Index Creation**: Creates performance indexes

### Manual Migration Verification

```python
import aiosqlite

async def verify_migration(db_path):
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

## Performance Considerations

### Index Usage
- Queries on `status`, `pipeline_id`, and timestamps are optimized
- Composite indexes support common filtering patterns
- Consider your query patterns when adding custom indexes

### WAL Mode
The backend enables WAL (Write-Ahead Logging) mode for:
- Better concurrent read/write performance
- Reduced blocking during writes
- Improved durability

### Memory Management
- Large JSON blobs in `pipeline_context` and `last_step_output` are stored as text
- Consider archiving old workflows to manage database size
- Monitor `memory_usage_mb` for resource-intensive workflows

## Best Practices

### 1. Regular Cleanup
```python
# Set up automated cleanup
async def maintenance_cleanup():
    # Clean up workflows older than 30 days
    deleted = await backend.cleanup_old_workflows(days_old=30)
    print(f"Maintenance: deleted {deleted} old workflows")
```

### 2. Monitoring Setup
```python
# Regular health checks
async def health_check():
    stats = await backend.get_workflow_stats()

    # Alert on high failure rate
    total = stats['total_workflows']
    failed = stats['status_counts'].get('failed', 0)
    failure_rate = failed / total if total > 0 else 0

    if failure_rate > 0.1:  # 10% failure rate
        print(f"WARNING: High failure rate: {failure_rate:.2%}")

    # Alert on stuck workflows
    running = stats['status_counts'].get('running', 0)
    if running > 50:
        print(f"WARNING: Many running workflows: {running}")
```

### 3. Error Tracking
```python
# Monitor specific error patterns
async def error_analysis():
    failed_workflows = await backend.get_failed_workflows(hours_back=24)

    error_patterns = {}
    for workflow in failed_workflows:
        error = workflow['error_message']
        if error:
            error_patterns[error] = error_patterns.get(error, 0) + 1

    # Report common errors
    for error, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True):
        print(f"Error: {error} (occurred {count} times)")
```

## Future Enhancements

The optimized schema enables future features:

1. **Web Dashboard**: Real-time workflow monitoring
2. **Admin CLI**: Command-line workflow management
3. **Alerting**: Automated notifications for failures
4. **Analytics**: Performance trend analysis
5. **Workflow Templates**: Reusable pipeline configurations

## Troubleshooting

### Common Issues

1. **Migration Failures**: Check database permissions and disk space
2. **Performance Issues**: Verify indexes are created and being used
3. **Memory Usage**: Monitor JSON blob sizes in context and output fields

### Debug Queries

```sql
-- Check index usage
SELECT * FROM sqlite_master WHERE type = 'index' AND tbl_name = 'workflow_state';

-- Check table size
SELECT COUNT(*) as total_rows FROM workflow_state;

-- Check for data integrity issues
SELECT run_id, status FROM workflow_state WHERE status NOT IN ('running', 'paused', 'completed', 'failed', 'cancelled');
```
