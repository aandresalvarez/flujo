# Admin Quick Reference

## Common Admin Operations

### Initialize Backend
```python
from flujo.state import SQLiteBackend
from pathlib import Path

backend = SQLiteBackend(Path("workflow_state.db"))
```

### List Workflows
```python
# All workflows (paginated)
workflows = await backend.list_workflows(limit=50, offset=0)

# Running workflows only
running = await backend.list_workflows(status="running")

# Failed workflows only
failed = await backend.list_workflows(status="failed")

# Specific pipeline
pipeline_workflows = await backend.list_workflows(pipeline_id="my_pipeline")
```

### Get Statistics
```python
stats = await backend.get_workflow_stats()

print(f"Total: {stats['total_workflows']}")
print(f"Running: {stats['status_counts'].get('running', 0)}")
print(f"Failed: {stats['status_counts'].get('failed', 0)}")
print(f"Success Rate: {(stats['status_counts'].get('completed', 0) / stats['total_workflows']) * 100:.1f}%")
```

### Find Failed Workflows
```python
# Last 24 hours
recent_failures = await backend.get_failed_workflows(hours_back=24)

# Last week
weekly_failures = await backend.get_failed_workflows(hours_back=168)

for wf in recent_failures:
    print(f"{wf['run_id']}: {wf['error_message']}")
```

### Cleanup Operations
```python
# Delete workflows older than 30 days
deleted = await backend.cleanup_old_workflows(days_old=30)

# Delete workflows older than 7 days
weekly_cleanup = await backend.cleanup_old_workflows(days_old=7)
```

## Direct SQL Queries

### Performance Monitoring
```sql
-- Slow workflows (>5 minutes)
SELECT run_id, pipeline_name, execution_time_ms
FROM workflow_state
WHERE execution_time_ms > 300000
ORDER BY execution_time_ms DESC;

-- High memory usage
SELECT run_id, pipeline_name, memory_usage_mb
FROM workflow_state
WHERE memory_usage_mb > 100
ORDER BY memory_usage_mb DESC;
```

### Error Analysis
```sql
-- Most common errors
SELECT error_message, COUNT(*) as count
FROM workflow_state
WHERE status = 'failed' AND error_message IS NOT NULL
GROUP BY error_message
ORDER BY count DESC;

-- Failed workflows by pipeline
SELECT pipeline_name, COUNT(*) as failures
FROM workflow_state
WHERE status = 'failed'
GROUP BY pipeline_name
ORDER BY failures DESC;
```

### Recent Activity
```sql
-- Workflows created in last hour
SELECT run_id, pipeline_name, status, created_at
FROM workflow_state
WHERE created_at > datetime('now', '-1 hour')
ORDER BY created_at DESC;

-- Status changes in last 24 hours
SELECT run_id, pipeline_name, status, updated_at
FROM workflow_state
WHERE updated_at > datetime('now', '-1 day')
ORDER BY updated_at DESC;
```

## Health Check Script

```python
async def health_check():
    stats = await backend.get_workflow_stats()

    # Check failure rate
    total = stats['total_workflows']
    failed = stats['status_counts'].get('failed', 0)
    failure_rate = (failed / total) * 100 if total > 0 else 0

    if failure_rate > 10:
        print(f"🚨 High failure rate: {failure_rate:.1f}%")

    # Check stuck workflows
    running = stats['status_counts'].get('running', 0)
    if running > 50:
        print(f"⚠️  Many running workflows: {running}")

    # Check recent activity
    recent = stats['recent_workflows_24h']
    if recent < 5:
        print(f"📉 Low activity: {recent} workflows in 24h")

    return failure_rate < 10 and running < 50 and recent >= 5
```

## Monitoring Setup

### Regular Cleanup
```python
import asyncio
from datetime import datetime, timezone

async def maintenance_cleanup():
    while True:
        try:
            deleted = await backend.cleanup_old_workflows(days_old=30)
            print(f"{datetime.now(timezone.utc).isoformat()}: Cleaned up {deleted} old workflows")
        except Exception as e:
            print(f"Cleanup error: {e}")

        # Run every 6 hours
        await asyncio.sleep(6 * 60 * 60)
```

### Error Alerting
```python
async def error_monitor():
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

## Database Management

### Check Database Size
```sql
-- Table size
SELECT COUNT(*) as total_rows FROM workflow_state;

-- Indexes
SELECT * FROM sqlite_master WHERE type = 'index' AND tbl_name = 'workflow_state';

-- Database size (if using file backend)
SELECT page_count * page_size as size_bytes
FROM pragma_page_count(), pragma_page_size();
```

### Backup and Restore
```bash
# Backup
sqlite3 workflow_state.db ".backup backup_$(date +%Y%m%d_%H%M%S).db"

# Restore
sqlite3 workflow_state.db ".restore backup_20241201_143022.db"
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   ```sql
   SELECT run_id, pipeline_name, memory_usage_mb
   FROM workflow_state
   WHERE memory_usage_mb > 500
   ORDER BY memory_usage_mb DESC;
   ```

2. **Stuck Workflows**
   ```sql
   SELECT run_id, pipeline_name, updated_at
   FROM workflow_state
   WHERE status = 'running'
   AND updated_at < datetime('now', '-1 hour');
   ```

3. **Data Integrity**
   ```sql
   SELECT run_id, status
   FROM workflow_state
   WHERE status NOT IN ('running', 'paused', 'completed', 'failed', 'cancelled');
   ```

### Performance Tuning

```sql
-- Enable WAL mode for better concurrency
PRAGMA journal_mode = WAL;

-- Check query performance
EXPLAIN QUERY PLAN SELECT * FROM workflow_state WHERE status = 'running';

-- Analyze table for better query planning
ANALYZE workflow_state;
```
