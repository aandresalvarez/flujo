# Database Backends: SQLite and PostgreSQL

Flujo supports two production-ready database backends for persistent workflow state: **SQLite** and **PostgreSQL**. This guide explains when to use each, how to configure them, manage migrations, and optimize performance.

## Quick Decision Guide

| Use Case | Recommended Backend | Reason |
|----------|-------------------|--------|
| Development & Testing | SQLite | Zero configuration, fast setup |
| Single-server deployments | SQLite | Simple, reliable, sufficient for most workloads |
| High-volume production (>1000 runs/day) | PostgreSQL | Better concurrency, advanced indexing |
| Multi-server/microservices | PostgreSQL | Shared state, connection pooling |
| Complex metadata queries | PostgreSQL | JSONB with GIN indexes for fast filtering |
| Embedded/IoT applications | SQLite | Lightweight, no external dependencies |

## SQLite Backend

### Overview

SQLite is a self-contained, file-based database that requires no server process. It's perfect for development, single-server deployments, and applications with moderate concurrency requirements.

### Key Features

- **Zero Configuration**: Just specify a file path
- **Durable Persistence**: Workflows survive restarts
- **WAL Mode**: Write-Ahead Logging for better concurrency
- **Automatic Migrations**: Schema upgrades handled automatically
- **Thread-Safe**: Concurrent access with proper locking
- **Portable**: Single file, easy to backup and move

### Installation

SQLite support is built-in—no additional dependencies required.

```python
from flujo.state.backends.sqlite import SQLiteBackend
from pathlib import Path

backend = SQLiteBackend(Path("workflow_state.db"))
```

### Configuration

#### Via Code

```python
from pathlib import Path
from flujo.state.backends.sqlite import SQLiteBackend

# Absolute path
backend = SQLiteBackend(Path("/var/lib/flujo/workflow_state.db"))

# Relative path (resolved from current working directory)
backend = SQLiteBackend(Path("workflow_state.db"))
```

#### Via Configuration File (`flujo.toml`)

```toml
[state]
uri = "sqlite:///./workflow_state.db"  # Relative path
# or
uri = "sqlite:////var/lib/flujo/workflow_state.db"  # Absolute path
```

#### Via Environment Variable

```bash
export FLUJO_STATE_URI="sqlite:///./workflow_state.db"
```

### URI Format

SQLite URIs follow this pattern:

```
sqlite:///<path>
```

**Examples:**
- `sqlite:///./workflow_state.db` → Relative to current working directory
- `sqlite:///workflow_state.db` → Relative to current working directory
- `sqlite:////var/lib/flujo/workflow_state.db` → Absolute path (note the four slashes)
- `sqlite:////tmp/flujo_ops.db` → Absolute path on Unix systems

**Path Resolution Rules:**
- **Absolute paths**: `sqlite:////abs/path/to/db` (four slashes) are used as-is
- **Relative paths**: `sqlite:///./db` or `sqlite:///db` are resolved relative to the current working directory
- The path normalization follows [RFC 3986](https://datatracker.ietf.org/doc/html/rfc3986#section-3)

### Basic Usage

```python
from flujo.state.backends.sqlite import SQLiteBackend
from pathlib import Path
from datetime import datetime, timezone

# Initialize
backend = SQLiteBackend(Path("workflow_state.db"))

# Save state
state = {
    "run_id": "run_123",
    "pipeline_id": "data_processing",
    "pipeline_name": "Data Processing Pipeline",
    "pipeline_version": "1.0.0",
    "current_step_index": 2,
    "pipeline_context": {"input_data": "sample.csv"},
    "status": "running",
    "created_at": datetime.now(timezone.utc),
    "updated_at": datetime.now(timezone.utc),
    "total_steps": 5,
}
await backend.save_state("run_123", state)

# Load state
loaded = await backend.load_state("run_123")

# List runs with filtering
runs = await backend.list_runs(
    status="running",
    pipeline_name="Data Processing Pipeline",
    limit=10,
    offset=0,
    metadata_filter={"batch_id": "batch-001"}
)
```

### Migrations

SQLite migrations are handled automatically. When you upgrade Flujo, the backend detects schema changes and applies migrations on first access.

```python
# Migration happens automatically
backend = SQLiteBackend(Path("existing_workflow_state.db"))
# Schema is upgraded transparently on first use
```

**Manual Migration** (if needed):

```bash
# Run migrations manually
flujo migrate --dry-run  # Preview changes
flujo migrate            # Apply migrations
```

### Performance Optimization

#### WAL Mode

SQLite uses Write-Ahead Logging (WAL) mode by default, which provides:
- Better concurrency (readers don't block writers)
- Improved performance
- Automatic crash recovery

No configuration needed—it's enabled automatically.

#### Indexes

The backend creates strategic indexes automatically:

```sql
-- These are created automatically
CREATE INDEX idx_workflow_state_status ON workflow_state(status);
CREATE INDEX idx_workflow_state_pipeline_id ON workflow_state(pipeline_id);
CREATE INDEX idx_workflow_state_created_at ON workflow_state(created_at);
CREATE INDEX idx_workflow_state_updated_at ON workflow_state(updated_at);
```

#### Large Dataset Tips

```python
# Use pagination for large result sets
runs = await backend.list_runs(limit=100, offset=0)

# Filter early to reduce result size
recent_failures = await backend.list_runs(
    status="failed",
    limit=50
)

# Regular cleanup to prevent bloat
deleted = await backend.cleanup_old_workflows(days_old=90)
```

### Limitations

- **Concurrent Writers**: SQLite handles concurrent reads well, but many concurrent writers can cause lock contention
- **File Size**: While SQLite can handle large databases (100GB+), performance degrades with very large files
- **Network Access**: SQLite files must be accessible from the application server (no remote access)

### Best Practices

1. **Regular Backups**: Copy the database file regularly
   ```bash
   cp workflow_state.db workflow_state_backup_$(date +%Y%m%d).db
   ```

2. **Cleanup Old Data**: Prevent database bloat
   ```python
   await backend.cleanup_old_workflows(days_old=30)
   ```

3. **Monitor File Size**: Watch for database growth
   ```python
   import os
   size_mb = os.path.getsize("workflow_state.db") / (1024 * 1024)
   if size_mb > 1000:  # 1GB threshold
       print("Database is getting large, consider cleanup")
   ```

4. **Use Separate Databases**: For different environments or pipeline types
   ```python
   prod_backend = SQLiteBackend(Path("prod_workflow_state.db"))
   dev_backend = SQLiteBackend(Path("dev_workflow_state.db"))
   ```

## PostgreSQL Backend

### Overview

PostgreSQL is a powerful, open-source relational database ideal for high-volume production deployments, multi-server architectures, and applications requiring advanced querying capabilities.

### Key Features

- **High Concurrency**: Excellent performance with many concurrent connections
- **JSONB Support**: Native JSON storage with GIN indexes for fast queries
- **Connection Pooling**: Built-in async connection pool management
- **Advanced Indexing**: GIN indexes for efficient metadata filtering
- **Scalability**: Handles millions of workflow runs
- **Multi-Server**: Shared state across multiple application instances

### Installation

PostgreSQL support requires the `asyncpg` package:

```bash
# Install with postgres extra
pip install flujo[postgres]

# Or install asyncpg directly
pip install asyncpg
```

**Note**: If you configure a Postgres URI but `asyncpg` is not installed, Flujo will raise a clear error with installation instructions.

### Configuration

#### Via Code

```python
from flujo.state.backends.postgres import PostgresBackend

# Connection string (DSN format)
backend = PostgresBackend(
    "postgres://user:password@localhost:5432/flujo_db",
    auto_migrate=True,
    pool_min_size=1,
    pool_max_size=10
)
```

#### Via Configuration File (`flujo.toml`)

```toml
[state]
uri = "postgres://user:password@localhost:5432/flujo_db"

[state.postgres]
pool_min_size = 1
pool_max_size = 10
auto_migrate = true
```

#### Via Environment Variable

```bash
export FLUJO_STATE_URI="postgres://user:password@localhost:5432/flujo_db"
export FLUJO_AUTO_MIGRATE="true"
```

### Connection String Format

PostgreSQL connection strings follow the standard DSN format:

```
postgres://[user[:password]@][host][:port][/database][?param1=value1&...]
postgresql://[user[:password]@][host][:port][/database][?param1=value1&...]
```

**Examples:**
- `postgres://user:pass@localhost:5432/flujo_db`
- `postgres://user@localhost/flujo_db` (default port 5432, no password)
- `postgresql://user:pass@db.example.com:5432/flujo_db?sslmode=require`

**Common Parameters:**
- `sslmode=require` - Require SSL connection
- `connect_timeout=10` - Connection timeout in seconds
- `application_name=flujo` - Application identifier

### Connection Pooling

PostgresBackend uses asyncpg's connection pooling for efficient resource management:

```python
backend = PostgresBackend(
    dsn,
    pool_min_size=1,    # Minimum connections in pool
    pool_max_size=10,   # Maximum connections in pool
    auto_migrate=True
)
```

**Pool Sizing Guidelines:**
- **Small deployments** (< 100 concurrent requests): `pool_min_size=1, pool_max_size=5`
- **Medium deployments** (100-1000 concurrent): `pool_min_size=2, pool_max_size=10`
- **Large deployments** (> 1000 concurrent): `pool_min_size=5, pool_max_size=20`

### Basic Usage

```python
from flujo.state.backends.postgres import PostgresBackend
from datetime import datetime, timezone

# Initialize
backend = PostgresBackend(
    "postgres://user:pass@localhost:5432/flujo_db",
    auto_migrate=True
)

# Save state
state = {
    "run_id": "run_123",
    "pipeline_id": "data_processing",
    "pipeline_name": "Data Processing Pipeline",
    "pipeline_version": "1.0.0",
    "current_step_index": 2,
    "pipeline_context": {"input_data": "sample.csv"},
    "status": "running",
    "created_at": datetime.now(timezone.utc),
    "updated_at": datetime.now(timezone.utc),
    "total_steps": 5,
    "metadata": {"batch_id": "batch-001", "priority": "high"},
}
await backend.save_state("run_123", state)

# Load state
loaded = await backend.load_state("run_123")

# List runs with metadata filtering (uses GIN index)
runs = await backend.list_runs(
    status="running",
    pipeline_name="Data Processing Pipeline",
    limit=10,
    offset=0,
    metadata_filter={"batch_id": "batch-001"}  # Fast with GIN index
)
```

### Migrations

PostgreSQL migrations are managed through SQL migration files in `flujo/state/migrations/`.

#### Automatic Migrations

By default, migrations are applied automatically on backend initialization:

```python
backend = PostgresBackend(dsn, auto_migrate=True)  # Default
```

#### Manual Migrations

For production deployments, you may want to run migrations manually:

```bash
# Preview migrations
flujo migrate --dry-run

# Apply all pending migrations
flujo migrate

# Apply up to specific version
flujo migrate --target-version 2
```

#### Migration Files

Migrations are numbered sequentially:

- `001_init.sql` - Initial schema creation
- `002_metadata_index.sql` - GIN index for metadata JSONB column

**Example Migration** (`002_metadata_index.sql`):

```sql
BEGIN;

-- Create GIN index for efficient JSONB querying on metadata
CREATE INDEX IF NOT EXISTS idx_workflow_state_metadata_gin 
ON workflow_state USING GIN (metadata);

-- Update schema version
INSERT INTO flujo_schema_versions (version, applied_at) 
VALUES (2, NOW())
ON CONFLICT (version) DO NOTHING;

COMMIT;
```

#### Migration Tracking

Migrations are tracked in the `flujo_schema_versions` table:

```sql
SELECT version, applied_at 
FROM flujo_schema_versions 
ORDER BY version;
```

#### Disabling Auto-Migration

For production environments where you want explicit control:

```python
backend = PostgresBackend(dsn, auto_migrate=False)

# Then run migrations manually
# flujo migrate
```

If `auto_migrate=False` and schema is missing, the backend will raise an error with instructions.

### Performance Optimization

#### GIN Indexes for Metadata

PostgreSQL uses GIN (Generalized Inverted Index) indexes for efficient JSONB queries:

```sql
-- This index is created by migration 002
CREATE INDEX idx_workflow_state_metadata_gin 
ON workflow_state USING GIN (metadata);
```

This enables fast queries like:

```python
# This query uses the GIN index for fast filtering
runs = await backend.list_runs(
    metadata_filter={"batch_id": "batch-001", "priority": "high"}
)
```

**GIN Index Benefits:**
- Fast containment checks (`@>` operator)
- Efficient filtering on nested JSON structures
- Scales well with large metadata objects

#### Query Optimization

```python
# Good: Uses indexes
runs = await backend.list_runs(
    status="running",
    pipeline_name="Data Processing",
    limit=100
)

# Good: Metadata filter uses GIN index
runs = await backend.list_runs(
    metadata_filter={"batch_id": "batch-001"}
)

# Avoid: Full table scan (no filters)
all_runs = await backend.list_runs()  # Use pagination!
```

#### Connection Pool Tuning

Monitor pool usage and adjust:

```python
# For high-throughput applications
backend = PostgresBackend(
    dsn,
    pool_min_size=5,
    pool_max_size=20
)
```

### Advanced Features

#### JSONB Metadata Queries

PostgreSQL's JSONB support enables powerful metadata queries:

```python
# Filter by nested metadata
runs = await backend.list_runs(
    metadata_filter={
        "user": {"id": 123, "role": "admin"},
        "tags": ["urgent", "production"]
    }
)
```

#### Direct SQL Queries

For advanced analytics, you can query the database directly:

```sql
-- Get workflow statistics
SELECT 
    pipeline_name,
    status,
    COUNT(*) as count,
    AVG(EXTRACT(EPOCH FROM (updated_at - created_at))) as avg_duration_seconds
FROM workflow_state
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY pipeline_name, status
ORDER BY count DESC;

-- Find workflows with specific metadata
SELECT run_id, pipeline_name, metadata
FROM workflow_state
WHERE metadata @> '{"batch_id": "batch-001"}'::jsonb;
```

### Best Practices

1. **Use Connection Pooling**: Let the backend manage connections
   ```python
   backend = PostgresBackend(dsn, pool_max_size=10)
   ```

2. **Enable SSL in Production**: Use SSL for secure connections
   ```
   postgres://user:pass@host:5432/db?sslmode=require
   ```

3. **Monitor Pool Usage**: Watch for connection pool exhaustion
   ```python
   # Check pool stats (if available in your monitoring)
   # pool.size - current connections
   # pool.max_size - maximum connections
   ```

4. **Regular Vacuum**: PostgreSQL handles this automatically, but monitor
   ```sql
   -- Check table bloat
   SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
   FROM pg_tables
   WHERE schemaname = 'public';
   ```

5. **Backup Strategy**: Use PostgreSQL's native backup tools
   ```bash
   # pg_dump for backups
   pg_dump -h localhost -U user -d flujo_db > backup.sql
   ```

## Migration System

### How Migrations Work

Flujo uses a versioned migration system:

1. **Migration Files**: SQL files in `flujo/state/migrations/` numbered sequentially
2. **Version Tracking**: `flujo_schema_versions` table tracks applied migrations
3. **Automatic Application**: Migrations run automatically on backend initialization (if `auto_migrate=True`)
4. **Idempotent**: Migrations can be run multiple times safely

### Migration File Format

```sql
BEGIN;

-- Your migration SQL here
CREATE INDEX IF NOT EXISTS idx_example ON table_name(column_name);

-- Always update schema version
INSERT INTO flujo_schema_versions (version, applied_at) 
VALUES (3, NOW())
ON CONFLICT (version) DO NOTHING;

COMMIT;
```

### Creating New Migrations

1. **Create Migration File**: Add `00X_description.sql` to `flujo/state/migrations/`
2. **Use Transactions**: Wrap in `BEGIN;` and `COMMIT;`
3. **Update Version**: Always increment and record the version number
4. **Test**: Run migrations in a test environment first

**Example** (`003_add_custom_column.sql`):

```sql
BEGIN;

ALTER TABLE workflow_state 
ADD COLUMN IF NOT EXISTS custom_field TEXT;

INSERT INTO flujo_schema_versions (version, applied_at) 
VALUES (3, NOW())
ON CONFLICT (version) DO NOTHING;

COMMIT;
```

### Migration Best Practices

1. **Always Use IF NOT EXISTS**: Prevents errors on re-runs
   ```sql
   CREATE INDEX IF NOT EXISTS ...
   ALTER TABLE ... ADD COLUMN IF NOT EXISTS ...
   ```

2. **Test Backwards Compatibility**: Ensure old code works with new schema
3. **Document Breaking Changes**: Note any API changes in migration comments
4. **Run in Production Carefully**: Test migrations in staging first

## Choosing Between SQLite and PostgreSQL

### Use SQLite When:

- ✅ Development and testing environments
- ✅ Single-server deployments
- ✅ Low to moderate concurrency (< 50 concurrent writes)
- ✅ Simple deployment requirements
- ✅ Embedded or IoT applications
- ✅ Prototyping and demos

### Use PostgreSQL When:

- ✅ Production deployments with high volume (> 1000 runs/day)
- ✅ Multi-server or microservices architecture
- ✅ High concurrency requirements (> 50 concurrent writes)
- ✅ Complex metadata queries with JSONB
- ✅ Need for advanced analytics and reporting
- ✅ Shared state across multiple application instances
- ✅ Enterprise-grade reliability and monitoring

### Migration Path

You can migrate from SQLite to PostgreSQL:

1. **Export from SQLite**:
   ```python
   # Export all runs
   runs = await sqlite_backend.list_runs()
   ```

2. **Import to PostgreSQL**:
   ```python
   # Import runs
   for run in runs:
       state = await sqlite_backend.load_state(run['run_id'])
       await postgres_backend.save_state(run['run_id'], state)
   ```

3. **Update Configuration**: Change `FLUJO_STATE_URI` to Postgres connection string

## Troubleshooting

### SQLite Issues

**Database is locked**
- **Cause**: Too many concurrent writers
- **Solution**: Reduce concurrency or use PostgreSQL for high-write scenarios

**Database file not found**
- **Cause**: Path resolution issue
- **Solution**: Use absolute paths or verify current working directory

**Slow queries**
- **Cause**: Missing indexes or large database
- **Solution**: Ensure indexes exist, use pagination, cleanup old data

### PostgreSQL Issues

**Connection refused**
- **Cause**: Database server not running or wrong host/port
- **Solution**: Verify connection string and database server status

**Pool exhaustion**
- **Cause**: Too many concurrent connections
- **Solution**: Increase `pool_max_size` or reduce application concurrency

**Migration errors**
- **Cause**: Manual schema changes or migration conflicts
- **Solution**: Check `flujo_schema_versions` table, run migrations manually

**Slow metadata queries**
- **Cause**: Missing GIN index
- **Solution**: Ensure migration 002 is applied:
  ```sql
  SELECT version FROM flujo_schema_versions WHERE version = 2;
  ```

## Configuration Examples

### Development (SQLite)

```toml
# flujo.toml
[state]
uri = "sqlite:///./dev_workflow_state.db"
```

### Production (PostgreSQL)

```toml
# flujo.toml
[state]
uri = "postgres://user:password@db.example.com:5432/flujo_prod"

[state.postgres]
pool_min_size = 2
pool_max_size = 10
auto_migrate = false  # Run migrations manually in production
```

### Environment-Specific

```bash
# Development
export FLUJO_STATE_URI="sqlite:///./dev_workflow_state.db"

# Staging
export FLUJO_STATE_URI="postgres://user:pass@staging-db:5432/flujo_staging"

# Production
export FLUJO_STATE_URI="postgres://user:pass@prod-db:5432/flujo_prod?sslmode=require"
```

## Summary

Both SQLite and PostgreSQL are production-ready backends for Flujo:

- **SQLite**: Perfect for development, single-server deployments, and moderate workloads
- **PostgreSQL**: Ideal for high-volume production, multi-server architectures, and advanced querying

Choose based on your deployment requirements, concurrency needs, and scalability goals. Both backends support automatic migrations, provide excellent performance, and offer comprehensive observability features.
