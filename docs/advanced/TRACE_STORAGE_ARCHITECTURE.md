# Trace Storage Architecture

## Current Implementation (v2.0)

### Schema Design
The current implementation uses a normalized schema for optimal querying and analytics:

```sql
CREATE TABLE spans (
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
);

-- Performance indexes for efficient querying
CREATE INDEX idx_spans_run_id ON spans(run_id);
CREATE INDEX idx_spans_status ON spans(status);
CREATE INDEX idx_spans_name ON spans(name);
CREATE INDEX idx_spans_start_time ON spans(start_time);
CREATE INDEX idx_spans_parent_span_id ON spans(parent_span_id);
```

### Architectural Benefits

#### ✅ **Advantages of Normalized Approach**
- **Server-Side Querying**: Full SQL querying capabilities on individual spans
- **Analytics Support**: Direct SQL analytics on span data across runs
- **Storage Efficiency**: Optimized storage with proper indexing
- **Performance**: Fast queries with strategic indexes
- **Scalability**: Supports large-scale deployments
- **Flexibility**: Extensible schema for future enhancements

#### ✅ **Query Capabilities**

**Span-Level Queries:**
```sql
-- Find all failed spans across all runs
SELECT s.name, s.status, s.start_time, s.end_time, r.pipeline_name
FROM spans s
JOIN runs r ON s.run_id = r.run_id
WHERE s.status = 'failed'
ORDER BY s.start_time DESC;

-- Calculate average span duration by name
SELECT name, AVG(end_time - start_time) as avg_duration, COUNT(*) as count
FROM spans
WHERE end_time IS NOT NULL
GROUP BY name
ORDER BY avg_duration DESC;

-- Find spans with specific attributes
SELECT span_id, name, attributes
FROM spans
WHERE json_extract(attributes, '$.error_type') = 'timeout';
```

**Analytics Queries:**
```sql
-- Performance analysis by pipeline
SELECT r.pipeline_name,
       COUNT(*) as total_spans,
       AVG(s.end_time - s.start_time) as avg_duration,
       SUM(CASE WHEN s.status = 'failed' THEN 1 ELSE 0 END) as failed_count
FROM spans s
JOIN runs r ON s.run_id = r.run_id
WHERE s.end_time IS NOT NULL
GROUP BY r.pipeline_name
ORDER BY avg_duration DESC;

-- Error pattern analysis
SELECT name, status, COUNT(*) as count
FROM spans
WHERE status = 'failed'
GROUP BY name, status
ORDER BY count DESC;
```

### API Design

#### Core Trace Methods
```python
async def save_trace(self, run_id: str, trace: Dict[str, Any]) -> None:
    """Persist a trace tree as normalized spans for a given run_id."""

async def get_trace(self, run_id: str) -> Any:
    """Retrieve and reconstruct the trace tree for a given run_id."""
```

#### Span-Level Query Methods
```python
async def get_spans(
    self,
    run_id: str,
    status: Optional[str] = None,
    name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get individual spans with optional filtering."""

async def get_span_statistics(
    self,
    pipeline_name: Optional[str] = None,
    time_range: Optional[tuple] = None
) -> Dict[str, Any]:
    """Get aggregated span statistics."""
```

### CLI Commands

**Trace Visualization:**
```bash
# Show hierarchical trace tree
flujo lens trace <run_id>

# List individual spans with filtering
flujo lens spans <run_id> --status completed --name step_1

# Show aggregated statistics
flujo lens stats --pipeline my_pipeline --hours 24
```

### Implementation Details

#### Trace Tree Reconstruction
The system automatically reconstructs hierarchical trace trees from normalized spans:

1. **Extraction**: Trace trees are flattened into individual spans during save
2. **Storage**: Each span is stored as a separate row with parent-child relationships
3. **Reconstruction**: Hierarchical structure is rebuilt using `parent_span_id` references
4. **Querying**: Both tree and span-level access are supported

#### Performance Optimizations
- **Batch Insertion**: Uses `executemany` for efficient span storage
- **Strategic Indexing**: Indexes on common query patterns
- **Foreign Key Constraints**: Ensures data integrity with cascade deletion
- **Connection Pooling**: Efficient database connection management

#### Error Handling
- **Graceful Degradation**: Trace failures don't break pipeline execution
- **Data Validation**: Robust handling of malformed trace data
- **Recovery Mechanisms**: Automatic cleanup and retry logic

### Migration from v1.0

The v2.0 implementation replaces the previous JSON blob storage with a fully normalized schema. This provides:

1. **Enhanced Query Capabilities**: Full SQL querying on span data
2. **Better Performance**: Optimized storage and indexing
3. **Analytics Support**: Built-in statistical analysis
4. **Future-Proof Design**: Extensible for advanced features

### Decision Record

**Date**: 2025-01-14
**Decision**: Implement normalized span storage (v2.0)
**Rationale**:
- Provides full querying and analytics capabilities
- Optimized for performance and scalability
- Supports production monitoring and debugging
- Foundation for advanced observability features

**Key Benefits**:
- Server-side span querying and filtering
- Cross-run analytics and statistics
- Performance optimization with proper indexing
- Extensible architecture for future enhancements

---

## Related Documentation
- [Tracing Guide](../guides/tracing_guide.md)
- [Trace Contract](../reference/trace_contract.md)
- [CLI Trace Visualization](../cookbook/console_tracer.md)
