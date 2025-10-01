# Tracing and Debugging Guide

Flujo provides rich internal tracing and visualization capabilities (FSD-12) that help you debug and analyze pipeline execution. This guide shows you how to use these features effectively.

## Overview

The tracing system captures:
- **Hierarchical execution flow** with parent-child relationships
- **Precise timing** for each step and the overall pipeline
- **Status tracking** (running, completed, failed)
- **Metadata** including attempts, costs, and token counts
- **Error details** with sanitized feedback

## Enabling Tracing

Tracing is enabled by default when you create a `Flujo` instance:

```python
from flujo import Flujo, Pipeline, Step

# Tracing is enabled by default
flujo = Flujo(pipeline=pipeline)

# Or explicitly enable it
flujo = Flujo(pipeline=pipeline, enable_tracing=True)
```

## Accessing Trace Information

After running a pipeline, you can access the trace tree:

```python
async for result in flujo.run_async("input_data"):
    pass

# Access the trace tree
if result.trace_tree:
    print(f"Root span: {result.trace_tree.name}")
    print(f"Status: {result.trace_tree.status}")
    print(f"Duration: {result.trace_tree.end_time - result.trace_tree.start_time:.3f}s")
    print(f"Children: {len(result.trace_tree.children)}")

# Access step history
for step in result.step_history:
    print(f"{step.name}: {'✅' if step.success else '❌'} ({step.latency_s:.3f}s)")
```

## CLI Debugging Tools

Flujo provides powerful CLI tools for inspecting traces. **New in Lens v2.0**: Enhanced performance, partial ID matching, JSON output, and more!

> 💡 **Quick Start**: See [Lens Quick Start Guide](lens_quickstart.md) for all new features.

### List All Runs

```bash
flujo lens list

# Filter by status
flujo lens list --status failed

# Limit results
flujo lens list --limit 100
```

Shows all pipeline runs with basic information:
- Run ID
- Pipeline name
- Status
- Start time
- Duration

### Find Run by Partial ID (New!)

```bash
# Fuzzy search by partial ID
flujo lens get abc123

# Works with any substring
flujo lens get ec00798f
```

Quickly find runs without copying full 32-character IDs.

### View Run Details

```bash
# Basic details (supports partial IDs)
flujo lens show <run_id>

# Show everything
flujo lens show <run_id> --verbose

# Show final output only
flujo lens show <run_id> --final-output

# Export as JSON for automation
flujo lens show <run_id> --json

# Combine flags
flujo lens show <run_id> --verbose --final-output --json
```

Shows detailed information about a specific run:
- Pipeline configuration with rich formatting
- Step results with execution times
- Final output (with `--final-output`)
- Error details (if any)

**New Features**:
- ✅ Partial run_id matching (use 8-12 chars instead of full 32)
- ✅ JSON output for CI/CD pipelines
- ✅ Final output display with `--final-output`
- ✅ Configurable timeout with `--timeout`
- ✅ Better error messages with troubleshooting suggestions

### View Hierarchical Trace

```bash
flujo lens trace <run_id>

# Control prompt preview length
flujo lens trace <run_id> --prompt-preview-len 500
```

Displays a tree-based view of the execution trace:
```
pipeline_root (completed, 1.234s)
├── step1 (completed, 0.123s)
├── loop1 (completed, 0.456s)
│   ├── loop_step (completed, 0.234s)
│   └── loop_step (completed, 0.222s)
├── conditional1 (completed, 0.345s)
│   └── high_branch (completed, 0.111s)
└── final_step (completed, 0.234s)
```

### List Individual Spans

```bash
flujo lens spans <run_id>

# Filter by status
flujo lens spans <run_id> --status completed

# Filter by name
flujo lens spans <run_id> --name step1
```

Shows all spans with filtering options:
- `--status completed` - Only show completed spans
- `--name step1` - Only show spans with specific name

### View Statistics

```bash
flujo lens stats

# Filter by pipeline
flujo lens stats --pipeline my_pipeline

# Specify time range
flujo lens stats --hours 48
```

Shows aggregated statistics:
- Total spans
- Status breakdown (completed/failed/running)
- Average duration by step name
- Step count by name

## Advanced Usage

### Custom Pipeline with Tracing

```python
from flujo import Pipeline, Step, Flujo
from flujo.domain.models import PipelineContext

async def simple_step(input_data: str, context: PipelineContext) -> str:
    return f"processed_{input_data}"

async def another_step(input_data: str, context: PipelineContext) -> str:
    return f"enhanced_{input_data}"

# Create pipeline
pipeline = Pipeline(steps=[
    Step.from_callable(simple_step, name="step1"),
    Step.from_callable(another_step, name="step2"),
])

# Run with tracing
flujo = Flujo(pipeline=pipeline, enable_tracing=True)
async for result in flujo.run_async("test_input"):
    pass

# Analyze trace
if result.trace_tree:
    print(f"Pipeline completed in {result.trace_tree.end_time - result.trace_tree.start_time:.3f}s")

    for child in result.trace_tree.children:
        print(f"  {child.name}: {child.status} ({child.end_time - child.start_time:.3f}s)")
```

### Error Handling

The tracing system gracefully handles errors:

```python
async def failing_step(input_data: str, context: PipelineContext) -> str:
    raise ValueError("Intentional failure")

pipeline = Pipeline(steps=[
    Step.from_callable(failing_step, name="failing_step"),
])

flujo = Flujo(pipeline=pipeline, enable_tracing=True)
async for result in flujo.run_async("test_input"):
    pass

# Even failed pipelines generate traces
if result.trace_tree:
    failed_step = None
    for child in result.trace_tree.children:
        if child.name == "failing_step":
            failed_step = child
            break

    if failed_step and failed_step.status == "failed":
        print(f"Step failed: {failed_step.attributes.get('feedback', 'Unknown error')}")
```

### Performance Analysis

Use traces to identify performance bottlenecks:

```python
# After running a pipeline
if result.trace_tree:
    # Find the slowest step
    slowest_step = max(result.step_history, key=lambda s: s.latency_s)
    print(f"Slowest step: {slowest_step.name} ({slowest_step.latency_s:.3f}s)")

    # Find failed steps
    failed_steps = [s for s in result.step_history if not s.success]
    if failed_steps:
        print(f"Failed steps: {[s.name for s in failed_steps]}")
```

## Best Practices

### 1. Use Descriptive Step Names

```python
# Good
Step.from_callable(process_data, name="data_processing")

# Avoid
Step.from_callable(process_data, name="step1")
```

### 2. Handle Trace Access Gracefully

```python
if result.trace_tree:
    # Access trace information
    pass
else:
    # Tracing might be disabled or failed
    print("No trace information available")
```

### 3. Use CLI for Complex Analysis

For large pipelines, use the CLI tools instead of programmatic access:

```bash
# Find failed runs
flujo lens list --status failed

# Quick search by partial ID
flujo lens get abc123

# Export run data as JSON for analysis
flujo lens show <run_id> --json | jq '.steps[] | {name: .step_name, duration: .execution_time_ms}'

# Get statistics
flujo lens stats --hours 24
```

**New in v2.0**: 
- Partial ID matching makes debugging 5x faster
- JSON output enables powerful automation
- Better error messages guide you to solutions
- See [Lens Quick Start](lens_quickstart.md) for more examples

### 4. Monitor Performance Overhead

Tracing adds minimal overhead (< 50% increase), but monitor in production:

```python
import time

# Test without tracing
start = time.time()
flujo_no_trace = Flujo(pipeline=pipeline, enable_tracing=False)
# ... run pipeline
no_trace_time = time.time() - start

# Test with tracing
start = time.time()
flujo_with_trace = Flujo(pipeline=pipeline, enable_tracing=True)
# ... run pipeline
with_trace_time = time.time() - start

overhead = (with_trace_time / no_trace_time - 1) * 100
print(f"Tracing overhead: {overhead:.1f}%")
```

## Troubleshooting

### No Trace Information

If `result.trace_tree` is `None`:
1. Check that tracing is enabled: `Flujo(pipeline, enable_tracing=True)`
2. Verify the pipeline completed (even if it failed)
3. Check for any trace serialization errors in logs

### CLI Commands Not Working

If CLI commands fail:
1. Ensure you have a SQLite database: `flujo_ops.db`
2. Check that the run_id exists: `flujo lens list`
3. Verify database permissions and connectivity
4. **New**: Try partial ID matching instead of full run_id
5. **New**: Increase timeout if needed: `flujo lens show <run_id> --timeout 30`
6. See [Lens Quick Start](lens_quickstart.md) for troubleshooting tips

### Performance Issues

If tracing causes performance problems:
1. Monitor overhead with the test above
2. Consider disabling tracing for high-frequency pipelines
3. Use filtering in CLI commands to reduce data transfer

## Integration with Other Tools

### SQLite Backend

Traces are automatically persisted to the SQLite backend:
- `spans` table stores hierarchical trace data
- `runs` table stores pipeline metadata
- Audit logging tracks trace access

### OTLP Export

For production environments, enable OTLP export:
```bash
export OTLP_EXPORT_ENABLED=true
export OTLP_ENDPOINT=https://your-otlp-endpoint
```

This sends traces to external observability platforms like:
- OpenTelemetry Collector
- Honeycomb
- Datadog
- Jaeger

## Related Documentation

- **[Lens Quick Start](lens_quickstart.md)** - Quick reference for lens CLI commands
- **[Lens Improvements](lens_improvements.md)** - Complete guide to lens v2.0 improvements
- **[Debugging with Replay](debugging_with_replay.md)** - Deterministic replay for debugging

## Examples

See the following examples for practical usage:
- `examples/test_tracing_demo.py` - Simple tracing demo
- `examples/fsd_12_tracing_demo.py` - Comprehensive FSD-12 showcase
- `tests/integration/test_fsd_12_tracing_complete.py` - Complete test suite
