# Building Middleware with `TaskClient`

The `flujo.client.TaskClient` module turns the internal workflow state backend into a stable
API for middleware, dashboards, or background workers. It wraps the configured state backend
(`flujo.toml` or `FLUJO_STATE_URI`) and exposes typed DTOs so that you never touch database
tables or private models directly.

## Quick start

```python
from flujo.client import TaskClient, TaskStatus
from flujo.domain.dsl import Pipeline, Step

# Reuse your configured backend automatically
client = TaskClient()

# Filter by pipeline name, status, or arbitrary metadata JSON (needs Postgres for server-side filters)
paused = await client.list_tasks(
    status=TaskStatus.PAUSED,
    metadata_filter={"batch_id": "Batch-101"},
)

if not paused:
    return

run_id = paused[0].run_id
detail = await client.get_task(run_id)
print(detail.last_prompt)  # HITL pause message from context.pause_message

# Resume by reusing your pipeline object
pipeline = Pipeline.from_step(Step.human_in_the_loop("Approval"))
final_result = await client.resume_task(run_id, pipeline, input_data={"approved": True})
```

## System-wide markers

Use the same state backend for connector watermarks or other process-wide markers. Both
SQLite and Postgres implementations expose a persistent key/value store.

```python
await client.set_system_state("customers:eu", {"cursor": "evt_123"})
marker = await client.get_system_state("customers:eu")
```

The returned object is a typed `SystemState` DTO with `key`, `value`, and `updated_at`.
