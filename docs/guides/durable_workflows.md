# Durable and Resumable Workflows

Running a long or important workflow can feel a bit like playing an old video game – if the power goes out you lose all your progress. `Flujo` avoids this problem by saving the state of a pipeline after each successful step. When your process restarts, the run can pick up exactly where it left off.

## Why Durability Matters

Servers restart, deployments happen and sometimes a workflow simply has to wait for human input. By persisting the current step index and context, a pipeline can safely resume instead of starting over.

## The StateBackend Interface

Flujo uses a pluggable backend interface for persistence. Every backend implements the same three methods:

```python
from flujo.state.backends.base import StateBackend

class StateBackend(ABC):
    """Abstract interface for workflow state persistence."""

    async def save_state(self, run_id: str, state: Dict[str, Any]) -> None:
        ...
    async def load_state(self, run_id: str) -> Optional[Dict[str, Any]]:
        ...
    async def delete_state(self, run_id: str) -> None:
        ...
```

## WorkflowState Model

The runner serializes a `WorkflowState` object which contains the pipeline version, step index and JSON‑serializable context.

```python
from flujo.state.models import WorkflowState

WorkflowState(
    run_id: str,
    pipeline_id: str,
    pipeline_name: str,
    pipeline_version: str,
    current_step_index: int,
    pipeline_context: Dict[str, Any],
    last_step_output: Any | None,
    status: Literal["running", "paused", "completed", "failed", "cancelled"],
    created_at: datetime,
    updated_at: datetime,
)
```

## Execution Lifecycle

1. A `WorkflowState` is created after the first step completes.
2. The state is updated after every successful step.
3. On completion or failure the final status is recorded.

## Getting Started

Here is a minimal example using the `SQLiteBackend`:

```python
from pathlib import Path
from flujo import Flujo, Step, PipelineRegistry, step
from flujo.state import SQLiteBackend

@step
async def greet(name: str) -> str:
    return f"Hello, {name}!"

pipeline = greet
registry = PipelineRegistry()
registry.register(pipeline, "hello", "1.0.0")
backend = SQLiteBackend(Path("workflow.db"))

runner = Flujo(
    registry=registry,
    pipeline_name="hello",
    pipeline_version="1.0.0",
    state_backend=backend,
)
result = runner.run("world", run_id="run1")

# To resume later just create a new runner with the same ``run_id``
runner2 = Flujo(
    registry=registry,
    pipeline_name="hello",
    pipeline_version="1.0.0",
    state_backend=backend,
)
resumed = runner2.run("world", run_id="run1")
```

## Built-in Backends

### InMemoryBackend

For tests and demos. Data lives only in memory so everything is lost on restart.
The backend uses the same serialization helpers as the file and SQLite
backends, ensuring custom types round-trip consistently.

### FileBackend

Stores each run as a JSON file in a directory. Great for simple serverless setups but not safe for concurrent writes.

### SQLiteBackend

Persists state in a single SQLite database file. Robust enough for many production scenarios and perfect for local development. For comprehensive documentation including admin queries, performance optimization, and operational best practices, see the [SQLite Backend Guide](sqlite_backend_guide.md).

See also: [Managing Pipeline Versions](pipeline_versioning.md).
