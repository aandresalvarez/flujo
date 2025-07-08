# Durable and Resumable Workflows

Long-running or mission critical tasks should be able to survive a crash or restart. Think of it like a video game that lets you **save your progress**. Flujo solves this with a pluggable *StateBackend* that stores the workflow state after each step.

## Conceptual Overview

### The `StateBackend` Interface
```python
class StateBackend(ABC):
    async def save_state(self, run_id: str, state: Dict[str, Any]) -> None: ...
    async def load_state(self, run_id: str) -> Optional[Dict[str, Any]]: ...
    async def delete_state(self, run_id: str) -> None: ...
```
A backend can persist to memory, files, databases, or any other system.

### The `WorkflowState` Model
Flujo stores a serialized `WorkflowState` containing the `run_id`, pipeline version, current step index and the JSON-serializable pipeline context.

### The Execution Lifecycle
1. A unique `run_id` is attached to the `PipelineContext`.
2. After each successful step Flujo saves the state via the backend.
3. On startup Flujo checks for an existing state and resumes from the saved step.

## Getting Started: A Simple Durable Run

Below is a minimal example using the `SQLiteBackend`.
```python
from pathlib import Path
from flujo import Flujo, PipelineRegistry, step, Step
from flujo.state.backends.sqlite import SQLiteBackend

@step
async def to_upper(text: str) -> str:
    return text.upper()

pipeline = to_upper >> Step.human_in_the_loop("approve", message_for_user="Approve?")

registry = PipelineRegistry()
registry.register(pipeline, name="demo", version="1.0.0")
backend = SQLiteBackend(Path("workflow_state.db"))

# First run will pause for approval and persist state
runner = Flujo(
    registry=registry,
    pipeline_name="demo",
    pipeline_version="1.0.0",
    state_backend=backend,
)
paused = runner.run("hello", initial_context_data={"run_id": "example"})

# Later we create a new runner with the same run_id and resume
resumer = Flujo(
    registry=registry,
    pipeline_name="demo",
    pipeline_version="1.0.0",
    state_backend=backend,
)
final = resumer.resume_async(paused, "yes")
```

## Built-in Backends

### `InMemoryBackend`
For testing and demos. Data lives only in memory and is lost when the process exits.

### `FileBackend`
Persists state as JSON files. Good for simple serverless environments. Concurrency is limited.

### `SQLiteBackend`
Stores state in a robust SQLite database. Suitable for many production setups with a single file.
