# Managing Pipeline Versions

Deploying new code while old workflows are still running can lead to inconsistencies. `Flujo` solves this with a simple registry that stores named pipelines and their versions.

## Introduction

Register each pipeline with a unique name and semantic version. When a durable run resumes it will load the exact version it started with, even if newer versions are deployed.

## How It Works

```python
from flujo import Step, PipelineRegistry, Flujo, step

@step
async def v1(text: str) -> str:
    return text.upper()

@step
async def v2(text: str) -> str:
    return text.lower()

pipeline_v1 = v1
pipeline_v2 = v2

registry = PipelineRegistry()
registry.register(pipeline_v1, "example", "1.0.0")
registry.register(pipeline_v2, "example", "2.0.0")

runner = Flujo(registry=registry, pipeline_name="example", pipeline_version="1.0.0")
```

When the runner starts a durable workflow it records the version in the saved `WorkflowState`. If you later register `2.0.0` and resume the run, the registry will still return `1.0.0` for that run.

## Practical Example: A Safe Deployment

```python
# Start v1
run_id = "deploy-demo"
runner = Flujo(
    registry=registry,
    pipeline_name="example",
    pipeline_version="1.0.0",
    state_backend=SQLiteBackend("state.db"),
)
partial = None
async for item in runner.run_async("DATA", run_id=run_id):
    partial = item
    break  # Simulate restart after first step

# Register an incompatible v2
registry.register(pipeline_v2, "example", "2.0.0")

# Resume the original run
runner_resume = Flujo(
    registry=registry,
    pipeline_name="example",
    pipeline_version="latest",
    state_backend=SQLiteBackend("state.db"),
)
final = None
async for item in runner_resume.run_async("DATA", run_id=run_id):
    final = item

assert final.pipeline_version == "1.0.0"  # Still uses v1
```

This pattern allows you to deploy new code without interrupting in-flight runs.

See also: [Durable and Resumable Workflows](durable_workflows.md).
