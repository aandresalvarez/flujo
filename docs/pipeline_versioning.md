# Managing Pipeline Versions

When you deploy new code while old workflows are still running you risk breaking in-flight runs. The `PipelineRegistry` solves this by tracking multiple versions of each pipeline.

## How It Works
Register each pipeline with a unique name and version. We recommend [Semantic Versioning](https://semver.org/).
```python
from flujo import PipelineRegistry, Step

registry = PipelineRegistry()
registry.register(Step.from_mapper(str.upper), name="demo", version="1.0.0")
registry.register(Step.from_mapper(lambda x: x + "!"), name="demo", version="2.0.0")
```

A `Flujo` runner is created with a registry, a pipeline name and a version:
```python
runner = Flujo(registry=registry, pipeline_name="demo", pipeline_version="1.0.0")
```
When resuming a durable workflow Flujo always loads the exact version that started the run, ensuring stability.

## Practical Example: A Safe Deployment

```python
from flujo import PipelineRegistry, Flujo, step
from flujo.state.backends.sqlite import SQLiteBackend

@step
async def version_one(text: str) -> str:
    return text.upper()

@step
async def version_two(text: str) -> str:
    return text[::-1]

registry = PipelineRegistry()
registry.register(version_one, "demo", "1.0.0")
backend = SQLiteBackend("workflow_state.db")
runner = Flujo(registry=registry, pipeline_name="demo", pipeline_version="1.0.0", state_backend=backend)
paused = runner.run("hi", initial_context_data={"run_id": "safe"})

# Deploy new, incompatible version
registry.register(version_two, "demo", "2.0.0")

resumer = Flujo(registry=registry, pipeline_name="demo", pipeline_version="1.0.0", state_backend=backend)
final = resumer.resume_async(paused, "continue")
```
This run completes using the original `1.0.0` logic even though `2.0.0` is now registered.
