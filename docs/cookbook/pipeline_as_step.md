# Cookbook: Building Modular Workflows with `as_step`

Use the **`as_step`** composition pattern when you need to encapsulate a complex sub-workflow into a single, reusable step within a larger pipeline. This pattern enables you to build hierarchical, modular workflows that are easier to understand, test, and maintain.

## The Problem

As your AI workflows grow in complexity, you face the challenge of managing that complexity. A monolithic pipeline with dozens of steps becomes difficult to read, test, and maintain. You need a way to break down large workflows into smaller, reusable components while maintaining the benefits of unified execution and observability.

## The Solution

The `Flujo.as_step()` method allows you to wrap an entire `Flujo` runner into a single step that can be used within another pipeline. This creates a "pipeline of pipelines" pattern that promotes modularity and reusability.

## Basic Example: Pipeline of Pipelines

Here's a simple example showing how to compose pipelines using `as_step`:

```python
from flujo import Flujo, Step
from flujo.testing.utils import StubAgent

# Create a sub-pipeline that processes text
text_processor = StubAgent(["processed text"])
sub_pipeline = Step("process", text_processor)

# Create a Flujo runner for the sub-pipeline
sub_runner = Flujo(sub_pipeline, context_model=PipelineContext)

# Wrap the sub-pipeline as a step
sub_step = sub_runner.as_step(name="text_processor")

# Create a master pipeline that uses the sub-pipeline
master_pipeline = sub_step >> Step("finalize", StubAgent(["final result"]))
master_runner = Flujo(master_pipeline, context_model=PipelineContext)

# Run the master pipeline
result = await master_runner.run_async("input text")
```

## Handling State: Context Propagation

One of the key benefits of `as_step` is that the `PipelineContext` from the outer pipeline is automatically passed to the inner pipeline. This means state flows seamlessly between your master pipeline and its sub-pipelines.

```python
from flujo import Flujo, Step
from flujo.domain.models import PipelineContext

class Incrementer:
    async def run(self, data: int, *, context: PipelineContext | None = None) -> dict:
        assert context is not None
        current = context.scratchpad.get("counter", 0)
        return {"scratchpad": {"counter": current + data}}

# Create a sub-pipeline that updates context
inner_runner = Flujo(
    Step("inc", Incrementer(), updates_context=True),
    context_model=PipelineContext,
)

# Wrap it as a step
pipeline = inner_runner.as_step(name="inner")
runner = Flujo(pipeline, context_model=PipelineContext)

# Run with initial context
result = await runner.run_async(
    2,
    initial_context_data={
        "initial_prompt": "goal",
        "scratchpad": {"counter": 1}
    }
)

# The context is automatically propagated
assert result.final_pipeline_context.scratchpad["counter"] == 3
```

## Context Firewall: `inherit_context=False`

Sometimes you want to isolate a sub-pipeline's context to prevent unintended side effects. Use `inherit_context=False` to create a context firewall:

```python
class Incrementer:
    async def run(self, data: int, *, context: PipelineContext | None = None) -> dict:
        assert context is not None
        current = context.scratchpad.get("counter", 0)
        return {"scratchpad": {"counter": current + data}}

inner_runner = Flujo(
    Step("inc", Incrementer(), updates_context=True),
    context_model=PipelineContext,
)

# Create an isolated sub-pipeline
pipeline = inner_runner.as_step(name="inner", inherit_context=False)
runner = Flujo(pipeline, context_model=PipelineContext)

result = await runner.run_async(
    2,
    initial_context_data={
        "initial_prompt": "goal",
        "scratchpad": {"counter": 1}
    }
)

# The context is NOT propagated - counter remains 1
assert result.final_pipeline_context.scratchpad["counter"] == 1
```

## Managing Shared Resources

`AppResources` are seamlessly passed through to nested pipelines, allowing you to share database connections, API clients, and other long-lived resources:

```python
from flujo import Flujo, Step, AppResources
from flujo.domain.models import PipelineContext

class Res(AppResources):
    counter: int = 0

class UseRes:
    async def run(self, data: int, *, resources: Res) -> int:
        resources.counter += data
        return resources.counter

# Create a sub-pipeline that uses shared resources
inner_runner = Flujo(
    Step("res", UseRes()),
    context_model=PipelineContext,
)

# Wrap it as a step
pipeline = inner_runner.as_step(name="inner")
res = Res()
runner = Flujo(pipeline, context_model=PipelineContext, resources=res)

await runner.run_async(5, initial_context_data={"initial_prompt": "goal"})

# The shared resource is updated
assert res.counter == 5
```

## Durability and Crash Recovery

`as_step` works seamlessly with state persistence and resumption. When a nested pipeline crashes, you can resume execution and it will correctly continue within the sub-pipeline:

```python
from flujo import Flujo, Step
from flujo.domain.models import PipelineContext
from flujo.state.backends.sqlite import SQLiteBackend

# Create a sub-pipeline with state persistence
inner_pipeline = (
    Step.from_callable(lambda x: x + 1, name="first") >>
    Step.from_callable(lambda x: x + 1, name="second")
)

inner_runner = Flujo(
    inner_pipeline,
    context_model=PipelineContext,
    state_backend=SQLiteBackend(Path("state.db")),
    delete_on_completion=False,
)

# Wrap it as a step
nested = inner_runner.as_step(name="nested", inherit_context=False)

# Create a master pipeline
outer_pipeline = (
    Step.from_callable(lambda x: x + 1, name="outer_start") >>
    nested >>
    Step.from_callable(lambda x: x + 1, name="outer_end")
)

outer_runner = Flujo(outer_pipeline, context_model=PipelineContext)

# Run the pipeline
result = await outer_runner.run_async(0, initial_context_data={"initial_prompt": "start"})

# The nested pipeline result is preserved
inner_result = result.step_history[1].output
assert isinstance(inner_result, PipelineResult)
assert len(inner_result.step_history) == 2
assert inner_result.step_history[0].output == 1
assert inner_result.step_history[1].output == 2
```

## Best Practices and When to Use `as_step`

### When to Use `as_step`

- **Modularity**: Break large, complex pipelines into smaller, focused components
- **Reusability**: Create pipeline components that can be reused across different workflows
- **Testing**: Test sub-pipelines in isolation before composing them
- **Team Development**: Different team members can work on different sub-pipelines
- **Complexity Management**: Hide implementation details behind a clean interface

### When NOT to Use `as_step`

- **Simple Pipelines**: For pipelines with just a few steps, `as_step` adds unnecessary complexity
- **Sequential Operations**: If you just want to chain operations, use the `>>` operator directly
- **Performance Critical**: There's a small overhead to the `as_step` wrapper

### Naming and Organization

```python
# Good: Clear, descriptive names
data_processing_pipeline = Flujo(data_steps, context_model=PipelineContext)
data_step = data_processing_pipeline.as_step(name="data_processing")

# Good: Logical grouping
validation_pipeline = Flujo(validation_steps, context_model=PipelineContext)
validation_step = validation_pipeline.as_step(name="validation")

# Master pipeline composition
master_pipeline = data_step >> validation_step >> output_step
```

### Testing Sub-Pipelines

Test your sub-pipelines in isolation before composing them:

```python
import pytest

@pytest.mark.asyncio
async def test_data_processing_pipeline():
    data_runner = Flujo(data_steps, context_model=PipelineContext)
    result = await data_runner.run_async("test input")
    assert result.step_history[-1].output == "expected output"

@pytest.mark.asyncio
async def test_master_pipeline_with_sub_pipelines():
    data_step = data_runner.as_step(name="data")
    validation_step = validation_runner.as_step(name="validation")
    master_pipeline = data_step >> validation_step

    master_runner = Flujo(master_pipeline, context_model=PipelineContext)
    result = await master_runner.run_async("test input")
    assert result.step_history[-1].output == "final result"
```

## Related Guides

- [Pipeline DSL Guide](../pipeline_dsl.md) - Learn about the core pipeline composition patterns
- [Durable Workflows](../guides/durable_workflows.md) - Understand state persistence and crash recovery
- [Pipeline Context](../pipeline_context.md) - Learn about context management and state propagation
- [The Flujo Way](../The_flujo_way.md) - Discover the core principles and patterns of Flujo
