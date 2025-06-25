# Testing Guide

This guide provides recommended patterns and utilities for writing unit and integration tests for `flujo` pipelines, steps and agents.

## Unit Testing Pipelines with `StubAgent`

Use `flujo.testing.utils.StubAgent` to replace real agents with predictable canned outputs. Provide a list of outputs and the stub will return them in order:

```python
from flujo import Step, Flujo
from flujo.testing.utils import StubAgent, gather_result

pipeline = Step("review", StubAgent(["ok"])) >> Step("solve", StubAgent(["answer"]))
runner = Flujo(pipeline)
result = await gather_result(runner, "prompt")
assert result.step_history[-1].output == "answer"
```

## Testing Plugins with `DummyPlugin`

`DummyPlugin` simulates plugin behaviour. You can configure each call to pass or fail by providing a sequence of `PluginOutcome` objects:

```python
from flujo import Step, Flujo
from flujo.domain.plugins import PluginOutcome
from flujo.testing.utils import StubAgent, DummyPlugin, gather_result

plugin = DummyPlugin([
    PluginOutcome(success=False, feedback="bad"),
    PluginOutcome(success=True),
])
step = Step.solution(StubAgent(["first", "fixed"]), plugins=[plugin], max_retries=2)
runner = Flujo(step)
result = await gather_result(runner, "input")
assert step.agent.call_count == 2
assert result.step_history[0].attempts == 2
```

## Testing Decorated `@step` Functions

Functions decorated with `@step` return a `Step` object. When testing them in isolation you can execute the step directly using `Step.arun()` if available, or by wrapping the step in a `Flujo` runner:

```python
from flujo import step, Flujo
from flujo.testing.utils import gather_result

@step
async def add_one(x: int) -> int:
    return x + 1

# Recommended helper when available
result = await add_one.arun(1)
assert result == 2

# Fallback: run through the engine
# result = await gather_result(Flujo(add_one), 1)
```

## Typed `PipelineContext`

When using a typed context, pass `context_model` and `initial_context_data` to the runner and assert the final context state:

```python
from pydantic import BaseModel
from flujo import Step, Flujo
from flujo.testing.utils import StubAgent, gather_result

class Ctx(BaseModel):
    count: int = 0

agent = StubAgent([1, 2])
step1 = Step("inc1", agent)
step2 = Step("inc2", agent)
runner = Flujo(step1 >> step2, context_model=Ctx, initial_context_data={"count": 0})
result = await gather_result(runner, 0)
assert result.final_pipeline_context.count == 2
```

## Testing Steps that Use `AppResources`

Shared resources can be injected into agents and plugins. In tests, pass mock resources to `Flujo` and verify interactions:

```python
from unittest.mock import MagicMock
from flujo import Step, Flujo, AppResources
from flujo.testing.utils import StubAgent, gather_result

class MyResources(AppResources):
    db: MagicMock

class WriteAgent:
    async def run(self, data: str, *, resources: MyResources) -> str:
        resources.db.write(data)
        return "done"

resources = MyResources(db=MagicMock())
pipeline = Step("write", WriteAgent())
runner = Flujo(pipeline, resources=resources)
await gather_result(runner, "record")
resources.db.write.assert_called_once_with("record")
```

## Common Pitfalls

### Unconfigured mocks

When a mocked agent returns the default `Mock` object, the engine will raise:

```text
TypeError: Step 'my_step' returned a Mock object. This is usually due to an unconfigured mock in a test. Please configure your mock agent to return a concrete value.
```

Configure your mock to return a value:

```python
from unittest.mock import AsyncMock

my_agent = AsyncMock()
my_agent.run.return_value = "expected"
```

See the [Troubleshooting Guide](troubleshooting.md) for more details.

## Testing Individual Steps

Use the :meth:`Step.arun` method to exercise a single step in isolation. This
executes the step's underlying agent without any orchestration logic.

```python
from my_app.steps import process_data_step  # a @step decorated object
import asyncio

async def test_process_data_step() -> None:
    input_data = "some raw data"
    expected_output = "SOME RAW DATA"

    result = await process_data_step.arun(input_data)

    assert result == expected_output
```

