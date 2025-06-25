# Testing Guide

This guide explains how to write reliable unit and integration tests for `flujo` pipelines. It highlights the built in utilities found in `flujo.testing.utils` and showcases patterns for testing steps, pipelines, and resources.

## 1. Unit Testing Pipelines with `StubAgent`

`StubAgent` lets you replace real agents with predictable canned outputs. Provide a list of responses and the stub will return them sequentially whenever `run()` is called.

```python
from flujo import Flujo, Step
from flujo.testing.utils import StubAgent

# Two step pipeline that normally calls real agents
pipeline = Step("draft", StubAgent(["draft-1", "draft-2"])) >> Step("review", StubAgent(["ok"]))

async def test_pipeline() -> None:
    runner = Flujo(pipeline)
    result = await runner.arun("start")
    assert result.step_history[-1].output == "ok"
    assert pipeline.steps[0].agent.inputs == ["start"]
```

Use this pattern to verify branching logic or retry behaviour without making API calls.

## 2. Testing Steps with `DummyPlugin`

`DummyPlugin` simulates validation plugins. Pass a sequence of `PluginOutcome` objects to control whether a step succeeds or fails on each attempt.

```python
from unittest.mock import MagicMock
from flujo import Flujo, Step
from flujo.domain import PluginOutcome
from flujo.testing.utils import StubAgent, DummyPlugin

plugin = DummyPlugin([
    PluginOutcome(success=False, feedback="bad input"),
    PluginOutcome(success=True),
])
step = Step("validate", StubAgent(["fixed", "final"]), plugins=[plugin])

async def test_plugin_step() -> None:
    runner = Flujo(step)
    result = await runner.arun("data")
    assert plugin.call_count == 2
    assert result.step_history[0].output == "final"
```

## 3. Testing Individual Steps

Use the :meth:`Step.arun` method to execute a single step in isolation. This bypasses pipeline orchestration and is ideal for fast unit tests.

```python
from flujo import step

@step
async def uppercase(text: str) -> str:
    return text.upper()

async def test_uppercase() -> None:
    result = await uppercase.arun("hi")
    assert result == "HI"
```

## 4. Pipelines with a Typed `PipelineContext`

When your pipeline uses a context model, provide `initial_context_data` to the runner and assert the `final_pipeline_context` in your test.

```python
from pydantic import BaseModel
from flujo import Flujo, Step, step
from flujo.testing.utils import StubAgent

class Ctx(BaseModel):
    counter: int = 0

@step
async def increment(x: int, *, pipeline_context: Ctx) -> int:
    pipeline_context.counter += 1
    return x + 1

pipeline = Step("a", increment) >> Step("b", increment)

async def test_context_flow() -> None:
    runner = Flujo(pipeline, context_model=Ctx, initial_context_data={"counter": 0})
    result = await runner.arun(1)
    assert result.step_history[-1].output == 3
    assert result.final_pipeline_context.counter == 2
```

## 5. Steps Requiring `AppResources`

Agents and plugins can declare a `resources` dependency. Pass mock resources to the runner and verify interactions.

```python
from unittest.mock import MagicMock
from flujo import Flujo, Step, AppResources

class MyResources(AppResources):
    db: MagicMock

class LookupAgent:
    async def run(self, user_id: int, *, resources: MyResources) -> str:
        return resources.db.get_user(user_id)

async def test_with_resources() -> None:
    resources = MyResources(db=MagicMock())
    resources.db.get_user.return_value = "Alice"
    runner = Flujo(Step("lookup", LookupAgent()), resources=resources)
    result = await runner.arun(1)
    resources.db.get_user.assert_called_once_with(1)
    assert result.step_history[0].output == "Alice"
```

## 6. Common Pitfalls

If a mocked agent returns the default `Mock` object, the engine raises:

```text
TypeError: Step 'my_step' returned a Mock object. This is usually due to an unconfigured mock in a test.
```

Always set a return value on your mocks. See the [Troubleshooting Guide](troubleshooting.md) for more details.
