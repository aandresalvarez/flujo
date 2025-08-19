# Testing Guide

This guide explains how to write reliable unit and integration tests for `flujo` pipelines. It highlights the built in utilities found in `flujo.testing.utils` and showcases patterns for testing steps, pipelines, and resources.

## 1. Unit Testing Pipelines with `StubAgent`

`StubAgent` lets you replace real agents with predictable canned outputs. Provide a list of responses and the stub will return them sequentially whenever `run()` is called.

```python
from flujo import Flujo, Step
from flujo.testing.utils import StubAgent

# Two step pipeline that normally calls real agents
pipeline = Step("draft", StubAgent(["First draft: Hello world", "Second draft: Hello, world!"])) >> Step("review", StubAgent(["APPROVED"]))

async def test_pipeline() -> None:
    runner = Flujo(pipeline)
    result = await runner.arun("hello world")
    assert result.step_history[-1].output == "APPROVED"
    assert pipeline.steps[0].agent.inputs == ["hello world"]
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
    PluginOutcome(success=False, feedback="Invalid JSON format"),
    PluginOutcome(success=True),
])
step = Step("validate", StubAgent(["Fixed JSON: {'name': 'John'}", "Validated: {'name': 'John', 'age': 30}"]), plugins=[plugin])

async def test_plugin_step() -> None:
    runner = Flujo(step)
    result = await runner.arun("data")
    assert plugin.call_count == 2
    assert result.step_history[0].output == "Validated: {'name': 'John', 'age': 30}"
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
from flujo import Flujo, Step, step
from flujo.domain.models import PipelineContext
from flujo.testing.utils import StubAgent

class Ctx(PipelineContext):
    counter: int = 0

@step
async def increment(x: int, *, context: Ctx) -> int:
    context.counter += 1
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

## 6. Testing Application Code with `override_agent`

The `override_agent` context manager provides a clean way to test application code that uses `flujo` pipelines internally. This is especially useful when you want to test your application logic without running expensive or slow production agents.

### Basic Usage

```python
from flujo import Step, Pipeline
from flujo.testing import override_agent, StubAgent

class ProductionAgent:
    """A production agent that might be expensive or slow to run."""

    async def run(self, data: str, **kwargs) -> str:
        # Simulate expensive operation
        await asyncio.sleep(0.1)
        return f"expensive_result: {data.upper()}"

class ApplicationService:
    """Example application service that uses flujo pipelines internally."""

    def __init__(self):
        self.pipeline = (
            Step("Process", ProductionAgent()) >>
            Step("Validate", ProductionAgent())
        )
        self.runner = Flujo(self.pipeline)

    async def process_data(self, data: str) -> str:
        """Process data using the internal pipeline."""
        result = None
        async for item in self.runner.run_async(data):
            result = item
        return result.step_history[-1].output

# Test the application service with overridden agents
async def test_application_service():
    service = ApplicationService()
    fast_test_agent = StubAgent(["Processed: test_input", "Validated: Processed: test_input"])

    # Override both steps in the pipeline
    with override_agent(service.pipeline.steps[0], fast_test_agent):
        with override_agent(service.pipeline.steps[1], fast_test_agent):
            result = await service.process_data("test_input")
            assert result == "Validated: Processed: test_input"
```

### Testing Different Scenarios

You can use `override_agent` to test different scenarios without modifying your application code:

```python
async def test_different_scenarios():
    service = ApplicationService()

    # Test success scenario
    success_agent = StubAgent(["Successfully processed: test"])
    with override_agent(service.pipeline.steps[0], success_agent):
        result = await service.process_data("test")
        assert result == "Successfully processed: test"

    # Test failure scenario
    failure_agent = StubAgent([RuntimeError("Test failure")])
    with override_agent(service.pipeline.steps[0], failure_agent):
        try:
            await service.process_data("test")
            assert False, "Expected exception"
        except RuntimeError as e:
            assert str(e) == "Test failure"
```

### Automatic Cleanup

The context manager automatically restores the original agent when the `with` block exits, even if an exception occurs:

```python
async def test_agent_restoration():
    original_agent = ProductionAgent()
    step = Step("test", original_agent)

    # Verify original agent is set
    assert step.agent is original_agent

    # Use context manager and raise an exception
    try:
        with override_agent(step, StubAgent(["test"])):
            assert step.agent is not original_agent
            raise RuntimeError("Test exception")
    except RuntimeError:
        pass

    # Verify original agent is still restored
    assert step.agent is original_agent
```

### Benefits of `override_agent`

1. **Fast Execution**: No expensive operations during testing
2. **Predictable Outputs**: Use `StubAgent` for controlled responses
3. **Automatic Cleanup**: Original agents are restored automatically
4. **Exception Safety**: Agents restored even if tests fail
5. **Simple Syntax**: Clean context manager interface
6. **No Code Changes**: Test application code without modifying it

## 7. Assertion Utilities

`flujo` provides helpful assertion utilities to simplify testing of pipeline behavior, especially when dealing with validation results and context updates.

### `assert_validator_failed`

Asserts that a specific validator failed during a pipeline run. This is useful for testing scenarios where you expect a validation step to produce a failure.

```python
from flujo import Flujo, Step
from flujo.testing.utils import StubAgent
from flujo.domain.validation import validator

@validator
def always_fail_validator(output: str) -> tuple[bool, str | None]:
    return False, "This validator always fails!"

pipeline = (
    Step.solution(StubAgent(["Generated content: Hello world"]))
    >> Step.validate(validators=[always_fail_validator])
)

async def test_validator_failure():
    runner = Flujo(pipeline)
    result = await runner.arun("input")

    # Assert that the 'always_fail_validator' failed
    assert_validator_failed(result, "always_fail_validator", "This validator always fails!")
```

### `assert_context_updated`

Asserts that the final pipeline context contains specific expected updates. This is useful for testing that your steps correctly modify the shared pipeline context.

```python
from flujo import Flujo, Step
from flujo.domain.models import PipelineContext
from flujo.testing.utils import StubAgent

class MyContext(PipelineContext):
    my_value: int = 0

@Step
async def increment_context(input_data: str, *, context: MyContext) -> str:
    context.my_value += 1
    return input_data

pipeline = (
    Step.solution(StubAgent(["Initial data: test"]))
    >> increment_context
)

async def test_context_update():
    runner = Flujo(pipeline, context_model=MyContext, initial_context_data={"my_value": 0})
    result = await runner.arun("test")

    # Assert that 'my_value' in the context was updated to 1
    assert_context_updated(result, my_value=1)
```

## 8. Common Pitfalls

If a mocked agent returns the default `Mock` object, the engine raises:

```text
TypeError: Step 'my_step' returned a Mock object. This is usually due to an unconfigured mock in a test.
```

Always set a return value on your mocks. See the [Troubleshooting Guide](troubleshooting.md) for more details.
