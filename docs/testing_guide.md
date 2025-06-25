# Testing Guide

This guide covers best practices for testing `flujo` pipelines.

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
