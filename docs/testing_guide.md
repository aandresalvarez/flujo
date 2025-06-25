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
