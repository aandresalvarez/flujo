# Extending pydantic-ai-orchestrator

## Adding a Custom Agent

```python
from pydantic_ai import Agent
class MyAgent(Agent):
    ...
```

## Customizing the Reflection Agent

While the library provides a default `reflection_agent`, you may want to use a
different model or configuration for the reflection step. Use the factory
function `get_reflection_agent()` to create a custom instance.

```python
from pydantic_ai_orchestrator import (
    Orchestrator,
    review_agent,
    solution_agent,
    validator_agent,
    get_reflection_agent,  # Import the factory
)

custom_reflection_agent = get_reflection_agent(model="anthropic:claude-3-haiku")

orch = Orchestrator(
    review_agent,
    solution_agent,
    validator_agent,
    reflection_agent=custom_reflection_agent,
)
```
