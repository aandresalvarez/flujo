# Usage

## CLI

```bash
orch solve "Write a summary of this document."
orch show-config
orch bench --prompt "hi" --rounds 3
orch --profile
```

## API

```python
from pydantic_ai_orchestrator import Orchestrator
orch = Orchestrator()
result = orch.run_sync("Write a poem.")
print(result)
```

## Environment Variables

- `ORCH_OPENAI_API_KEY` (required)
- `ORCH_LOGFIRE_API_KEY` (optional)
- `ORCH_REFLECTION_ENABLED` (default: true)
- `ORCH_MAX_ITERS`, `ORCH_K_VARIANTS`

## Scoring
- Ratio and weighted scoring supported
- Reward model stub included (extendable)

## Reflection
- Reflection agent can be toggled via `ORCH_REFLECTION_ENABLED`

## Weighted Scoring

You can provide weights for checklist items to customize the scoring logic. This is useful when some criteria are more important than others.

Provide the weights via the `Task` metadata:

```python
from pydantic_ai_orchestrator import Orchestrator, Task

orch = Orchestrator(...)
task = Task(
    prompt="Generate a Python class.",
    metadata={
        "weights": [
            {"item": "Has a docstring", "weight": 0.7},
            {"item": "Includes type hints", "weight": 0.3},
        ]
    }
)
result = orch.run_sync(task)
```
The `scorer` setting must be set to `"weighted"`.