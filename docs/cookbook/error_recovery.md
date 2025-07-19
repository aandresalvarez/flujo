# Cookbook: Error Recovery with Fallback Steps

## The Problem

LLM calls occasionally fail or produce unusable results. You want the pipeline to recover gracefully instead of crashing.

## The Solution

Use `Step.fallback()` to declare a backup step that runs when the primary step fails after its retries are exhausted.

```python
from flujo import Step, Flujo
from flujo.testing.utils import StubAgent

# Primary step that fails after retries
class FailingAgent:
    async def run(self, data: str, **kwargs) -> str:
        raise RuntimeError("API rate limit exceeded")

primary = Step("primary", FailingAgent(), max_retries=1)
# Backup step that provides a simpler, more reliable solution
backup = Step("backup", StubAgent(["Fallback result: Simplified analysis completed"]))
primary.fallback(backup)

runner = Flujo(primary)
result = runner.run("data")
print(result.step_history[0].output)  # -> "Fallback result: Simplified analysis completed"
```

`StepResult.metadata_["fallback_triggered"]` will be `True` when the fallback runs successfully.
