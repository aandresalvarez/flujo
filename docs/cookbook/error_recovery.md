# Cookbook: Error Recovery with Fallback Steps

## The Problem

LLM calls occasionally fail or produce unusable results. You want the pipeline to recover gracefully instead of crashing.

## The Solution

Declare a backup step with the `fallback` argument so it runs when the primary step fails after its retries are exhausted.

```python
from flujo import step, Flujo
from flujo.testing.utils import StubAgent

@step
async def backup(x: str) -> str:
    return "ok"

@step(retries=1, fallback=backup)
async def primary(x: str) -> str:
    return "fail"

runner = Flujo(primary)
result = runner.run("data")
print(result.step_history[0].output)  # -> "ok"
```

`StepResult.metadata_["fallback_triggered"]` will be `True` when the fallback runs successfully.

