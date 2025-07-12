# Cookbook: Error Recovery with Fallback Steps

## The Problem

LLM calls occasionally fail or produce unusable results. You want the pipeline to recover gracefully instead of crashing.

## The Solution

Use `Step.fallback()` to declare a backup step that runs when the primary step fails after its retries are exhausted.

```python
from flujo import Step, Flujo
from flujo.testing.utils import StubAgent

primary = Step("primary", StubAgent(["fail"]), max_retries=1)
backup = Step("backup", StubAgent(["ok"]))
primary.fallback(backup)

runner = Flujo(primary)
result = runner.run("data")
print(result.step_history[0].output)  # -> "ok"
```

`StepResult.metadata_["fallback_triggered"]` will be `True` when the fallback runs successfully.
