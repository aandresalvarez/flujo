# Building Resilient Pipelines with Fallbacks

The `Step.fallback()` method lets you declare a backup step that runs if the primary step fails.
This is useful for handling transient errors or providing a simpler model when a complex one is unreliable.

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

When the fallback runs successfully, `StepResult.metadata_['fallback_triggered']` is set to `True` and the pipeline continues normally.
