# Cookbook: Building Resilient Pipelines with Caching and Fallbacks

This guide covers two key strategies for building robust and performant pipelines: caching for performance optimization and fallbacks for error handling.

## Improving Performance with Caching

Expensive or deterministic steps can be wrapped with `Step.cached()` to avoid
recomputing results. The wrapper stores successful `StepResult` objects in a
cache backend and reuses them on subsequent runs.

```python
from flujo import Step
from flujo.infra.caching import InMemoryCache
from flujo.testing.utils import StubAgent

# Expensive step that performs complex analysis
slow_step = Step.solution(StubAgent(["Complex analysis result: Data processed successfully"]))
cached = Step.cached(slow_step, cache_backend=InMemoryCache())
```

When the same input (together with the same context and resources and the same
step definition) is encountered again, the cached result is returned and
`StepResult.metadata_["cache_hit"]` is set to `True`.

Cache keys include a stable hash of the wrapped step's entire configuration so
that steps with the same name but different behaviors do not collide.

## Building Resilient Pipelines with Fallbacks

The `Step.fallback()` method lets you declare a backup step that runs if the primary step fails.
This is useful for handling transient errors or providing a simpler model when a complex one is unreliable.

```python
from flujo import Step, Flujo
from flujo.testing.utils import StubAgent

# Primary step that fails due to external service issues
class FailingAgent:
    async def run(self, data: str, **kwargs) -> str:
        raise RuntimeError("External API unavailable")

primary = Step("primary", FailingAgent(), max_retries=1)
# Backup step that uses a more reliable, simpler approach
backup = Step("backup", StubAgent(["Backup processing completed successfully"]))
primary.fallback(backup)

runner = Flujo(primary)
result = runner.run("data")
print(result.step_history[0].output)  # -> "Backup processing completed successfully"
```

When the fallback runs successfully, `StepResult.metadata_['fallback_triggered']` is set to `True` and the pipeline continues normally.
Resource usage from the fallback is added to the main step result, and circular
fallbacks raise `InfiniteFallbackError`.

## Performance Tips

- Use the `context_include_keys` parameter of `Step.parallel()` to copy only the
  context fields required by each branch. This avoids expensive deep copies for
  large contexts.
- When processing long iterables with `Step.map_over()` keep the context model
  minimal so each iteration stays lightweight.
