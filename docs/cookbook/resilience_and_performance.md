# Cookbook: Improving Performance with Caching

Expensive or deterministic steps can be wrapped with `Step.cached()` to avoid
recomputing results. The wrapper stores successful `StepResult` objects in a
cache backend and reuses them on subsequent runs.

```python
from flujo import Step
from flujo.caching import InMemoryCache
from flujo.testing.utils import StubAgent

slow_step = Step.solution(StubAgent(["ok"]))
cached = Step.cached(slow_step, cache_backend=InMemoryCache())
```

When the same input (along with the same context and resources) is encountered
again, the cached result is returned and `StepResult.metadata_["cache_hit"]` is
set to `True`.

