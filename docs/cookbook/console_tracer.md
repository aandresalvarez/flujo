# Cookbook: Using the Console Tracer

## The Problem

When developing pipelines, it can be hard to understand what happens at each step. You want instant feedback in your terminal with minimal setup.

## The Solution

The `ConsoleTracer` provides rich, colorized output for every step in a run. Enable it via the `local_tracer` parameter when constructing `Flujo`.

```python
from flujo import Flujo, Step
from flujo.tracing import ConsoleTracer
from flujo.testing.utils import StubAgent

step = Step("example", StubAgent(["ok"]))

# Quick enablement with the built-in defaults
runner = Flujo(step, local_tracer="default")

# Or configure it yourself
custom = ConsoleTracer(level="debug", log_inputs=True)
runner_custom = Flujo(step, local_tracer=custom)
```

With `level="info"` only high‑level events are printed. `level="debug"` also shows inputs and outputs.
