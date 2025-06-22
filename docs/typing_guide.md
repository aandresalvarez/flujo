# Typing Guide

`flujo` embraces Python's type hints to make pipelines safer and easier to use.
The `Step.from_callable` factory automatically infers a step's input and output
types from an async function.

```python
from flujo import Step

async def process(data: str) -> int:
    return len(data)

step = Step.from_callable(process)
```

Static type checkers like `mypy` infer `step` as `Step[str, int]` so pipelines
compose cleanly without manual casts. If a callable lacks annotations, the
factory falls back to `Any`.
