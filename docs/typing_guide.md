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

```python
from pydantic import BaseModel

class Info(BaseModel):
    message: str
    count: int

async def increment(info: Info) -> Info:
    info.count += 1
    return info

typed = Step.from_callable(increment)
# inferred as Step[Info, Info]

async def untyped(x):
    return x

any_step = Step.from_callable(untyped)
# inferred as Step[Any, Any]

async def uses_context(data: str, *, pipeline_context: Info) -> str:
    return data + pipeline_context.message

context_step = Step.from_callable(uses_context)
```

Keyword-only parameters such as ``pipeline_context`` or ``resources`` are
ignored when inferring the input type. They are still passed through during
execution, allowing you to access shared state or application resources without
affecting the step's signature.
