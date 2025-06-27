# Cookbook: Adapter Step

Use an **Adapter Step** when you need to transform raw data or combine context before passing it to the next agent. Inline mappers work for simple cases, but adapter steps keep your pipelines readable and testable.

## Example

```python
from pydantic import BaseModel
from flujo import Flujo, adapter_step, step

class ComplexInput(BaseModel):
    text: str
    length: int

@adapter_step
async def build_input(data: str) -> ComplexInput:
    return ComplexInput(text=data, length=len(data))

@step
async def summarize(inp: ComplexInput) -> str:
    return inp.text[:3]

pipeline = build_input >> summarize
runner = Flujo(pipeline)
result = runner.run("hello")
assert result.step_history[-1].output == "hel"
```

## Testing Adapter Steps

Use `arun()` to exercise an adapter step in isolation.

```python
import pytest

@adapter_step
async def build(x: str) -> ComplexInput:
    return ComplexInput(text=x, length=len(x))

@pytest.mark.asyncio
async def test_build():
    out = await build.arun("ok")
    assert out.length == 2
```
