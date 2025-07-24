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
    return inp.text[:4]

pipeline = build_input >> summarize
runner = Flujo(pipeline)
result = runner.run("helpful")
assert result.step_history[-1].output == "help"  # 'help' is the first four letters of 'helpful'
```

## Testing Adapter Steps

Use `arun()`
