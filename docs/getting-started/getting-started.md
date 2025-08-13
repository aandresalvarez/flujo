# Getting Started

This short tutorial builds a small pipeline that prints Hello, World.

```python
from flujo import Flujo, step

@step
async def print_step(input_text: str) -> None:
    print(input_text)

pipeline = print_step
Flujo(pipeline).run("Hello, World")
```

Run the script and you should see the greeting printed to the console.
