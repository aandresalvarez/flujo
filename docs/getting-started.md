# Getting Started

This short tutorial builds a small pipeline that prints **Hello, World**.

```python
from flujo import Pipeline, Step

# Define a simple step
class PrintStep(Step[str, None]):
    def run(self, input: str) -> None:
        print(input)

# Assemble and run the pipeline
pipeline = Pipeline(steps=[PrintStep()])
pipeline.run_sync("Hello, World")
```

Run the script and you should see the greeting printed to the console.
