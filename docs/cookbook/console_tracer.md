# Cookbook: Using the Console Tracer

The `ConsoleTracer` is a hook that provides detailed, colorized output of your pipeline's execution to the console. It's a great tool for debugging and understanding how your pipeline is working.

## Usage

To use the `ConsoleTracer`, you first need to instantiate it. You can then pass it to the `Flujo` runner in the `hooks` list.

```python
from flujo import Step, Flujo
from flujo.infra.console_tracer import ConsoleTracer
from flujo.testing.utils import StubAgent

# Create a tracer
tracer = ConsoleTracer()

# Create a simple pipeline
pipeline = Step.from_mapper(lambda x: x.upper())

# Create a runner with the tracer hook
runner = Flujo(pipeline, hooks=[tracer.hook])

# Run the pipeline
runner.run("hello world")
```

This will print a series of panels to the console, showing the start and end of the pipeline, as well as the start and end of each step.

## Configuration

The `ConsoleTracer` can be configured to control the level of detail and the appearance of the output.

*   `level`: The logging level. Can be `"info"` or `"debug"`. In `"debug"` mode, the tracer will also log the inputs and outputs of each step.
*   `log_inputs`: Whether to log the inputs of each step. Defaults to `True`.
*   `log_outputs`: Whether to log the outputs of each step. Defaults to `True`.
*   `colorized`: Whether to use color in the output. Defaults to `True`.

```python
# Create a tracer with custom configuration
tracer = ConsoleTracer(
    level="info",
    log_inputs=False,
    log_outputs=False,
    colorized=False,
)
```
