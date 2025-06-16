# Lifecycle Hooks

This cookbook demonstrates how to use lifecycle hooks to observe or modify a pipeline run.

```python
from flujo import Flujo, Step, PipelineAbortSignal

async def logger_hook(**kwargs):
    print("event:", kwargs.get("event_name"))

async def abort_on_fail(**kwargs):
    if kwargs.get("event_name") == "on_step_failure":
        raise PipelineAbortSignal("aborted from cookbook")

pipeline = Step("s1", agent=MagicMock(return_value="ok"))
runner = Flujo(pipeline, hooks=[logger_hook, abort_on_fail])
result = runner.run("hello")
```

Hooks receive keyword arguments specific to the event. Raising `PipelineAbortSignal`
terminates the run gracefully.
