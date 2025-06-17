# Cookbook: Observing Runs with Lifecycle Hooks

## The Problem

You need to integrate custom logic into the pipeline's execution flow. For example, you might want to log every step's result to a monitoring service, send a Slack notification on failure, or even programmatically stop a run based on some custom condition.

## The Solution

The `Flujo` runner accepts an optional `hooks` parameter, which is a list of `async` functions. These functions are called at specific points in the pipeline lifecycle. You can also raise the `PipelineAbortSignal` exception from a hook to gracefully stop the run.

```python
from unittest.mock import MagicMock
from flujo import Flujo, Step, Pipeline
from flujo.exceptions import PipelineAbortSignal

# --- Define Hook Functions ---

async def simple_logger_hook(**kwargs):
    """A hook that prints the name of every event."""
    event_name = kwargs.get('event_name')
    print(f"HOOK FIRED: {event_name}")

async def abort_on_failure_hook(**kwargs):
    """A hook that gracefully stops the pipeline if any step fails."""
    if kwargs.get('event_name') == 'on_step_failure':
        step_name = kwargs['step_result'].name
        print(f"HOOK: Step '{step_name}' failed. Aborting the run.")
        raise PipelineAbortSignal(f"Aborted due to failure in '{step_name}'")

# --- Setup and Run ---
failing_step = Step("failing_step", agent=MagicMock(side_effect=RuntimeError("An error occurred!")))
pipeline = Step("successful_step", agent=MagicMock(return_value="ok")) >> failing_step

runner = Flujo(
    pipeline,
    hooks=[simple_logger_hook, abort_on_failure_hook]
)

# The run will be stopped by the hook, not crash from the agent's error.
result = runner.run("start")

print(f"\nPipeline finished. It ran {len(result.step_history)} steps before being aborted by the hook.")
```

### How It Works

1.  We define two hooks: `simple_logger_hook` and `abort_on_failure_hook`. Both are `async def` functions that accept `**kwargs`.
2.  We register these hooks by passing them in a list to the `Flujo` constructor.
3.  As the pipeline runs, the `Flujo` engine calls every registered hook for each event (`pre_run`, `pre_step`, etc.). The `simple_logger_hook` prints the name of each event as it happens.
4.  The `failing_step`'s agent raises a `RuntimeError`. The engine catches this, marks the step as failed, and then fires the `on_step_failure` event.
5.  Our `abort_on_failure_hook` receives this event, checks the event name, and raises `PipelineAbortSignal`.
6.  The engine catches this special signal and gracefully terminates the entire run, returning the partial `PipelineResult`.

This provides a powerful and clean way to add cross-cutting concerns like logging, metrics, and custom control flow to your pipelines.
