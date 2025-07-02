# Typed Pipeline Context

`Flujo` can share a mutable Pydantic model across all steps in a single run. This is useful for accumulating metrics or passing configuration.

A context instance is created for every call to `run()` and is available to steps, agents, and plugins that declare a `context` parameter.

For complete details on implementing context aware components see the [Stateful Pipelines](typing_guide.md#stateful-pipelines-the-contextaware-protocols) section of the Typing Guide.

## Best Practices for Custom Context Models

To create your own context model, **inherit from `flujo.domain.models.PipelineContext`**.
This base class provides important built-in fields managed by the engine:

- `initial_prompt: str` – automatically populated with the first input of each `run()` call.
- `scratchpad: Dict[str, Any]` – a general-purpose dictionary for transient state.
- `hitl_history: List[HumanInteraction]` – records all human-in-the-loop interactions.
- `command_log: List[ExecutedCommandLog]` – tracks commands issued by an `AgenticLoop`.

A minimal custom context looks like this:

```python
from flujo.domain.models import PipelineContext
from pydantic import Field

class MyDiscoveryContext(PipelineContext):
    frontier: list[int] = Field(default_factory=list)
    seen_ids: set[int] = Field(default_factory=set)

runner = Flujo(
    my_pipeline,
    context_model=MyDiscoveryContext,
    initial_context_data={"frontier": [123]},
)
runner.run("My first input")
```

The runner automatically fills `initial_prompt` when you call `run()`. You only
pass data for your custom fields.
