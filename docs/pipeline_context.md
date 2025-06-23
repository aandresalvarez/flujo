# Typed Pipeline Context

`Flujo` can maintain a mutable Pydantic model instance that is shared across every step during a single run. This feature is sometimes called the *pipeline scratchpad* or *shared context*.

## Why use a context?

- Accumulate metrics or intermediate results across steps.
- Provide configuration or runtime parameters to non‑adjacent steps.
- Keep your data flow explicit and type safe.

## Defining a context model

```python
from pydantic import BaseModel

class MyContext(BaseModel):
    user_query: str
    counter: int = 0
```

## Initializing the runner

```python
runner = Flujo(
    pipeline,
    context_model=MyContext,
    initial_context_data={"user_query": "hello"},
)
```

The initial data is validated against the Pydantic model. If validation fails a `PipelineContextInitializationError` is raised and the run is aborted.

## Accessing the context in components

Implement the `ContextAwareAgentProtocol` or `ContextAwarePluginProtocol` to receive a strongly typed context without casts.

```python
from flujo.domain.agent_protocol import ContextAwareAgentProtocol
from flujo.domain.plugins import ContextAwarePluginProtocol, PluginOutcome

class CountingAgent(ContextAwareAgentProtocol[str, str, MyContext]):
    async def run(self, data: str, *, pipeline_context: MyContext, **kwargs: Any) -> str:
        pipeline_context.counter += 1
        return data

class MyPlugin(ContextAwarePluginProtocol[MyContext]):
    async def validate(self, data: dict[str, Any], *, pipeline_context: MyContext, **kwargs: Any) -> PluginOutcome:
        return PluginOutcome(success=True)
```

Legacy agents that merely accept a `pipeline_context` parameter will still work but will trigger a `DeprecationWarning`. Updating them to implement the `ContextAware` protocol is recommended.

## Lifecycle

A fresh context instance is created for every call to `run()` or `run_async()`. Mutations by one step are visible to all subsequent steps in that run. Separate runs do not share state unless you explicitly pass previous context data as `initial_context_data`.

## Retrieving the final state

After execution, `PipelineResult.final_pipeline_context` holds the mutated context instance:

```python
result = runner.run("hi")
print(result.final_pipeline_context.counter)
```

For a complete example, see the [Typed Pipeline Context section](pipeline_dsl.md#typed-pipeline-context) of the Pipeline DSL guide. A runnable demonstration is available in [this script on GitHub](https://github.com/aandresalvarez/flujo/blob/main/examples/06_typed_context.py).
