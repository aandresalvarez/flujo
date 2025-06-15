# Typed Pipeline Context

`PipelineRunner` can maintain a mutable Pydantic model instance that is shared across every step during a single run. This feature is sometimes called the *pipeline scratchpad* or *shared context*.

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
runner = PipelineRunner(
    pipeline,
    context_model=MyContext,
    initial_context_data={"user_query": "hello"},
)
```

The initial data is validated against the Pydantic model. If validation fails a `PipelineContextInitializationError` is raised and the run is aborted.

## Accessing the context in steps

Any step agent or plugin that wants access should declare a keyword-only argument named `pipeline_context`. It is recommended to type-hint this parameter with your context model.

```python
class CountingAgent(AsyncAgentProtocol[str, str]):
    async def run(
        self,
        data: str,
        *,
        pipeline_context: Optional[MyContext] = None,
        **kwargs: Any,
    ) -> str:
        if pipeline_context:
            pipeline_context.counter += 1
        return data
```

Plugins follow the same convention:

```python
class MyPlugin(ValidationPlugin):
    async def validate(
        self,
        payload: dict[str, Any],
        *,
        pipeline_context: Optional[MyContext] = None,
    ) -> PluginOutcome:
        ...
```

If a component does not accept this parameter and also does not use `**kwargs`, a `TypeError` will occur when a context is provided.

## Lifecycle

A fresh context instance is created for every call to `run()` or `run_async()`. Mutations by one step are visible to all subsequent steps in that run. Separate runs do not share state unless you explicitly pass previous context data as `initial_context_data`.

## Retrieving the final state

After execution, `PipelineResult.final_pipeline_context` holds the mutated context instance:

```python
result = runner.run("hi")
print(result.final_pipeline_context.counter)
```

For a complete example, see the [Typed Pipeline Context section](pipeline_dsl.md#typed-pipeline-context) of the Pipeline DSL guide. A runnable demonstration is available in [this script on GitHub](https://github.com/aandresalvarez/rloop/blob/main/examples/06_typed_context.py).
