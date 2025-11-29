# Typing Guide

`flujo` uses Python type hints to help you build robust pipelines. This guide shows how the `@step` decorator, `ContextAware` protocols and static type checkers work together.

## The `@step` Decorator: Effortless Typed Steps

Wrapping an async function with `@step` automatically creates a `Step[In, Out]` object using the function's signature. No manual generics are required.

```python
from flujo import step

async def legacy_process(data: str) -> int:
    return len(data)

process_step = step(legacy_process)  # Step[str, int]
```

The decorator can also be used directly on your function:

```python
from flujo import step

@step
async def to_upper(text: str) -> str:
    return text.upper()
```

Here `to_upper` is already a `Step[str, str]` ready to be composed with other steps.

### Background steps and validation

Steps configured with `execution_mode="background"` pass their **input** through to downstream steps. The type validator accounts for this by using the step's input type when checking compatibility, avoiding false mismatches when the background step's declared output differs from the pass-through value.

```python
from flujo import step, StepConfig

bg_cfg = StepConfig(execution_mode="background")

@step(config=bg_cfg)
async def enqueue(task: str) -> dict:
    # Runs in background; downstream receives the original string
    return {"scheduled": task}

@step
async def consume(task: str) -> str:
    return task.upper()

# Valid: background step passes its input type (str) to consume()
pipeline = enqueue >> consume
```

## Type Safety in Pipelines

Pipelines are strongly typed. If you try to chain incompatible steps, static analyzers such as `mypy` will flag an error.

```python
from flujo import step

@step
async def first(x: str) -> int:
    return len(x)

@step
async def second(x: str) -> str:
    return x

pipeline = first >> second  # ❌ mypy: incompatible types
```

Because `first` outputs an `int` while `second` expects a `str`, `mypy` warns that the composition is invalid.

## Stateful Pipelines: The `ContextAware` Protocols

To share state across steps, define a Pydantic model and have your agents or plugins implement one of the context aware protocols. They receive a typed context instance automatically.

> **Parameter Naming:**
> Steps, agents, and plugins can declare a `context` parameter to receive the shared context.

```python
from flujo.domain.models import PipelineContext
from flujo.domain.agent_protocol import ContextAwareAgentProtocol
from flujo.domain.plugins import ContextAwarePluginProtocol, PluginOutcome

class MyContext(PipelineContext):
    user_query: str
    counter: int = 0

class CountingAgent(ContextAwareAgentProtocol[str, str, MyContext]):
    async def run(self, data: str, *, context: MyContext, **_: object) -> str:
    context.counter += 1
    return data

class MyPlugin(ContextAwarePluginProtocol[MyContext]):
    async def validate(self, data: dict[str, object], *, context: MyContext, **_: object) -> PluginOutcome:
        return PluginOutcome(success=True)
```

Every call to `Flujo.run()` creates a fresh context instance. Mutations are visible to all subsequent steps.

## A Complete Example

```python
from flujo import Flujo, step, PipelineResult
from flujo.domain.models import PipelineContext

class Ctx(PipelineContext):
    history: list[str] = []

@step
async def record(text: str, *, context: Ctx) -> str:
    context.history.append(text)
    return text.upper()

@step
async def cheer(text: str) -> str:
    return f"{text}!"

pipeline = record >> cheer
runner = Flujo(pipeline, context_model=Ctx)
result: PipelineResult[str] = runner.run("hello")
print(result.final_pipeline_context.history)  # ['hello']
print(result.step_history[-1].output)        # 'HELLO!'
```

## JSON Types: Prefer `JSONObject` over `dict[str, Any]`

Architecture checks require new code to avoid `Dict[str, Any]` for unstructured JSON. Use the shared alias instead:

```python
from flujo.type_definitions.common import JSONObject

def make_payload(user_id: str) -> JSONObject:
    return {"user_id": user_id, "metadata": {"source": "cli"}}
```

This keeps JSON shapes consistent across agents, plugins, and steps while satisfying the type-safety gate.

This pipeline records each input in the context while producing an enthusiastic response.
