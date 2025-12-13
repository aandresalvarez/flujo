# Type-Safe Patterns Quickstart

This guide shows recommended patterns for building type-safe Flujo pipelines.

## Typed Contexts (Recommended)

Use typed contexts instead of scratchpad for pipeline state:

```python
from flujo.domain.context_mixins import BaseContext, typed_context
from flujo.domain.dsl.step_decorators import step

# Define your typed context
class MyContext(BaseContext):
    counter: int = 0
    result: str | None = None
    user_query: str = ""

Ctx = typed_context(MyContext)

# Steps can use sink_to for simple outputs
@step(sink_to="counter")
async def increment(data: int) -> int:
    return data + 1

# Or updates_context with output_keys for structured updates
@step(updates_context=True, output_keys=["result"])
async def process(query: str) -> dict[str, str]:
    return {"result": f"Processed: {query}"}
```

## Adapter Steps (When Type Bridging is Required)

Use explicit adapters with allowlist tokens:

```python
from flujo.domain.dsl.step_decorators import adapter_step

# Adapters must declare their identity for governance
@adapter_step(
    adapter_id="dict-to-model",
    adapter_allow="core-migration",  # Must match allowlist
)
async def bridge_types(data: dict) -> MyModel:
    return MyModel(**data)
```

## Type-Inferred Pipelines

Pipeline composition tracks input/output types:

```python
from flujo.domain.dsl.step import Step
from flujo.domain.dsl.pipeline import Pipeline

@step
async def fetch(query: str) -> dict:
    return {"data": query}

@step  
async def transform(data: dict) -> str:
    return str(data)

# Pipeline infers: str -> dict -> str
pipeline = fetch >> transform

# Access inferred types
print(f"Input: {pipeline.input_type}")   # str
print(f"Output: {pipeline.output_type}") # str
```

## Running with Type Checking

Enable strict validation at runtime:

```python
from flujo import Flujo
from flujo.state.backends import InMemoryStateBackend

runner = Flujo(state_backend=InMemoryStateBackend())

# Type mismatches raise immediately
result = await runner.run_async(
    pipeline,
    initial_input="Hello",
    context_model=MyContext,
)
```

## CI Integration

Type-safety is enforced via:
- `make lint` runs type-safety baseline checks
- Architecture tests guard Any/cast counts
- Pipeline validation errors on type mismatches

See [Context Strict Mode](../context_strict_mode.md) for details.
