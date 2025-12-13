# Typed Pipeline Context

`Flujo` can share a mutable Pydantic model across all steps in a single run. This is useful for accumulating metrics or passing configuration.

A context instance is created for every call to `run()` and is available to steps, agents, and plugins that declare a `context` parameter.

For complete details on implementing context aware components see the [Stateful Pipelines](../advanced/typing_guide.md#stateful-pipelines-the-contextaware-protocols) section of the Typing Guide.

## Best Practices for Custom Context Models

To create your own context model, **inherit from `flujo.models.PipelineContext`**.
This base class provides important built-in fields managed by the engine:

- `initial_prompt: str` – automatically populated with the first input of each `run()` call.
- `import_artifacts: ImportArtifacts` – a structured, dict-like store for transient state and import hand-offs.
- `step_outputs: Dict[str, Any]` – prior step outputs keyed by step name.
- `hitl_history: List[HumanInteraction]` – records all human-in-the-loop interactions.
- `command_log: List[ExecutedCommandLog]` – tracks commands issued by an agentic loop pipeline.

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

## Built-in Context Helpers

Flujo provides three built-in skills for type-safe context manipulation: `context_set`, `context_merge`, and `context_get`. These eliminate boilerplate and prevent type errors.

### `flujo.builtins.context_set`

Sets a value at a specific context path (supports nested paths with dot notation).

**Usage in YAML**:
```yaml
- kind: step
  name: set_user_name
	  agent:
	    id: "flujo.builtins.context_set"
	  input:
	    path: "import_artifacts.user_name"
	    value: "Alice"
	  updates_context: true
```

**Usage in Python**:
```python
from flujo.builtins import context_set

	@step
	async def initialize(data: str, *, context: PipelineContext):
	    await context_set(path="import_artifacts.counter", value=0, context=context)
	    return data
```

### `flujo.builtins.context_merge`

Merges a dictionary into the context at a specific path.

**Usage in YAML**:
```yaml
- kind: step
  name: merge_settings
	  agent:
	    id: "flujo.builtins.context_merge"
	  input:
	    path: "import_artifacts.settings"
	    value:
	      theme: "dark"
	      notifications: true
	  updates_context: true
```

**Usage in Python**:
```python
from flujo.builtins import context_merge

	@step
	async def add_settings(data: str, *, context: PipelineContext):
	    settings = {"theme": "dark", "notifications": True}
	    await context_merge(path="import_artifacts.settings", value=settings, context=context)
	    return data
```

### `flujo.builtins.context_get`

Gets a value from the context with an optional default fallback.

**Usage in YAML**:
```yaml
- kind: step
  name: get_counter
	  agent:
	    id: "flujo.builtins.context_get"
	  input:
	    path: "import_artifacts.counter"
	    default: 0
```

**Usage in Python**:
```python
from flujo.builtins import context_get

	@step
	async def read_counter(data: str, *, context: PipelineContext):
	    counter = await context_get(path="import_artifacts.counter", default=0, context=context)
	    return f"Counter is at {counter}"
```

### Nested Path Support

All helpers support dot-separated paths for nested fields:

```yaml
# Set a deeply nested value
- kind: step
  name: set_nested
	  agent:
	    id: "flujo.builtins.context_set"
	  input:
	    path: "import_artifacts.user.preferences.theme"
	    value: "dark"
	  updates_context: true

# Get a nested value with fallback
- kind: step
  name: get_nested
	  agent:
	    id: "flujo.builtins.context_get"
	  input:
	    path: "import_artifacts.user.preferences.theme"
	    default: "light"
```

### When to Use

- **Use context helpers** for simple get/set/merge operations (no custom logic needed)
- **Use custom Python skills** when you need complex transformations or validation
- **Use `sink_to` in HITL steps** for automatic storage of human responses

**Example**: See `examples/context_helpers_demo.yaml` for a complete working example.

## Context Updates in Loops

When using `@step(updates_context=True)` within loop steps (`Step.loop_until()`), context updates are now properly applied between iterations. This ensures that state changes from one iteration are available in subsequent iterations.

### How It Works

1. **Context Isolation**: Each loop iteration operates on a deep copy of the context to prevent unintended side effects
2. **Context Merging**: After each iteration, context updates are merged back to the main context
3. **State Persistence**: Changes made by `@step(updates_context=True)` steps persist across iterations

### Example: Iterative Refinement

```python
from flujo import Step, Pipeline, step
from flujo.domain.models import PipelineContext

class RefinementContext(PipelineContext):
    current_definition: str
    is_clear: bool = False
    iteration_count: int = 0

@step(updates_context=True)
async def assess_and_refine(definition: str, *, context: RefinementContext) -> dict:
    """Assess definition clarity and update context."""
    context.iteration_count += 1

    # Simulate AI assessment
    if "clear" in definition.lower() or context.iteration_count >= 3:
        context.is_clear = True
        return {
            "is_clear": True,
            "current_definition": definition
        }
    else:
        # Simulate clarification needed
        context.is_clear = False
        return {
            "is_clear": False,
            "current_definition": f"{definition} (clarified)"
        }

# Create loop with context updates
loop_body = Pipeline.from_step(assess_and_refine)

def exit_when_clear(output: dict, context: RefinementContext) -> bool:
    """Exit loop when definition is clear."""
    return context.is_clear

refinement_loop = Step.loop_until(
    name="refinement_loop",
    loop_body_pipeline=loop_body,
    exit_condition_callable=exit_when_clear,
    max_loops=5
)

# Run with context
runner = Flujo(
    refinement_loop,
    context_model=RefinementContext
)

initial_context = {
    "initial_prompt": "test",
    "current_definition": "ambiguous definition",
    "is_clear": False,
    "iteration_count": 0
}

result = None
async for item in runner.run_async("ambiguous definition", initial_context_data=initial_context):
    result = item

# Context updates are properly applied
assert result.final_pipeline_context.is_clear is True
assert result.final_pipeline_context.iteration_count >= 1
```

### Best Practices

1. **Use Explicit State**: Define clear boolean flags instead of parsing strings
2. **Validate Context**: Use Pydantic validators to ensure context consistency
3. **Handle Errors**: Implement proper error handling for context updates
4. **Test Thoroughly**: Verify context state across multiple iterations

### Debugging Context Updates

If context updates aren't working as expected:

1. **Check Step Decorator**: Ensure `@step(updates_context=True)` is used
2. **Verify Return Type**: Return a dictionary or Pydantic model from the step
3. **Inspect Context**: Use logging to verify context state between iterations
4. **Test Isolation**: Verify that context updates don't interfere with loop logic

### Migration from String Parsing

**Before (Fragile)**:
```python
@step
async def assess_definition(definition: str) -> str:
    # Parse string to determine state
    if "[CLARITY_CONFIRMED]" in definition:
        return "clear"
    return "needs_clarification"
```

**After (Robust)**:
```python
@step(updates_context=True)
async def assess_definition(definition: str, *, context: MyContext) -> dict:
    # Use explicit boolean flags
    if "clear" in definition.lower():
        return {"is_clear": True}
    return {"is_clear": False}
```
