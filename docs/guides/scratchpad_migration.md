# Scratchpad to Typed Context Migration Guide

This guide helps migrate from `ctx.scratchpad` usage to properly typed context fields.

## Why Migrate?

Scratchpad is now **reserved for framework metadata only**. User data must use typed context fields for:

- **Type safety**: Compile-time validation via mypy/pyright
- **Validation**: Pydantic enforces schema at runtime
- **Discoverability**: Context fields are self-documenting
- **Maintainability**: Refactoring tools understand typed fields

## Migration Steps

### 1. Define Typed Context

Replace scratchpad-based contexts:

```python
# Before: Dict-based scratchpad
context = PipelineContext(scratchpad={"counter": 0})

# After: Typed context
from flujo.domain.context_mixins import BaseContext, typed_context

class MyContext(BaseContext):
    counter: int = 0
    intermediate_result: str | None = None

Ctx = typed_context(MyContext)
```

### 2. Update Step Output Mapping

Replace scratchpad writes with `sink_to` or `output_keys`:

```python
# Before: Writing to scratchpad via updates_context
@step(updates_context=True)
async def my_step(data):
    return {"scratchpad": {"result": data}}

# After: Explicit output mapping
@step(sink_to="result")
async def my_step(data: str) -> str:
    return data
```

### 3. Update Templated Inputs

Replace scratchpad references in templates:

```yaml
# Before
input: "{{ ctx.scratchpad.user_query }}"

# After  
input: "{{ ctx.user_query }}"
```

### 4. Use the Codemod Helper

For large codebases, use the automated codemod:

```bash
python scripts/codemods/scratchpad_to_typed.py --apply src/
```

This conservatively rewrites `ctx.scratchpad["foo"]` → `ctx.foo`.

## Validation Errors

After migration, you may encounter:

- **`CTX-SCRATCHPAD`**: Writing to/reading from scratchpad in templated input or sink
- **`CTX-OUTPUT-KEYS`**: `updates_context=True` without declaring output target

Both indicate incomplete migration. Fix by adding proper `sink_to` or `output_keys`.

## Scratchpad Is Fully Removed

`scratchpad` no longer exists on `PipelineContext` and any payload or template that references it will fail validation.
Framework metadata that used to live under scratchpad has been promoted to typed fields:

- `status`, `current_state`, `next_state` (state machine)
- `granular_state` and related counters (granular execution)
- Loop/HITL fields such as `loop_*`, `pause_message`, `paused_step_input`, `hitl_data`

Use these typed fields or `import_artifacts` for transient structured state.
