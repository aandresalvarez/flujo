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

## Framework-Reserved Scratchpad Keys

Some keys remain valid in scratchpad for framework use:

- `status`, `current_state` (state machine)
- `turn_index`, `history` (granular execution)
- `slots_*` (slot synthesis helpers)

These are allowlisted and won't trigger validation errors.
