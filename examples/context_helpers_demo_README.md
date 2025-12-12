# Context Helpers Demo

This example demonstrates Flujo's built-in context manipulation skills: `context_set`, `context_merge`, and `context_get`.

## What This Demo Shows

1. **Setting simple values**: Store primitive values in context using `context_set`
2. **Merging dictionaries**: Merge complex objects into context using `context_merge`
3. **Getting values**: Retrieve values with fallback defaults using `context_get`
4. **Type-safe operations**: All operations are type-safe and work with nested paths

## Built-in Skills Used

### `flujo.builtins.context_set`
Sets a value at a specific context path.

**Input**:
```yaml
path: "import_artifacts.field_name"
value: <any value>
```

**Example**:
```yaml
- kind: step
  name: set_counter
  agent:
    id: "flujo.builtins.context_set"
  input:
    path: "import_artifacts.counter"
    value: 0
  updates_context: true
```

### `flujo.builtins.context_merge`
Merges a dictionary into the context at a specific path.

**Input**:
```yaml
path: "import_artifacts.field_name"
value: <dictionary to merge>
```

**Example**:
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

### `flujo.builtins.context_get`
Gets a value from the context with an optional default.

**Input**:
```yaml
path: "import_artifacts.field_name"
default: <fallback value>
```

**Example**:
```yaml
- kind: step
  name: get_counter
  agent:
    id: "flujo.builtins.context_get"
  input:
    path: "import_artifacts.counter"
    default: 0
```

## Running the Demo

```bash
uv run flujo run --pipeline examples/context_helpers_demo.yaml --input "start"
```

## Key Benefits

1. **Type Safety**: Unlike direct context manipulation, these helpers are type-safe
2. **Nested Paths**: Supports dot-separated paths like `import_artifacts.user.settings.theme`
3. **Graceful Defaults**: `context_get` provides fallbacks for missing values
4. **No Boilerplate**: Eliminates custom Python skills for simple context operations

## When to Use

- **Use context helpers** for simple get/set/merge operations
- **Use custom skills** when you need complex logic or transformations
- **Use `sink_to` in HITL steps** for automatic storage of human responses
