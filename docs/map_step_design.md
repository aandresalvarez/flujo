# MapStep Design Notes

MapStep provides a declarative way to map a pipeline across items stored in the pipeline context.
This design keeps configuration (what to run) separate from transient runtime state (what’s being iterated).

## Goals
- Declarative: configure with `pipeline_to_run` and `iterable_input`.
- Predictable: expose stable attributes for mapping functions and loop control.
- Safe: avoid direct context mutation; always use utilities for updates.

## Key Behaviors
- `pipeline_to_run`: the actual pipeline executed per item. Preserved as `original_body_pipeline` for clarity.
- `loop_body_pipeline`: a no-op placeholder for introspection only; execution uses `original_body_pipeline`.
- Mapping functions: initialized as attributes for direct access (`initial_input_to_loop_body_mapper`, `iteration_input_mapper`, `loop_output_mapper`).
- Loop count: `max_retries` starts at `1` and updates to the iterable length after the first mapping.
- Errors: requires a context; iterable must be a non-string iterable.

## Why This Design
- Aligns with Pydantic model expectations (no custom constructors).
- Keeps runtime state (`items`, `results`) excluded from serialization.
- Matches team guidance: policy-driven execution, control-flow exceptions never swallowed, and explicit state transitions.

## Example
```
map_step = Step.map_over(
  name="process_items",
  pipeline_to_run=Pipeline.from_step(process),
  iterable_input="items",
)
```

This configuration instructs Flujo to run `process` for each `context.items`, collecting outputs safely.
