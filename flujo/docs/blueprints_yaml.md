# Declarative YAML Blueprints (Progressive v0)

Flujo supports a minimal YAML blueprint to define pipelines as a canonical, human-friendly spec that compiles into the existing typed DSL.

Status: v1 (progressive)

- Supported constructs:
  - Simple steps with `name`, `config`, `updates_context`, `validate_fields`
  - Parallel steps with `branches`, `merge_strategy`, `on_branch_failure`, `context_include_keys`, `field_mapping`, `ignore_branch_names`
- Agent resolution: you can reference an async callable or agent via import string (e.g., `my_pkg.my_mod:async_fn`). If omitted, a passthrough agent is used.

Example:

```yaml
version: "0.1"
steps:
  - kind: step
    name: preprocess
  - kind: parallel
    name: fanout
    merge_strategy: context_update
    branches:
      a:
        - kind: step
          name: step_a1
      b:
        - kind: step
          name: step_b1
  - kind: step
    name: postprocess
```

CLI:

```bash
flujo run pipeline.yaml --input "Hello"
```

Notes:
- The YAML is compiled to `Pipeline` and `Step` objects under-the-hood; validation and policies remain unchanged.
- Merge strategies map to `MergeStrategy` and behave exactly as in the DSL.

Serialization:

```python
from flujo.domain.blueprint import dump_pipeline_blueprint_to_yaml
yaml_text = dump_pipeline_blueprint_to_yaml(pipeline)
```
