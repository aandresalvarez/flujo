# Declarative YAML Blueprints (Progressive v0)

Flujo supports a minimal YAML blueprint to define pipelines as a canonical, human-friendly spec that compiles into the existing typed DSL.

Status: v1 (progressive)

- Supported constructs:
  - Simple steps with `name`, `config`, `updates_context`, `validate_fields`
  - Parallel steps with `branches`, `merge_strategy`, `on_branch_failure`, `context_include_keys`, `field_mapping`, `ignore_branch_names`
  - Loop steps with `body`, `max_loops`, `exit_condition`, `initial_input_mapper`, `iteration_input_mapper`, `loop_output_mapper`
- Agent resolution: you can reference an async callable or agent via import string (e.g., `my_pkg.my_mod:async_fn`). If omitted, a passthrough agent is used.
  - NEW: You can declare agents inline under a top-level `agents` section and reference them with `uses: agents.<name>`.
  - NEW: You can compose pipelines via a top-level `imports` section and reference them with `uses: imports.<alias>`.
  - You can also use a skills catalog file (`skills.yaml`) in the same directory:
    ```yaml
    echo-skill:
      path: "mypkg.agents:Echo"
      description: "Echo agent"
    ```
    Then in `pipeline.yaml`:
    ```yaml
    steps:
      - kind: step
        name: echo
        agent:
          id: "echo-skill"
    ```
    The CLI will auto-load `skills.yaml` before parsing the YAML.

## Enhanced Loop Steps

Loop steps now support comprehensive input/output mapping for sophisticated agentic workflows:

```yaml
- kind: loop
  name: clarification_loop
  loop:
    body:
      - name: planner
        uses: agents.clarification_agent
      - name: executor
        agent:
          id: "command_executor"
    initial_input_mapper: "skills.helpers:map_initial_input"
    iteration_input_mapper: "skills.helpers:map_iteration_input"
    exit_condition: "skills.helpers:is_finish_command"
    loop_output_mapper: "skills.helpers:map_loop_output"
    max_loops: 10
```

**Loop Configuration Keys:**
- `body`: The pipeline to execute in each iteration
- `max_loops`: Maximum number of iterations (prevents infinite loops)
- `exit_condition`: Callable that returns `True` to stop the loop
- `initial_input_mapper`: Maps LoopStep input to first iteration's body input
- `iteration_input_mapper`: Maps previous iteration output to next iteration input
- `loop_output_mapper`: Maps final successful output to LoopStep output

**Use Cases:**
- Conversational AI loops with structured data transformation
- Iterative refinement workflows
- Agentic planning and execution cycles
- Quality improvement loops with context preservation

Example:

```yaml
version: "0.1"
agents:
  categorizer:
    model: "openai:gpt-4o-mini"
    system_prompt: "You are a JSON categorizer."
    output_schema:
      type: object
      properties:
        category:
          type: string
          enum: [news, sports, finance]
      required: [category]
steps:
  - kind: step
    name: categorize
    uses: agents.categorizer
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
- Imported sub-pipelines are compiled recursively and wrapped as a single `Step` via `pipeline.as_step(name=...)`.
- Step-level `input` supports dynamic templating with `{{ ... }}` using the advanced formatter. The template receives `context` and `previous_step` variables.

Budget governance: Administrators can define centralized budgets applicable to all pipelines (including YAML) in `flujo.toml`. See `docs/guides/cost_and_budget_governance.md`.

## Imports

Add other YAML blueprints as reusable sub-pipelines:

```yaml
version: "0.1"
imports:
  support: "./support_workflow.yaml"
steps:
  - kind: step
    name: use_support
    uses: imports.support
```

Relative paths are resolved from the directory of the parent YAML file.

Security:
- YAML imports of Python objects are restricted by an allow-list. Configure allowed modules in `flujo.toml`:

```toml
# flujo.toml
blueprint_allowed_imports = ["my_safe_pkg", "my_safe_pkg.agents"]
```

If a YAML references `uses: my_safe_pkg.agents:Echo`, it will be permitted. Imports from modules not on the list will be blocked.

## Templated Input

Override a step's input via static value or Jinja-like template rendered at runtime:

```yaml
steps:
  - kind: step
    name: support_assist
    uses: imports.support
    input: "{{ context.customer_first_question }}"
  - kind: step
    name: survey
    input: "Satisfaction including '{{ previous_step }}'"
```

Rendering context:
- `context`: the current `PipelineContext` (including values passed via `--context-data` or `--context-file`).
- `previous_step`: the previous step's output.

Serialization:

```python
from flujo.domain.blueprint import dump_pipeline_blueprint_to_yaml
yaml_text = dump_pipeline_blueprint_to_yaml(pipeline)

Plugins and skills:
- You can also declare packaged skills via entry points under the group `flujo.skills`. These are discovered automatically when installed.
- For development, you can set `FLUJO_REGISTER_MODULES=module1,module2` to auto-import modules that register skills on startup.
```
