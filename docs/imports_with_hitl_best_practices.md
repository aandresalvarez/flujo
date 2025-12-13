# Imports + HITL: Deterministic Hand‑Off Best Practices

This guide shows how to pass explicit artifacts from a parent pipeline into an imported child pipeline while preserving Human‑In‑The‑Loop (HITL) behavior and avoiding conversation bleed‑through.

Key goals:
- Deterministic input routing (no accidental reuse of status messages)
- Clear diagnostics in traces and logs
- HITL propagation working end‑to‑end

## Recommended Pattern

- Use `ImportStep.input_to = "import_artifacts"` or `"both"` and provide the artifact under `import_artifacts.initial_input` (or a custom key via `input_scratchpad_key`).
- Keep `inherit_conversation = true` when you need the child to see prior turns, but always route the artifact explicitly.
- Optionally add a status step before the import to print the first N characters of the artifact for easy log inspection.

Example:

```python
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.step import Step
from flujo.domain.dsl.import_step import ImportStep, OutputMapping

child = Pipeline(steps=[ ... ])

show_snippet = Step(
    name="show_definition_snippet",
    agent=lambda ctx: f"Definition (first 500 chars): {str(ctx)[:500]}",
)

imp = ImportStep(
    name="run_concept_discovery_subpipeline",
    pipeline=child,
    inherit_context=True,
    inherit_conversation=True,
    input_to="both",  # seed initial_prompt and import_artifacts
    input_scratchpad_key="initial_input",  # default
    outputs=[OutputMapping(child="import_artifacts.echo", parent="import_artifacts.child_echo")],
    updates_context=True,
)

parent = Pipeline(steps=[show_snippet, imp])
```

Seed the explicit artifact at call time:

```python
from flujo.application.runner import Flujo
from flujo.domain.models import PipelineContext

explicit_artifact = {"cohort_definition": "Influenza, SNOMED 6142004"}
runner = Flujo(
    pipeline=parent,
    context_model=PipelineContext,
    initial_context_data={"import_artifacts": {"initial_input": explicit_artifact}},
)
result = runner.run(initial_input="ignored")
```

## What Flujo Does For You

- Import input precedence is enforced in `ImportStep` policies:
  - When `input_to` includes `initial_prompt`, the child’s effective input is resolved from:
    1) `import_artifacts[input_scratchpad_key]` when present (explicit), else
    2) the parent step’s `data` (current output), else
    3) empty string.
  - A trace metadata entry `import.initial_input_resolved` records the origin and a short preview.
  - If a short, status‑like string is routed to `initial_prompt`, Flujo emits a warning suggesting explicit artifact hand‑off.

- HITL propagation:
  - With `propagate_hitl = true` (default), pauses from the child bubble up and are visible to the parent run.

## Testing Checklist

- Repeated imports: verify each import reads the intended artifact when `initial_input` changes between steps.
- Outputs mapping: use `outputs=[{child:"import_artifacts.echo", parent:"import_artifacts.child_echo"}]` to make merges explicit.
- Conversation: when `inherit_conversation=true`, prefer passing the artifact via `import_artifacts` or `both` so it cannot be shadowed by status messages.

## Troubleshooting

- Child saw a status string instead of your artifact:
  - Route the artifact via `import_artifacts.initial_input` and set `input_to="both"`.
  - Check `import.initial_input_resolved` in step metadata to confirm the origin.
- Still no candidates for a query (e.g., “Influenza”):
  - Add a low‑noise status step showing the first 500 chars of the artifact.
  - Add temporary diagnostics in your tools (e.g., print first raw candidate keys once per run).
