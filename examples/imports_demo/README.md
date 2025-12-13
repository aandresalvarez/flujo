# Imports Demo (ImportStep)

This example shows how to chain three imported pipelines (clarification → concept discovery → query builder) with predictable input propagation and context merging using ImportStep.

## Files

- `main.yaml` — Parent pipeline with `imports` and three ImportStep steps.
- `clarification.yaml` — Child pipeline that writes `import_artifacts.cohort_definition`.
- `concept_discovery.yaml` — Child pipeline that writes `import_artifacts.concept_sets`.
- `query_builder.yaml` — Child pipeline that writes `import_artifacts.final_sql`.
- `skills/helpers.py` — Simple Python skills used by the children.

## Run

From the repository root:

```bash
uv run flujo run examples/imports_demo/main.yaml --project examples/imports_demo --input "Define a cohort for diabetes patients"
```

- The parent passes the initial input to `clarification` as `initial_prompt`.
- Child updates merge back through `config.outputs`.
- No additional prompts are required between steps.

## Notes

- The parent uses `config.outputs` on each ImportStep to deep-merge precise fields back into the parent context.
- Relative imports and skills resolve automatically when you pass `--project examples/imports_demo`.
- To keep conversation behavior consistent across imports, set `inherit_conversation: true` under each step's `config` (optional; context fields are already inherited).

## Optional: HITL pause/resume across imports

Run the variant with a human review child:

```bash
uv run flujo run examples/imports_demo/main_with_hitl.yaml --project examples/imports_demo --input "Define a cohort for diabetes patients"
```

- The pipeline will pause at the imported `human_review` child and prompt you.
- Enter either a short confirmation (e.g., `Looks good`) or paste a JSON object like `{"name": "demo", "criteria": ["age > 18"]}`.
- The child maps the response to `import_artifacts.cohort_definition`; downstream steps proceed automatically.

Tip: If running non-interactively, you can resume programmatically via the Python API using `runner.resume_async(result, human_input)`. The CLI also auto-prompts in a TTY.
