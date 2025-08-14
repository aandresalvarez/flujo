# Validation Rules Reference

This page lists the static validation checks enforced by `pipeline.validate_graph()` and surfaced by `flujo validate` and `flujo run` (pre-run).

Each finding includes a rule ID, severity, message, and an optional suggestion with a concrete fix.

## V-P1: Parallel context merge conflict
- Severity: error (may appear as warning when insufficient static hints)
- Trigger: A `ParallelStep` uses the default `CONTEXT_UPDATE` strategy and two or more branches may update the same context field without an explicit `field_mapping`.
- Message: Potential/actual merge conflict for key(s) in ParallelStep.
- Suggestion: Set an explicit MergeStrategy (e.g., OVERWRITE or ERROR_ON_CONFLICT) or provide a `field_mapping` that ensures only one branch writes each key.

## V-A5: Unbound output warning
- Severity: warning
- Trigger: A step produces a meaningful output but the next step does not consume it (accepts `object`/`None`), and the producing step does not set `updates_context=True`.
- Message: The output of step X is not used by the next step Y.
- Suggestion: Set `updates_context=True` on the producing step or insert an adapter step to consume its output.

## V-F1: Incompatible fallback signature
- Severity: error
- Trigger: A step has `fallback_step` whose input type is not compatible with the original step input type.
- Message: Fallback step expects input T_fallback, which is not compatible with original input T_step.
- Suggestion: Ensure the fallback step accepts the same input type as the original step or add an adapter.

### Notes
- Run `flujo validate --strict` to exit non-zero on any errors.
- `flujo run` aborts before execution when errors are present.
- For YAML blueprints, these checks run after compilation to the typed DSL.
