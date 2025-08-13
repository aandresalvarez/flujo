### Flujo Validation Rules Reference

This page documents the rule IDs emitted by `Pipeline.validate_graph()` and related linters.

- V-A1: Missing agent on simple step
  - Checks: Non-complex `Step` without `agent`.
  - Why: Simple steps must have an agent to run.
  - Suggestion: Assign an agent via `Step.from_callable(...)` or a step factory.

- V-A2: Type mismatch between steps
  - Checks: Output type of previous step incompatible with input type of next step.
  - Why: Prevent runtime type errors.
  - Suggestion: Insert an adapter step or align function signatures.

- V-A3: Reused `Step` instance
  - Checks: Same `Step` object appears multiple times in a pipeline.
  - Why: Statefulness may cause side effects.
  - Suggestion: Create distinct `Step` instances for reuse.

- V-A4-ERR: Signature analysis failed
  - Checks: Could not analyze agent callable signature.
  - Why: Provides visibility into inspection issues.
  - Suggestion: Ensure agent is a plain async callable or exposes `_step_callable`.

- V-A5: Unbound output warning
  - Checks: Producer’s output is likely unused by next step and producer does not update context.
  - Why: Surfaces likely logic errors and wasted work.
  - Suggestion: Set `updates_context=True` or add an adapter step.

- V-F1: Incompatible fallback signature
  - Checks: `fallback_step` input type incompatible with primary step input type.
  - Why: Fallback must accept the same input as primary.
  - Suggestion: Align fallback input type or add an adapter step.

- V-P1: Parallel context merge conflict
  - Checks: `ParallelStep` with `CONTEXT_UPDATE` may write same keys across branches without disambiguation.
  - Why: Prevents nondeterministic or lossy context merges.
  - Suggestion: Provide `field_mapping`, set explicit merge strategy (e.g., `OVERWRITE`), or avoid overlapping writes.

- V-P1-W: Parallel merge heuristic warning
  - Checks: `ParallelStep` using `CONTEXT_UPDATE` without `field_mapping` where conflicts are possible but not provable.
  - Why: Early signal for potential conflicts.
  - Suggestion: Provide `field_mapping` or pick an explicit strategy (`OVERWRITE`/`ERROR_ON_CONFLICT`).


