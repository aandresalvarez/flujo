# StateMachine Transitions (Declarative)

This page documents the declarative `transitions` block for `StateMachineStep`.

## Overview

Declare how the state machine moves between states based on outcome events and optional conditions — no glue Python tools needed.

- Events: `success`, `failure`, `pause` (HITL)
- Source: `from` (state name or `*` wildcard)
- Destination: `to` (state name or a terminal entry in `end_states`)
- Condition: `when` (safe expression evaluated with `output` and `context`)

Transitions are evaluated in list order (first-match-wins) and take precedence over legacy scratchpad-based fallbacks. Use the typed `next_state` field to drive transitions from within a state.

## YAML Example

```yaml
- kind: StateMachine
  name: orchestrate
  start_state: clarification
  end_states: [done, failed]
  states:
    clarification:
      steps:
        - kind: hitl
          name: ask_for_input
          message: "Provide details?"
    concept_discovery:
      steps:
        - kind: step
          name: discover
          agent: null  # placeholder for your agent
    review:
      steps: []
    failed:
      steps: []

  transitions:
    - from: clarification
      on: pause
      to: clarification
    - from: clarification
      on: success
      to: concept_discovery
      when: "context.import_artifacts.get('cohort_definition')"
    - from: concept_discovery
      on: success
      to: review
    - from: "*"
      on: failure
      to: failed
```

## Event Semantics

- success: sub-pipeline completed; last step succeeded.
- failure: sub-pipeline completed; last step failed.
- pause: sub-pipeline raised a HITL pause (`HumanInTheLoopStep`).

On `pause`, the policy updates control metadata (`current_state` and `next_state`) and re-raises, allowing the runner to orchestrate resume.

## Conditions (`when`)

Use the safe expression engine with the following names:
- `context`: TemplateContextProxy for the pipeline context (e.g., `context.import_artifacts.get('x')`).
- `output`: A small payload `{event, last_output, last_step}`.

Examples:
- `context.import_artifacts.get('flag')`
- `output.event == 'success' and context.status == 'ready'`

Invalid expressions at runtime are treated as non-matches and logged; expressions are compiled at load time.

## Precedence and Backward Compatibility

- If transitions exist, they are applied first.
- If no transition matches, the policy falls back to `next_state` from the sub-pipeline context or outputs.
- `end_states` remain terminal regardless of transitions.
- Pipelines without `transitions` behave exactly as before.

## YAML `on` Key Note

Some YAML parsers coerce `on` into a boolean. If you encounter validation errors like `transitions.0.on: Field required`, quote the key and value:

```yaml
- from: "clarification"
  on: "success"
  to: "concept_discovery"
```

## Testing Guidance

- Unit: validate parsing, wildcard `from`, invalid `to`, and `on` values; ensure `when` compiles.
- Policy: verify success/failure/pause transitions and legacy fallback when no rule matches.
- Integration: verify `on: pause` self re-entry updates control metadata and the context is paused.

## Rationale

- Removes boilerplate and centralizes control flow in YAML.
- Improves pause/resume robustness with a single declarative rule.
- Aligns with policy-driven architecture and idempotent context handling.
