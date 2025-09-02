# FSD-0XX: Declarative State Transitions for StateMachineStep

Author: Flujo Core Team
Date: 2025-09-02
Status: Approved for implementation
Target Release: vNEXT (Phase 1 – Developer Experience)

## 1. Abstract

Introduce a declarative `transitions` block to `StateMachineStep` so authors can express state hops in YAML based on outcome events (`success`, `failure`, `pause`) with optional conditions (`when`). This removes boilerplate glue tools, makes pause/resume robust and explicit, and keeps control flow centralized in the policy layer while preserving Flujo’s non-negotiable principles.

## 2. Background & Problem Statement

Today, `StateMachineStep` requires users to set `scratchpad.next_state` from custom Python tools. This has downsides:
- Glue code: bespoke if/elif chains tightly coupled to pipeline internals.
- Pause/resume fragility: on HITL pause, users must manually force re-entry into the paused state.
- Hidden control flow: transitions live outside YAML, reducing blueprint clarity.

This FSD proposes a first-class, declarative transition model resolved by the `StateMachinePolicyExecutor` after each state finishes (or pauses), with strict adherence to control-flow exception safety and idempotent context handling.

## 3. Goals & Non-Goals

Goals
- Express state transitions in YAML via ordered rules with optional conditions.
- Robust pause handling via `on: pause` rules, ensuring correct re-entry on resume.
- Preserve policy-driven architecture; no changes to the executor dispatcher.
- Maintain idempotency: per-iteration isolation, only merge on success; never poison shared context.
- Backward-compatible: pipelines without `transitions` continue to work unchanged.

Non-Goals
- No new control-flow exception types.
- No expansion of the safe expression engine beyond existing capabilities.
- No changes to quota semantics or centralized configuration.

## 4. Design Overview

Add `transitions` to `StateMachineStep`. Each rule matches `(from, on, when)` to a `to` state.

YAML example:

```yaml
- kind: StateMachine
  name: orchestrate
  start_state: clarification
  end_states: [done, failed]
  states:
    clarification:
      steps:
        - name: run_clarification
          uses: imports.clarification
    concept_discovery:
      steps:
        - name: run_concept_discovery
          uses: imports.concept_discovery
    review: { steps: [ ... ] }
    failed: { steps: [] }
    done: { steps: [] }

  transitions:
    - from: clarification
      on: pause
      to: clarification
    - from: clarification
      on: success
      to: concept_discovery
      when: "context.scratchpad.cohort_definition"
    - from: concept_discovery
      on: pause
      to: concept_discovery
    - from: concept_discovery
      on: success
      to: review
    - from: "*"
      on: failure
      to: failed
```

Event mapping
- success: sub-pipeline completed and last `StepResult.success` is True.
- failure: sub-pipeline completed and last `StepResult.success` is False.
- pause: sub-pipeline raised `PausedException` (HITL).

Rule resolution: first-match-wins among rules in list order. `from: "*"` is a wildcard.

## 5. Architecture Alignment (Non-Negotiables)

- Policy-driven execution: All logic lives in `StateMachinePolicyExecutor`; `ExecutorCore` remains a dispatcher. No `isinstance` special-casing in core.
- Control-flow exception safety: Catch `PausedException` only to annotate control state in context, then re-raise immediately. Never convert to data failure.
- Context idempotency: Per-state iteration runs on isolated context; merge back only on completion (success/failure). On pause, do not merge iteration context.
- Proactive quotas: Unchanged; no reactive checks introduced.
- Centralized configuration: No direct env/TOML reads in policy/domain.
- Agent creation: Unchanged.

## 6. Detailed Design

### 6.1 DSL Schema

File: `flujo/domain/dsl/state_machine.py`

- Add Pydantic model:
  - `class TransitionRule(BaseModel):`
    - `from_state: str` (field name `from` in YAML; model alias)
    - `on: Literal["success", "failure", "pause"]`
    - `to: str`
    - `when: Optional[str] = None` (expression)
    - Internal (not serialized): `when_fn: Optional[Callable[[Any, Any], Any]] = PrivateAttr(default=None)`

- Extend `StateMachineStep`:
  - `transitions: list[TransitionRule] = Field(default_factory=list)`
  - `@model_validator(mode="after")`:
    - Ensure `to` is either a key in `states` or listed in `end_states`.
    - Ensure `from` is `*` or present in `states`.
    - Precompile `when` using `compile_expression_to_callable`, store in `when_fn`.

Notes
- Use Pydantic `Field(alias="from")` to accept `from:` in YAML while mapping to `from_state` internally.
- Validation errors should produce clear `BlueprintError` messages when raised via loader.

### 6.2 Blueprint Loader

File: `flujo/domain/blueprint/loader.py`

- In the StateMachine branch:
  - Parse `transitions` if present. Coerce to `TransitionRule` instances via `StateMachineStep` validation.
  - Maintain existing `states`, `start_state`, `end_states` handling.
  - Backward compatibility: if `transitions` missing, feature is inactive.

### 6.3 Policy Execution

File: `flujo/application/core/step_policies.py` (StateMachinePolicyExecutor)

Core loop changes (high-level algorithm):
1) Determine `current_state` from `context.scratchpad.current_state` or `step.start_state`.
2) At the start of each hop, write `context.scratchpad.current_state = current_state` (control metadata only; not part of iteration context) so resume knows where we were.
3) Build `iteration_context = ContextManager.isolate(context)`.
4) Run the state sub-pipeline via `core._execute_pipeline_via_policies` with cache disabled (unchanged).
5) On completion:
   - Aggregate totals and step_history (unchanged).
   - Merge sub-context into main context (unchanged) for success/failure paths.
   - Compute `event` as success/failure.
   - Resolve transition via `self._resolve_transition(step, current_state, event, output, context)` (see below).
   - If rule found: set `next_state = rule.to` and update `scratchpad.current_state = rule.to`.
   - Else: fallback to legacy `next_state` resolution from context or step outputs.
6) On pause (`PausedException`):
   - Compute `event = "pause"`.
   - Resolve transition with the current main context (do NOT merge iteration context).
   - If rule found: set `scratchpad.current_state = rule.to` and optionally `scratchpad.next_state = rule.to`.
   - Re-raise `PausedException` immediately.
7) Terminate when `current_state in end_states` or maximum hops reached.

Helper: `_resolve_transition(step, from_state, event, output, context)`
- Iterate `step.transitions` in order.
- Match `from_state` (or `*`) and `on == event`.
- If `when_fn` exists: call with `output=payload, context=context_proxy`; treat truthiness.
- Return the first match (first-match-wins) or `None`.

Expression payload (`output`) for `when` evaluation
- `{ "event": <event>, "last_output": <last_step_output_or_none>, "last_step": <last_step_result_summary> }`
- Keep it small and read-only; the evaluator will also see `context` via `TemplateContextProxy`.

Telemetry additions
- Log on every hop: starting state, event, rule match, applied transition, and terminal detection.
- Examples:
  - `[StateMachinePolicy] event=success from='clarification' matched to='concept_discovery'`
  - `[StateMachinePolicy] pause detected in state='clarification'; transitioning to='clarification' and re-raising`

### 6.4 Backward Compatibility & Precedence

- If `transitions` is empty/missing, behavior is unchanged.
- If transitions exist:
  - Transitions take precedence over implicit `scratchpad.next_state` produced by tools.
  - Fallback to legacy `next_state` only when no rule matches.
- `end_states` remain terminal regardless of transitions.

### 6.5 Security & Safety

- Use the existing safe expression engine. `when` expressions evaluate with:
  - `output`: a small payload dict containing `event`, `last_output`, `last_step`.
  - `context`: the pipeline context (via TemplateContextProxy).
  - Additionally, the engine exposes `previous_step` and `steps` for parity with other DSL expressions.
- Do not execute user code; only evaluate restricted AST.
- On expression failure, treat as non-match (log warning), do not crash policy.

## 7. Implementation Tasks & Checklist

### 7.1 Schema & Loader
- [x] Add `TransitionRule` model with aliases and validation.
- [x] Extend `StateMachineStep` with `transitions` and model validator to precompile `when`.
- [x] Update loader to accept `transitions` and attach to `StateMachineStep`.
- [x] Clear error messages for invalid `from`/`to`/`on` values.

### 7.2 Policy
- [x] Add `current_state` control metadata write at hop entry.
- [x] Wrap sub-pipeline execution with try/except for `PausedException` and apply rule resolution for `on: pause` before re-raising.
- [x] Implement `_resolve_transition` helper with first-match-wins semantics and `when_fn` evaluation.
- [x] Apply transition precedence over legacy `next_state` extraction.
- [x] Add structured telemetry logs.

### 7.3 Docs & Examples
- [x] Add docs section for StateMachine transitions with YAML samples and event semantics.
- [x] Update examples (architect pipeline) to show `on: pause` self-transition.
- [x] Add migration note: optional feature, BC maintained.

### 7.4 Tests

Unit tests (DSL & Loader)
- [x] Parse valid transitions including wildcard `from: "*"`.
- [x] Reject `to` states not in `states|end_states`.
- [x] Reject `from` states not in `states` (except `*`).
- [x] Reject invalid `on` values.
- [x] Load `when` expressions (compiled at load time; runtime errors treated as non-match).

Unit tests (Policy)
- [x] Success path: rule match moves to expected state; telemetry message present.
- [x] Failure path: wildcard rule directs to `failed` end-state.
- [x] Pause path: rule re-enters same state; current_state persisted; `PausedException` re-raised.
- [x] No matching rule: fallback to legacy next_state from outputs/context.
- [x] End-state detection (covered by existing tests).
- [x] Unknown-state handling (covered by existing tests).

Integration tests
- [x] YAML-defined orchestrator with pause transition and self re-entry; verify paused context and control metadata.
- [x] YAML-defined orchestrator across 3+ states; verify control flow and final state.
- [x] Conditional `when` depending on `context.scratchpad` values; verify both true/false branches.

Regression tests
- [x] Pipelines without `transitions` behave exactly as before (verified via legacy next_state flow and step_history assertions; existing suites also cover baseline behavior).
- [x] Ensure `breach_event` not used/introduced; guard added asserting StateMachine policy does not reference it.
- [x] Quota semantics unchanged; totals aggregate as before (unit test aggregates costs/tokens across states).
- [x] State history telemetry still coherent; existing parsing tests pass, and added assertions on step_history integrity.

Performance tests
- [ ] Measure overhead of rule resolution across 100 hops; verify negligible impact (<1% vs baseline) under typical pipelines.

## 8. Acceptance Criteria

- Authors can add a `transitions` block in YAML; rules are applied deterministically with first-match-wins.
- Pause handling is robust: on HITL pause, resume continues in the declared `to` state when a `pause` rule exists.
- No changes are required for existing pipelines; behavior is backward compatible when `transitions` is absent.
- All tests pass under `make all` including mypy strictness.
- No direct env/TOML reads introduced in policies/domain.

## 9. Telemetry & Observability

- New logs in `StateMachinePolicyExecutor`:
  - Hop start: `starting at state=...`
  - Rule match: `event={event} from={state} matched to={to}`
  - Pause handling: `pause detected in state=...; transitioning to=...`
  - Terminal: `reached terminal state=...`
- Span annotations remain unchanged; optional: add `executed_transition_from`, `executed_transition_to` attributes.

## 10. Risks & Mitigations

- Mis-specified transitions causing loops: mitigate with `max_hops = len(states) * 10` cap (already present); clear telemetry aids debugging.
- Expression misuse: safe evaluator guards; invalid expressions treated as non-match; provide crisp error messages during validation where possible.
- Context poisoning on pause: we never merge iteration context on pause; only write control metadata (`current_state`) to main context.

## 11. Rollout Plan

- Implement behind “presence of transitions” behavior (implicit feature flag).
- Land PR with unit + integration tests and docs.
- Release notes: highlight declarative transitions and pause handling improvement.

## 12. Work Estimates & Ownership

- Schema/Loader: 1–2 days
- Policy changes: 2–3 days
- Tests (unit/integration/regression): 2–3 days
- Docs & examples: 0.5–1 day

Owner: Core Runtime Team
Reviewers: DSL Owner, Policy/Executor Owner, Docs Lead

## 13. Open Questions

- Should transitions to undefined but declared end states be allowed without `states[...]` bodies? Current design: yes, if listed in `end_states`.
- Should `when` be allowed to reference `output.last_step.metadata_`? Current design: expression engine can reach it via `last_step` summary (we’ll expose a minimal dict).
- Should we synthesize a `failure` rule by default if none provided? Current design: no; maintain explicitness.

## 14. Pseudocode Snippets

Transition resolution in policy:

```python
def _resolve_transition(step, from_state, event, payload, context):
    for rule in getattr(step, "transitions", []) or []:
        if rule.on != event:
            continue
        if rule.from_state != "*" and rule.from_state != from_state:
            continue
        if rule.when_fn is not None:
            try:
                if not bool(rule.when_fn(payload, context)):
                    continue
            except Exception:
                telemetry.logfire.warning("[StateMachinePolicy] when() evaluation failed; skipping rule")
                continue
        return rule
    return None
```

Pause handling (simplified):

```python
try:
    pr = await core._execute_pipeline_via_policies(...)
    event = "success" if pr.step_history and pr.step_history[-1].success else "failure"
    rule = _resolve_transition(step, current_state, event, payload, last_context)
    if rule:
        set_next_state(rule.to)
    else:
        legacy_next_state_fallback()
except PausedException:
    payload = {"event": "pause", ...}
    rule = _resolve_transition(step, current_state, "pause", payload, last_context)
    if rule:
        set_current_state(rule.to)  # control metadata
    raise
```

## 15. Documentation Tasks

- New doc page: “StateMachine Transitions”
  - Syntax: fields, wildcard, events
  - `when` expressions; available variables
  - Pause/resume patterns and HITL notes
  - Precedence over legacy `next_state`
- Update examples and templates to demonstrate common patterns.

## 16. Regression Guardrails

- Grep-based CI check to ensure no `breach_event` usage is reintroduced.
- Ensure `ExecutorCore` remains free of StateMachine-specific logic.
- Keep coverage for pause paths and no-transition paths.

---

This FSD is the single source of truth for implementation and validation of declarative transitions in `StateMachineStep`. All changes must pass `make all` (format, lint, typecheck, tests) prior to merge.
