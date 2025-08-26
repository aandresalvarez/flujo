# FSD-028: Declarative LoopStep Init/Propagation/Output

Status: in_progress  
Status values: planned | in_progress | done

## Overview

- Goal: Reduce boilerplate for conversational and iterative workflows by adding declarative `init`, `propagation`, and `output` blocks to `loop` steps in YAML blueprints.
- Why: Today, common loop patterns (e.g., clarification loops) require 2–3 Python mappers. This hides intent, scatters logic, and slows down users.
- Approach: Compile new YAML blocks into existing loop mappers in the loader, plus a minimal policy hook for running `init` safely. Preserve all Flujo architectural guarantees.

## User Value & Roadmap Alignment

- Phase 1 (Generate): Fewer YAML lines and no helper modules increase Time‑to‑First‑Run and first-try success for `flujo create` generated pipelines.
- Phase 2 (Debug/Refine): Intent becomes visible in YAML; easier to reason about diffs and tune, while traces remain unchanged.
- Phase 3 (Teams): Clearer PR diffs and fewer moving parts (no ad‑hoc helpers) aid review and governance.

Success indicators
- ≥ 80% of conversational loops can be authored without custom mappers.
- −40–60% YAML lines for typical loops (vs. current examples).
- No regressions on pause/resume, quotas, or typed outcomes.

## Non‑Goals

- No change to default loop exit semantics or retry semantics.
- No new I/O behavior; CLI input precedence remains unchanged.
- No direct env or `flujo.toml` reads from policies/domain to drive these features.

## Proposed UX (minimal YAML)

The blocks are optional. If omitted, behavior matches today.

- Init (runs once before 1st iteration, on the isolated iteration context):
  loop.init supports `set`, `append`, and `merge` operations targeting `context.*` only.

- Propagation (defines next iteration’s input):
  `loop.propagation.next_input` accepts presets `context`, `previous_output`, or a template.

- Output (final loop output mapping):
  Either `loop.output_template: "..."` (single string), or `loop.output: { ... }` (object mapping).

Example (concise conversational loop):

```yaml
- kind: loop
  name: clarification_loop
  loop:
    init:
      - append: { target: "context.scratchpad.history", value: "User: {{ steps.get_goal.output }}" }
    body:
      # steps elided for brevity; they use updates_context: true
    propagation:
      next_input: context  # common case
    exit_expression: "context.scratchpad.last_agent_command.action == 'finish'"
    output_template: "{{ context.scratchpad.history | join('\n') }}"
```

## Natural-Language Presets (friendly YAML)

Audience: domain experts (e.g., medical scientists) who are smart but not YAML/templating experts. These presets read like instructions and compile to the same safe internals.

Principles
- Plain words over templates: prefer presets to `{{ ... }}`.
- Defaults everywhere: most loops should work with 2–4 lines.
- Progressive disclosure: advanced templates still work when needed.

New friendly keys (all optional)
- `conversation: true`
  - Ensures `context.scratchpad.history` exists (list)
  - Sets `propagation: context` as default
  - Sets the common exit alias `stop_when: agent_finished`
- `stop_when: agent_finished`
  - Alias for `exit_expression: "context.scratchpad.last_agent_command.action == 'finish'"`
- `propagation: context | previous_output`
  - Shorthand for `loop.propagation.next_input: context` (or default `previous_output`)
- `init.history.start_with: { from_step: <name>, prefix: "User: " }`
  - Seeds conversation history with a value derived from another step’s output
- `init.notes.set: "..."`
  - Writes to `context.scratchpad.notes` (string)
- `output: text: conversation_history`
  - Returns a newline-joined `context.scratchpad.history`
- `output: fields: { goal: initial_prompt, clarifications: conversation_history }`
  - Builds a final object from known fields; `conversation_history` expands to joined history

Friendly example (no templates)

```yaml
- kind: loop
  name: clarification_loop
  loop:
    conversation: true              # history + context propagation + agent_finished
    init:
      history:
        start_with:
          from_step: get_goal
          prefix: "User: "
    body:
      # steps elided for brevity; use updates_context: true
    output:
      text: conversation_history
```

Loader mapping (aliases → compiled behavior)
- `conversation: true` → ensure `context.scratchpad.history = []` if missing; set `propagation: context`; set `stop_when: agent_finished` if neither an explicit `exit_expression` nor `exit_condition` is provided
- `stop_when: agent_finished` → install the exit expression shown above (unless explicit exit already provided)
- `propagation: context|previous_output` → set `iteration_input_mapper` preset
- `init.history.start_with` → compiled to a first-iteration `append` op resolving `steps.<name>.output` with optional `prefix`
- `init.notes.set` → compiled to a first-iteration `set` op targeting `context.scratchpad.notes`
- `output.text: conversation_history` → compiled to `loop_output_mapper` that joins `context.scratchpad.history` with `"\n"`
- `output.fields` → compiled to `loop_output_mapper` that constructs an object; supports `initial_prompt` and `conversation_history` shorthands

## Architecture & Principles Compliance

- Policy‑Driven Execution: Execution logic stays in the Loop policy. The loader compiles declarative blocks to functions assigned on the LoopStep. A small policy hook runs `init` at the correct time on the isolated iteration context.
- Control Flow Exception Safety: All control‑flow exceptions (PausedException, PipelineAbortSignal, InfiniteRedirectError) are re‑raised. Declarative functions must not catch/convert them.
- Context Idempotency: Each iteration operates on `ContextManager.isolate(current_context)`; `init` applies to that isolation before Iteration 1. Merges only occur after successful iterations per existing policy behavior.
- Proactive Quota System: No changes to Reserve → Execute → Reconcile. No governor/breach_event. Limits continue to error precisely with preserved partial history.
- Centralized Configuration: No direct env/TOML reads; if future toggles are needed, use `config_manager` helpers.
- Agent Creation: No changes; feature acts at blueprint and policy layers.

## Detailed Design

1) Loader compilation (primary)
- File: `flujo/domain/blueprint/loader.py`
- Add support under `kind: loop` for optional keys:
  - `loop.init: [ { set|append|merge }... ]`
  - `loop.propagation.next_input: <preset|string template>`
  - `loop.output_template: <string template>`
  - `loop.output: <mapping of templates>` (mutually exclusive with `output_template`)
  - Friendly aliases (natural language):
    - `loop.conversation: true`
    - `loop.stop_when: agent_finished`
    - `loop.propagation: context | previous_output`
    - `loop.init.history.start_with: { from_step: <name>, prefix?: <str> }`
    - `loop.init.notes.set: <str>`
    - `loop.output.text: conversation_history`
    - `loop.output.fields: { goal?: initial_prompt, clarifications?: conversation_history, ... }`
- Reuse the existing “Declarative loop state sugar (M4)” machinery for `append/set/merge` ops (TemplateContextProxy, AdvancedPromptFormatter, target resolution restricted to `context.*`).
- Compile these blocks into callables:
  - `compiled_init_ops(ctx, prev_output) -> None` applying ops against the iteration context.
  - `iteration_input_mapper(prev_output, ctx, i) -> Any` honoring `next_input` presets:
    - `context` → return `ctx`
    - `previous_output` (default) → return `prev_output`
    - template → format with `{context, previous_step, steps}` and return rendered value (auto JSON decode when the template looks like JSON; otherwise return string).
  - `loop_output_mapper(output, ctx) -> Any`:
    - if `output_template`: format once at the end
    - if `output` mapping: format each field and return object
- Store `compiled_init_ops` on the step meta for the policy to invoke at Iteration 1, e.g., `step.meta["compiled_init_ops"] = fn`.
  - When friendly keys are present, compile them into the same callables (no new policy types).

2) Policy hook (minimal)
- File: `flujo/application/core/step_policies.py` (DefaultLoopStepExecutor)
- At the start of each iteration, right after `iteration_context = ContextManager.isolate(current_context)`, add:
  - If `iteration_count == 1` and `compiled_init_ops` exists in `loop_step.meta`, invoke it with `(iteration_context, current_data)` inside the same try block that already handles HITL `PausedException` correctly.
  - Do not catch control flow exceptions. Any other exception falls back to existing error handling (iteration mapper error path is acceptable if thrown when computing next_input; init exceptions should produce a failure StepResult with `exit_reason: initial_input_mapper_error`).
- No changes to quota handling or exit condition logic.

3) Templates & formatting
- Use `AdvancedPromptFormatter` + `TemplateContextProxy` already used by loader sugar.
- Provide `{context}`, `{previous_step}`, and `{steps}` variables. Preserve current joining/filter behaviors (e.g., `| join('\n')`).
- For `merge` ops, require rendered value to be a JSON mapping; ignore invalid values safely.
 - For friendly keys, avoid templates; when needed, the loader constructs the appropriate operations and mappers without exposing templating syntax to users.

4) Defaults (to minimize YAML)
- Omit `propagation` → identical to today: next input equals previous output.
- Omit `output`/`output_template` → identical to today: final loop output equals last iteration output, or MapStep’s collected results.
- Omit `init` → no initialization performed.

## Backward Compatibility

- All new keys are optional. Existing blueprints and recipes continue to work.
- Existing Python mappers remain supported; declarative blocks, if present, override them.
- `agentic_loop`’s special `output_template` stays intact and is harmonized with the general loop `output_template` behavior.

## Risks & Mitigations

- Incorrect context mutation in `init`: Apply only on iteration context isolation; never mutate the shared `current_context` directly. Covered by tests.
- Control flow swallowing: Ensure the policy hook re‑raises PausedException and similar. Add tests with HITL pause inside the first iteration.
- Template runtime errors: Loader compilers are defensive; runtime formatting errors surface as mapper errors with clear feedback.

## Test Plan

- Unit (loader):
  - Parses `loop.init/propagation/output[_template]` and compiles expected callables.
  - Parses friendly aliases: `conversation`, `stop_when`, `propagation` preset, `init.history.start_with`, `init.notes.set`, and `output.text/fields`.
  - Invalid targets (non‑`context.*`) are ignored.
  - Template rendering with `{context, previous_step, steps}` variables.
- Integration (policy):
  - `init` runs exactly once on isolated iteration context; values appear in pipeline context during the first body step.
  - `propagation: context` cycles full context through iterations.
  - `output_template` returns formatted final value.
  - Friendly outputs (`text: conversation_history`, `fields`) produce expected results.
  - Paused HITL step in Iteration 1 pauses the loop; no swallowing.
  - Quota breach preserves partial history and precise messages.
- E2E: Conversational clarification loop authored with only YAML runs end‑to‑end (no Python helpers), including pause/resume.

## Acceptance Criteria

- Users can remove `initial_input_mapper`, `iteration_input_mapper`, and `loop_output_mapper` from common loops.
- No regressions in `make all` (format, lint, mypy strict, tests, coverage).
- Observability: Loop spans and step histories remain unchanged; metadata includes evaluated exit expression when applicable.

## Work Breakdown

Execution is sequential and test‑gated. Do not start the next item until the prior item’s Exit Tests pass.

1. Loader: core declarative blocks  
   Status: done  
   Exit Tests: loader unit tests parse/compile `init`, `propagation.next_input`, `output[_template]`; invalid targets ignored.

2. Policy: iteration‑1 init hook  
   Status: done  
   Exit Tests: integration test shows init applies on isolated context; HITL pause re‑raises (no swallowing); idempotency preserved.

3. Aliases: friendly YAML presets  
   Status: done  
   Exit Tests: loader unit tests for `conversation`, `stop_when`, `propagation` preset, `init.history.start_with`, `init.notes.set`, `output.text/fields`.

4. E2E example (natural form)  
   Status: done  
   Exit Tests: end‑to‑end run of conversational loop without Python helpers, including pause/resume.

5. Docs & examples  
   Status: done  
   Exit Tests: examples render in docs site; copy‑paste runs locally.

6. Changelog entry  
   Status: done  
   Exit Tests: CHANGELOG updated under Unreleased → Added.

7. CI Gate  
   Status: planned  
   Exit Tests: `make all` passes (format, lint, mypy strict, tests, coverage).

## Next Steps (Milestones & Status)

These are queued after core delivery. Each item is independently testable and must pass its Exit Tests before starting the next.

NS1. Propagation preset: `propagation: auto`  
Status: done  
Design: chooses `context` when any body step uses `updates_context: true`, else `previous_output`. Explicit user setting always wins.  
Exit Tests: integration pipeline where body updates context → next input is full context; pipeline without updates → previous output; explicit `propagation` overrides auto; idempotency maintained.

NS2. MapStep equivalents: `map.init` and `map.finalize` sugars  
Status: in_progress  
Design: pre/post hooks compiled like loop init/output; run once per mapping run (not per item).  
Exit Tests: mapping pipeline applies `map.init` exactly once before first item; `map.finalize` once after last; results aggregated correctly; concurrency safety via ContextVar; mypy strict.

NS3. Parallel reduce sugar: `parallel.reduce`  
Status: done  
Design: declarative reducers (union/concat/dedupe/keys) applied to branch outputs with stable branch order and quota split preserved.  
Exit Tests: multi‑branch pipeline validates each reducer; order preserved; quotas split deterministically; error messages precise; no governor/breach_event.

NS4. CLI boosters  
Status: in_progress  
Design: `flujo create --wizard` emits natural YAML; `flujo explain` summarizes YAML intent in plain language.  
Exit Tests: golden‑file tests for wizard outputs across scenarios; explain renders concise, accurate summaries; help text updated.

## Rollout

- Ship behind no flags; behavior is additive and backward compatible.
- Announce as “Declarative Loop Enhancements” with before/after diff in README and a new example.

## Future Extensions (out of scope for this PR)

- `propagation: auto` preset (choose `context` when any body step uses `updates_context: true`).
- MapStep equivalents: `map.init` and `map.finalize` sugars for pre/post hooks.
- Parallel reduce sugar: declarative `parallel.reduce` for branch result merging.
 - CLI boosters: `flujo create --wizard` to generate the natural form interactively; `flujo explain` to summarize YAML intent in plain language.
