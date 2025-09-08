# Validation Rule Catalog

Each finding is identified by a rule ID. Severities are `error` or `warning`. You can override severities via `--rules` or profiles in `flujo.toml`.

Suppression:
- Inline comments in YAML: `# flujo: ignore <RULES...>` at the step mapping or list item.
- Per-step metadata: `meta.suppress_rules: ["V-T*", "V-*" ]` when constructing steps programmatically.

Template Rules

- V‑T1: previous_step.output misuse
  - Why: `previous_step` is a raw value and does not have an `.output` attribute; templating will render `null`.
  - Fix: use `{{ previous_step | tojson }}` or `{{ steps.<name>.output | tojson }}`.
  - Example:
    ```yaml
    # ❌
    input: "{{ previous_step.output }}"
    # ✅
    input: "{{ previous_step | tojson }}"
    ```

- V‑T2: `this` misuse outside map bodies
  - Why: `this` is only defined inside a map body.
  - Fix: restrict to map body or bind to a named variable available in scope.

- V‑T3: Unknown/disabled filter
  - Why: filter not present in the allowed set or configured allow-list.
  - Fix: edit `flujo.toml` `[settings].enabled_template_filters` or correct typos.

- V‑T4: Unknown `steps.<name>` reference
  - Why: invalid step name or referenced before it exists.
  - Fix: correct the step name or move the reference to a later step.

Import & Composition

- V‑I1: Import existence
  - Why: imported YAML path cannot be resolved.
  - Fix: correct the path relative to the parent YAML or ensure the file exists.

- V‑I2: Import outputs mapping sanity
  - Why: parent mapping uses an unknown root (e.g., `badroot.value`).
  - Fix: map under `scratchpad.<key>` or a known context field.

- V‑I3: Cyclic imports
  - Why: import graphs must be acyclic. Loader typically raises at compile time; validation guards recursion.
  - Fix: remove the cycle or redesign import structure.

Parallel & Orchestration

- V‑P1: Parallel context merge conflict risk
  - Why: default `CONTEXT_UPDATE` without `field_mapping` may merge conflicting keys.
  - Fix: add `field_mapping` or choose an explicit merge strategy.

- V‑P3: Parallel branch input uniformity
  - Why: branches expect heterogeneous input types but receive the same input.
  - Fix: add adapter steps per branch or unify input types.

- V‑SM1: StateMachine transitions validity
  - Why: invalid states or no path to an end state.
  - Fix: correct state names and transition rules.

Context & Persistence

- V‑C1: updates_context without mergeable output
  - Why: non-dict outputs cannot be merged into context; they may be dropped.
  - Fix: switch to an object output or provide an `outputs` mapping.

Agents & Providers

- V‑A1: Missing agent on step
  - Why: simple steps require an agent.
  - Fix: configure an `agent` or use `Step.from_callable()`.

- V‑A5: Unused output
  - Why: the step’s output is not consumed or merged; likely a logic bug.
  - Fix: set `updates_context: true` or insert an adapter to consume the value.

For a full list and examples, see the CLI docs and inline suggestions printed by `flujo validate`.
