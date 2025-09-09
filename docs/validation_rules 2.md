# Validation Rule Catalog

Each finding is identified by a rule ID. Severities are `error` or `warning`. You can override severities via `--rules` or profiles in `flujo.toml`.

Suppression:
- Inline comments in YAML: `# flujo: ignore <RULES...>` at the step mapping or list item.
- Per-step metadata: `meta.suppress_rules: ["V-T*", "V-*" ]` when constructing steps programmatically.

## <a id="v-t1"></a>V‑T1 — previous_step.output misuse
  - Why: `previous_step` is a raw value and does not have an `.output` attribute; templating will render `null`.
  - Fix: use `{{ previous_step | tojson }}` or `{{ steps.<name>.output | tojson }}`.
  - Example:
    ```yaml
    # ❌
    input: "{{ previous_step.output }}"
    # ✅
    input: "{{ previous_step | tojson }}"
    ```
## <a id="v-t2"></a>V‑T2 — `this` misuse outside map bodies
  - Why: `this` is only defined inside a map body.
  - Fix: restrict to map body or bind to a named variable available in scope.

## <a id="v-t3"></a>V‑T3 — Unknown/disabled filter
  - Why: filter not present in the allowed set or configured allow-list.
  - Fix: edit `flujo.toml` `[settings].enabled_template_filters` or correct typos.

## <a id="v-t4"></a>V‑T4 — Unknown `steps.<name>` reference
  - Why: invalid step name or referenced before it exists.
  - Fix: correct the step name or move the reference to a later step.

## <a id="v-i1"></a>V‑I1 — Import existence
  - Why: imported YAML path cannot be resolved.
  - Fix: correct the path relative to the parent YAML or ensure the file exists.

## <a id="v-i2"></a>V‑I2 — Import outputs mapping sanity
  - Why: parent mapping uses an unknown root (e.g., `badroot.value`).
  - Fix: map under `scratchpad.<key>` or a known context field.

## <a id="v-i3"></a>V‑I3 — Cyclic imports
  - Why: import graphs must be acyclic. Loader typically raises at compile time; validation guards recursion.
  - Fix: remove the cycle or redesign import structure.

## <a id="v-p1"></a>V‑P1 — Parallel context merge conflict risk
  - Why: default `CONTEXT_UPDATE` without `field_mapping` may merge conflicting keys.
  - Fix: add `field_mapping` or choose an explicit merge strategy.

## <a id="v-p3"></a>V‑P3 — Parallel branch input uniformity
  - Why: branches expect heterogeneous input types but receive the same input.
  - Fix: add adapter steps per branch or unify input types.

## <a id="v-sm1"></a>V‑SM1 — StateMachine transitions validity
  - Why: invalid states or no path to an end state.
  - Fix: correct state names and transition rules.

## <a id="v-c1"></a>V‑C1 — updates_context without mergeable output
  - Why: non-dict outputs cannot be merged into context; they may be dropped.
  - Fix: switch to an object output or provide an `outputs` mapping.

## <a id="v-a1"></a>V‑A1 — Missing agent on step
  - Why: simple steps require an agent.
  - Fix: configure an `agent` or use `Step.from_callable()`.

## <a id="v-a5"></a>V‑A5 — Unused output
  - Why: the step’s output is not consumed or merged; likely a logic bug.
  - Fix: set `updates_context: true` or insert an adapter to consume the value.

For a full list and examples, see the CLI docs and inline suggestions printed by `flujo validate`.
