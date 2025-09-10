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

---

## <a id="v-t5"></a>V‑T5 — Prior model field existence
  - Why: When the previous output is a known Pydantic model, referencing a missing attribute is a likely authoring error.
  - Fix: Reference an existing field or adapt the model/output mapping.

## <a id="v-t6"></a>V‑T6 — Non‑JSON where JSON expected
  - Why: Templated input appears to be JSON but is not valid JSON while the consumer expects JSON.
  - Fix: Use `| tojson` or ensure the template renders valid JSON.

## (TODO) V‑S2 — Response format vs stringification
  - Why: Declared structured output (JSON object/schema) paired with immediate stringification hints a mismatch.
  - Fix: Map/object-consume fields directly; if stringification is intended (e.g., logging), suppress.

## <a id="v-s3"></a>V‑S3 — `type: string` awareness
  - Why: Pure string schemas can limit downstream structure.
  - Fix: Consider structured schema or document the rationale.

## (TODO) V‑L1 — Loop exit coverage
  - Why: Loop body and mappers suggest the loop may never reach its exit condition (no context updates, no iteration_input_mapper, no loop_output_mapper).
  - Fix: Provide an `iteration_input_mapper`, update context in the loop body, or add `loop_output_mapper` so the exit condition can be satisfied.

## (TODO) V‑P2 — Parallel explicit conflicts
- Why: Explicit branch outputs map to the same parent keys without a merge strategy.
- Fix: Add `field_mapping` or change strategy to avoid conflicts.

## <a id="v-c1"></a>V‑C1 — `updates_context` without mergeable output
  - Why: Non‑dict outputs cannot be merged into context; they may be dropped.
  - Fix: Switch to an object output or provide an `outputs` mapping.

## <a id="v-c2"></a>V‑C2 — Scratchpad shape conflicts
  - Why: Mapping directly to the `scratchpad` root may assign a non‑object and corrupt the expected dict shape.
  - Fix: Map to `scratchpad.<key>` or ensure the value is an object (dict-like).

## (TODO) V‑C3 — Large literals in templates
  - Why: Very large embedded constants harm performance and logs.
  - Fix: Move large data to files/variables; reference by key. Threshold configurable via `FLUJO_VALIDATE_LARGE_LITERAL_THRESHOLD` (characters).

## <a id="v-a6"></a>V‑A6 — Unknown agent id/import path
  - Why: Unresolvable `agent` (string path) will fail at runtime.
  - Fix: Use a valid `package.module:attr` or declare the agent under `agents:` and reference it.

## <a id="v-a7"></a>V‑A7 — Invalid `max_retries`/`timeout` coercion
  - Why: Non‑coercible values for agent retry/timeout controls lead to unexpected behavior.
  - Fix: Provide integer values; strings must be numeric (e.g., `"3"`). Invalid values are ignored or defaulted.

## (TODO) V‑A7 — Invalid `max_retries`/`timeout` coercion
- Why: Non‑coercible values lead to unexpected behavior.
- Fix: Provide integers/valid durations.

## (TODO) V‑A8 — Structured output with non‑JSON mode
- Why: Declared structured output but provider is not returning JSON.
- Fix: Enable JSON mode or remove structured expectations.

## (TODO) V‑I4 — Aggregated child findings
  - Why: Parent import steps should surface child findings with alias/file context.
  - Fix: Use `--imports` to aggregate; address errors in the child and re‑run. Summary V‑I4 indicates counts; individual findings retain their original rule IDs.

## <a id="v-i5"></a>V‑I5 — Input projection coherence
  - Why: Parent→child input projection (initial_prompt vs scratchpad) may not match the child's expected input shape.
  - Fix: If the child expects objects, use `input_to=scratchpad` or `both` with `input_scratchpad_key`; if it expects strings, include `initial_prompt`.

## (TODO) V‑I6 — Inherit context consistency
- Why: Inconsistent `inherit_context` settings across import boundaries lead to surprises.
- Fix: Align settings or scope mappings to avoid leaks.
