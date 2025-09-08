# Validation Rule Catalog

Each finding is identified by a rule ID. Severities are `error` or `warning`. You can override severities via `--rules` or profiles in `flujo.toml`.

Suppression:
- Inline comments in YAML: `# flujo: ignore <RULES...>` at the step mapping or list item.
- Per-step metadata: `meta.suppress_rules: ["V-T*", "V-*" ]` when constructing steps programmatically.

Template Rules:
- V-T1: previous_step.output misuse — use `previous_step | tojson` or `steps.<name>.output | tojson`.
- V-T2: `this` misuse outside map bodies — use only inside map.
- V-T3: Unknown/disabled filter — add to allow-list in flujo.toml or fix misspelling.
- V-T4: Unknown `steps.<name>` reference — ensure the named step is prior.

Import & Composition:
- V-I1: Import existence — imported YAML path not found.
- V-I2: Import outputs mapping sanity — unknown parent root; prefer `scratchpad.<key>`.
- V-I3: Cyclic imports — typically raised by loader at compile time; validation guards against recursion.

Parallel & Orchestration:
- V-P1: Parallel context merge conflict risk — CONTEXT_UPDATE without field_mapping.
- V-P3: Parallel branch input uniformity — branches expect heterogeneous input types.
- V-SM1: StateMachine transitions validity — unknown states or unreachable paths.

Context & Persistence:
- V-C1: updates_context without mergeable output — scalar outputs aren’t merged; may be dropped.

Agents & Providers:
- V-A1: Missing agent on step.
- V-A5: Unused output — produced output is not consumed by the next step and not merged.

For a full list and examples, see the CLI docs and inline suggestions printed by `flujo validate`.

