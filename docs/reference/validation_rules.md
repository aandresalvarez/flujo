# Validation Rules

This page lists the rule categories enforced by `flujo validate` and how to interpret and address them. Rule IDs are stable and can be used in suppressions, profiles, and fixers.

Categories

- Templates (V‑T1..V‑T6)
  - V‑T1: `previous_step.output` misuse. Use `previous_step | tojson` or `steps.<name>.output | tojson`.
  - V‑T2: `this` outside a map body.
  - V‑T3: Unknown/disabled filter.
  - V‑T4: `steps.<name>` reference to a non‑prior step.
  - V‑T5: Missing field on prior model for `previous_step.<field>`.
  - V‑T6: Non‑JSON where JSON is expected.

- Schema (V‑S1..V‑S3)
  - V‑S1: Basic JSON Schema issues (e.g., array without `items`).
  - V‑S2: Structured output then likely stringified downstream.
  - V‑S3: `type=string` awareness when structure is expected.

- Context (V‑C1..V‑C3)
  - V‑C1: `updates_context` without a mergeable output.
  - V‑C2: Legacy scratchpad root mapping is now an error (map to `import_artifacts.<key>` instead).
  - V‑C3: Extremely large literal in templates (performance risk).

- Agents (V‑A1..V‑A8)
  - V‑A1: Missing agent.
  - V‑A2: Type mismatch between steps.
  - V‑A3: Reusing the same Step instance.
  - V‑A4‑ERR: Signature analysis failed.
  - V‑A5: Unused output.
  - V‑A6: Unknown agent id/import path.
  - V‑A7: Invalid `max_retries`/`timeout` coercion.
  - V‑A8: Structured output requested with non‑JSON response mode.

- Orchestration / Flow
  - V‑P1: Parallel context merge conflicts.
  - V‑P2: Explicit import outputs conflict across branches.
  - V‑P3: Heterogeneous first‑step input types in branches.
  - V‑L1: Loop exit coverage heuristic.
  - V‑CF1: Unconditional infinite loop heuristic.
  - V‑SM1: State machine unreachable start/end paths.

- Imports
  - V‑I1: Missing import file.
  - V‑I2: Import outputs mapping sanity (unknown parent root).
  - V‑I3: Import cycle detected.
  - V‑I4: Aggregated child findings.
  - V‑I5: Input projection coherence.
  - V‑I6: Inherit conversation consistency.

Suppressions

- YAML comments: add `# flujo: ignore <RULE|GLOB...>` to a step entry.
- Programmatic: set `step.meta['suppress_rules'] = ['V-T*']`.

Profiles and Rules Files

- Use `--rules rules.json` (JSON) or `--rules rules.toml` (TOML) to override severities.
- Use `--rules strict` to pick a named profile from `flujo.toml` `[validation.profiles.strict]`.
  - Overrides support globs, e.g., `V-T* = "off"`.

Fixers

Safe, opt‑in fixers can automatically fix some issues in YAML:

- V‑T1: rewrite `previous_step.output` to `previous_step | tojson`.
- V‑T3: correct common filter typos (`to_json`→`tojson`, `lowercase`→`lower`, etc.).
- V‑C2: replace `parent: scratchpad` with `parent: import_artifacts.<key>` (uses `import_artifacts.value` as a conservative default key).

Run with `--fix` (preview + apply) or `--fix-dry-run` (patch only), and restrict with `--fix-rules`.

SARIF

- `--format sarif` emits SARIF 2.1.0 with rule IDs, names, and help URIs (catalog linked). Map rule IDs in your code scanning tool as needed.
