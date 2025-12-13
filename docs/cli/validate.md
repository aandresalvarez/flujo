# flujo validate

Validate a pipeline file (YAML or Python) with structural, templating, import, and orchestration checks.

Overview

`flujo validate` statically analyzes your pipeline (YAML or Python) for common correctness risks before running it. It checks for missing agents, type mismatches, templating hazards, import graph issues, and orchestration pitfalls. Findings include a rule id, severity, a message, and suggestions. For YAML files, findings include file, line, and column when available.

Usage:

```
uv run flujo validate [PATH] [--strict/--no-strict] [--format text|json|sarif] [--imports/--no-imports] [--fail-on-warn] [--rules FILE|PROFILE] [--baseline FILE] [--update-baseline]
```

Options:
- `--strict/--no-strict` (default: strict): exit non-zero on errors.
- `--format`: output format. `text` (human-friendly), `json` (machine parsable), or `sarif` (2.1.0 for code scanning).
- `--imports/--no-imports` (default: imports): recursively validate imported blueprints.
- `--fail-on-warn`: treat warnings as errors (non-zero exit).
- `--rules`: either a path to a JSON/TOML mapping of rule severities (off|warning|error) with glob support (e.g., `{"V-T*":"off"}`), or a named profile from `flujo.toml` under `[validation.profiles.<name>]`.
- `--baseline FILE`: compare the current report to a previous JSON report; output reflects only added findings; prints a delta summary.
- `--update-baseline`: write the current (post-baseline) view back to `--baseline`.
- `--fix`: preview and apply safe, opt‑in auto‑fixes (see Fixers below).
- `--yes`: assume yes to prompts when using `--fix`.
- `--fix-rules`: comma‑separated list of fixer rules/globs to apply (e.g., `V-T1,V-C2*`).
- `--fix-dry-run`: print a unified diff patch without writing any changes (still shows metrics in JSON).

Features:
- Comment-based suppressions: add `# flujo: ignore <RULES...>` to a step mapping or list item to suppress findings for that step. Supports multiple rules and globs, e.g., `# flujo: ignore V-T1 V-P3` or `# flujo: ignore V-*`.
- Per-step metadata suppressions (programmatic): set `step.meta['suppress_rules'] = ["V-T*", "V-*"]` before calling `validate_graph()`.
- File/line/column enrichment: when validating a YAML file, findings include filename, line and column for the step location.
- Recursive imports: with `--imports` (default), imported blueprints are validated and findings aggregated at the parent step.

Rule Profiles & Rules File

- JSON/TOML rules file (override severities):
  - JSON example: `{ "V-T*": "off", "V-P3": "warning" }`
  - TOML example:
    ```toml
    [validation.rules]
    V-T* = "off"
    V-P3 = "warning"
    ```
  - Apply with: `--rules rules.json` or `--rules rules.toml`.

- Profiles in `flujo.toml`:
  ```toml
  [validation.profiles.strict]
  V-T* = "error"
  V-A5 = "warning"

  [validation.profiles.ci]
  V-*  = "warning"
  V-A1 = "error"
  ```
  Apply with: `--rules strict` or `--rules ci`.

Examples:

```
uv run flujo validate pipeline.yaml --format json
uv run flujo validate pipeline.yaml --rules rules.json
uv run flujo validate pipeline.yaml --rules strict --fail-on-warn
uv run flujo validate pipeline.yaml --format sarif > findings.sarif
uv run flujo validate pipeline.yaml --fix --yes --fix-rules V-T1,V-T3
uv run flujo validate pipeline.yaml --fix --fix-dry-run --fix-rules V-C2 --format json
```

Exit Codes

- `--strict` (default): non-zero exit when errors are present.
- `--fail-on-warn`: non-zero exit when warnings are present (after applying rule overrides/profiles).
- When `--baseline` is provided, exit codes reflect the post‑baseline view (i.e., only newly added findings count).

See Also

- Rule catalog: `reference/validation_rules.md`
- SARIF in CI: `docs/ci/sarif.md`

JSON fields

- `errors`, `warnings`, `is_valid`, `path` as expected.
- `counts` (when `FLUJO_CLI_TELEMETRY=1`): `{ "error": {"V-A1": 2}, "warning": {"V-T1": 1} }`.
- `baseline`: included when `--baseline` is used, with added/removed findings.
- `fixes`: present when `--fix` is used, e.g. `{ "applied": {"V-T1": 2}, "total_applied": 2 }`.
- `fixes_dry_run`: boolean present when `--fix-dry-run` is used.

Fixers

Safe, opt‑in fixers are applied only to YAML files and prompt for confirmation (unless `--yes` or CI JSON mode). Current fixers:

- V‑T1: Rewrite `previous_step.output` → `previous_step | tojson`.
- V‑T3: Correct common filter typos (e.g., `to_json`→`tojson`, `lowercase`→`lower`).
- V‑C2: Replace `parent: scratchpad` with `parent: import_artifacts.<key>` (uses `import_artifacts.value` as a conservative default key).

Use `--fix-rules` to restrict which fixers run, and `--fix-dry-run` to preview the patch without writing.
