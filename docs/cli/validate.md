# flujo validate

Validate a pipeline file (YAML or Python) with structural, templating, import, and orchestration checks.

Overview

`flujo validate` statically analyzes your pipeline (YAML or Python) for common correctness risks before running it. It checks for missing agents, type mismatches, templating hazards, import graph issues, and orchestration pitfalls. Findings include a rule id, severity, a message, and suggestions. For YAML files, findings include file, line, and column when available.

Usage:

```
uv run flujo validate [PATH] [--strict/--no-strict] [--format text|json|sarif] [--imports/--no-imports] [--fail-on-warn] [--rules FILE|PROFILE]
```

Options:
- `--strict/--no-strict` (default: strict): exit non-zero on errors.
- `--format`: output format. `text` (human-friendly), `json` (machine parsable), or `sarif` (2.1.0 for code scanning).
- `--imports/--no-imports` (default: imports): recursively validate imported blueprints.
- `--fail-on-warn`: treat warnings as errors (non-zero exit).
- `--rules`: either a path to a JSON/TOML mapping of rule severities (off|warning|error) with glob support (e.g., `{"V-T*":"off"}`), or a named profile from `flujo.toml` under `[validation.profiles.<name>]`.

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
```

Exit Codes

- `--strict` (default): non-zero exit when errors are present.
- `--fail-on-warn`: non-zero exit when warnings are present (after applying rule overrides/profiles).

See Also

- Rule catalog: `docs/validation_rules.md`
- SARIF in CI: `docs/ci/sarif.md`
