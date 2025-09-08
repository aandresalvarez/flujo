# flujo validate

Validate a pipeline file (YAML or Python) with structural, templating, import, and orchestration checks.

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
- Comment-based suppressions: add `# flujo: ignore <RULES...>` to a step or list item to suppress findings for that step. Supports multiple rules and globs, e.g., `# flujo: ignore V-T1 V-P3` or `# flujo: ignore V-*`.
- File/line/column enrichment: when validating a YAML file, findings include filename, line and column for the step location.

Examples:

```
uv run flujo validate pipeline.yaml --format json
uv run flujo validate pipeline.yaml --rules rules.json
uv run flujo validate pipeline.yaml --rules strict --fail-on-warn
uv run flujo validate pipeline.yaml --format sarif > findings.sarif
```

