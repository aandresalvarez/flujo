# CLI Flags, Project Root, and Exit Codes

This page summarizes the new CLI behavior added to improve CI and developer ergonomics.

## Global Flags

- `--project PATH`: Forces the project root. The directory is added to `PYTHONPATH` so imports like `skills.helpers` resolve when running from subdirectories or CI workspaces.
- `-v/--verbose`, `--trace`: Print full Python tracebacks for easier troubleshooting. Useful in CI logs.

Project root resolution order used by all commands:
1) `--project PATH`
2) `FLUJO_PROJECT_ROOT`
3) Auto-detect by looking up from the current directory for a folder that contains `pipeline.yaml` or `flujo.toml`.

When set, the project root is injected into `sys.path` automatically.

## Validate

- Top-level command: `flujo validate` (alias to the developer command)
- Strict-by-default: exits non-zero if errors are found.
  - Use `--no-strict` to relax and always return 0.
- CI-friendly output: `--format=json` emits a machine-readable payload:

```json
{
  "is_valid": true,
  "errors": [ { "rule_id": "...", "severity": "error", "message": "...", "step_name": "...", "suggestion": "..." } ],
  "warnings": [ ... ],
  "path": "/abs/path/to/pipeline.py|yaml"
}
```

Examples:
- `flujo validate` (uses project `pipeline.yaml` if path omitted)
- `flujo validate path/to/pipeline.yaml --format=json`
- `flujo --project . validate --no-strict`

Exit code in strict mode when invalid: `4` (see Exit Codes).

## Run

- `--dry-run`: Parse and validate only; do not execute the pipeline.
  - With `--json`, prints `{ "validated": true, "steps": ["..."] }`

Example:
- `flujo --project . run --dry-run --json`
- `FLUJO_PROJECT_ROOT=$PWD flujo run --input 'hello'`

## Error Messages and Troubleshooting

- Import errors now surface the actual missing module and hints:
  - `Import error: module 'skills.helpers' not found. Try setting PYTHONPATH=. or pass --project/FLUJO_PROJECT_ROOT`
- Add `-v` or `--trace` to print the full traceback.

## Stable Exit Codes

These codes are stable for CI and scripts:

- `0`: Success
- `1`: Runtime error (unhandled execution failure)
- `2`: Configuration/settings error
- `3`: Import/module resolution error
- `4`: Validation failed (strict mode)
- `130`: Interrupted by user (Ctrl+C)

## Quick CI Snippet

Example of gating on validation:

```sh
set -euo pipefail
json=$(flujo --project "$GITHUB_WORKSPACE" validate --format=json)
echo "$json" | jq .
if [ "$(echo "$json" | jq -r .is_valid)" != "true" ]; then
  echo "Validation failed" >&2
  exit 4
fi
```

