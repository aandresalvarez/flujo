# CLI Flags, Project Root, and Exit Codes

This page summarizes the new CLI behavior added to improve CI and developer ergonomics.

## Global Flags

- `--project PATH`: Forces the project root. The directory is added to `PYTHONPATH` so imports like `skills.helpers` resolve when running from subdirectories or CI workspaces.
- `-v/--verbose`, `--trace`: Print full Python tracebacks for easier troubleshooting. Useful in CI logs.
- Global flags go before the subcommand: for example, `flujo --verbose validate` (not `flujo validate --verbose`).
- Another example: `flujo --trace run --dry-run` (not `flujo run --trace --dry-run`).

Environment file path:
- Declare `env_file = ".env"` in `flujo.toml` to load API keys (e.g., `OPENAI_API_KEY`) from a specific file relative to the project root. The project scaffold provides `.env.example`; copy it to `.env` and fill your secrets. You can change the path to any file (e.g., `env_file = ".secrets"`).

API key precedence:
- Environment variables override everything (recommended for CI): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`.
- Then values from the `env_file` are considered (if configured).
- Finally, TOML `[settings]` values apply only when no env var is set.
  Tip: when running with `--debug`, consider printing which provider keys are present (never values) so you can confirm the effective source without leaking secrets.

Models and access:
- The example pipelines may reference future models (e.g., `openai:gpt-5`). Ensure your account has access, or switch to a widely available model like `openai:gpt-4o-mini`.
- If a model call is attempted without a real key, providers may fail after a few retries, making the run appear slow before failing with 401.
  - Use `flujo run --dry-run` to validate quickly without making provider calls.

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

## Paging and Colors

Some environments set `PAGER=less` and `LESS=-R`, which can cause Click/Typer to page output through `less`. If a command “hangs”, you might be in the pager.

To explicitly enable paging for long outputs, use: `flujo run --pager`.

Disable paging and/or color when needed:
- One-off: `CLICK_DISABLE_PAGER=1 flujo --help`
- Permanent (shell): `export CLICK_DISABLE_PAGER=1`
- No colors (also avoids pager triggers in some setups): `NO_COLOR=1 flujo validate`

The Flujo CLI aims to avoid paging for common commands; however, the above knobs ensure predictable behavior across shells and CI systems.

## Run

- `--dry-run`: Parse and validate only; do not execute the pipeline.
  - With `--json`, prints `{ "validated": true, "steps": ["..."] }`

Example:
- `flujo --project . run --dry-run --json`
- `FLUJO_PROJECT_ROOT=$PWD flujo run --input 'hello'`

### Debugging flags

- `--debug`: Enables step-by-step tracing and prints a compact Debug Trace tree at the end of the run.
  - Shows spans with attributes like `flujo.step.type`, `flujo.step.policy`, latency, and cache hits.
  - Includes notable events such as prompt injection previews.
- `--trace-preview-len N`: Sets max characters for prompt/response previews in the Debug Trace (default 1000).
- `--debug-prompts`: Also emit full, unredacted prompts and responses in trace events.
  - Unsafe for production; intended for local debugging only. Implies `--debug`.
- `--debug-export`: Enable full JSON debug log export. If `--debug-export-path` is omitted, Flujo auto-writes to `./debug/<timestamp>_<run_id>.json`.
- `--debug-export-path PATH`: Write the full JSON debug log to PATH.
  - `trace_tree`: complete spans, attributes, and events
  - `result.step_history`: full recursive step history with outputs/feedback/metrics
  - `final_context`: serialized context including `import_artifacts`, `step_outputs`, `conversation_history`, `hitl_history`, `command_log`
  - `env`: debug-related flags active for the run

Examples:
- Safe previews:
  - `flujo run --debug --trace-preview-len 1500 --input 'hello'`
- Full unredacted content (local only):
  - `flujo run --debug-prompts --trace-preview-len 4000 --input 'hello'`
- Export everything to a file:
  - `flujo run --debug --debug-export-path output/last_run_debug.json --input 'hello'`
  - or rely on auto-export to `./debug/...` by running: `flujo run --debug --debug-export --input 'hello'`

### Visualization and Output Controls

- `--live` / `--progress`: Lightweight live progress view (ConsoleTracer). Falls back gracefully if unavailable.
- `--summary`: Print a compact run summary (final output, totals, run ID).
- `--show-steps` / `--no-steps`: Toggle the Step Results table (default: show).
- `--show-context` / `--no-context`: Toggle final context printing (default: show).
- `--show-output-column` / `--no-output-column`: Show/hide the Output column in the Step Results table.
- `--output-preview-len N`: Trim long outputs in the table to N characters (default: 100).
- `--final-output-format {auto|raw|json|yaml|md}`: Control final output rendering (default: auto).
- `--pager`: Page the final rendering (useful for long outputs).
- `--only-steps name[,name...]`: Filter Step Results to selected step names.

Examples:
- `flujo run --live --summary --output-preview-len 200 --input 'hello'`
- `flujo run --final-output-format json --only-steps plan,execute --no-context`

## Lens

- `flujo lens list`: List stored runs.
- `flujo lens show <run_id>`: Show step-by-step details of a prior run.
- `flujo lens trace <run_id>`: Render the hierarchical execution trace for a prior run.
  - Options: `--prompt-preview-len N` to control preview length for prompt-related events.
- `flujo lens from-file <path>`: Render a saved debug JSON (created by `--debug-export`) as a rich trace tree.
  - Options: `--prompt-preview-len N` to control preview length.
  - Useful for offline analysis or sharing a single file that contains trace + step history + final context.
- `flujo lens evals`: List persisted shadow evaluation scores (when enabled and using a DB-capable state backend).
  - Options: `--run-id <run_id>` to filter, `--limit/-n` to cap results.

## Dev Commands

- `flujo dev import-openapi <spec>`: Generate Pydantic models from an OpenAPI/Swagger spec and (optionally) agent wrappers.
  - `spec` can be a local file path; wrapper generation currently expects a local file (URL support is best-effort).
  - Requires `datamodel-code-generator` to be installed.

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
