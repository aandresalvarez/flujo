Skills Catalog Example

This example demonstrates how Flujo auto-loads a local skills catalog when running a YAML pipeline.

Files:
- `skills.yaml` — Declares a custom `echo_cmd` skill that returns an async echo agent.
- `myskills.py` — Implements `make_echo()` factory that returns the async echo callable.
- `pipeline.yaml` — References `echo_cmd` and passes a static input string.

Run:
- From the repository root: `uv run flujo run examples/skills_catalog/pipeline.yaml --json`

Expected:
- The run resolves `echo_cmd` from `skills.yaml` and outputs `"hello world"`.

