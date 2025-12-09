## Adapter Allowlist (Strict DSL)

- All adapter steps (`meta.is_adapter=True`) must declare:
  - `adapter_id`: identifier present in `scripts/adapter_allowlist.json`
  - `adapter_allow`: token that matches the allowlist entry
- Validation errors:
  - `V-ADAPT-ALLOW`: adapter used without an allowlisted `adapter_id` or with mismatched token.
- Defaults:
  - Generic adapters created via `Step.from_callable(..., is_adapter=True)` set `adapter_id="generic-adapter"` and `adapter_allow="generic"`, with an allowlist entry preseeded.
- How to mark a custom adapter:
  ```python
  s = Step.from_callable(fn, name="my_adapter", is_adapter=True)
  s.meta["adapter_id"] = "my-boundary"
  s.meta["adapter_allow"] = "owner-token"
  # Add {"my-boundary": "owner-token"} to scripts/adapter_allowlist.json
  ```
- Lint/validation:
  - Enforced during `Pipeline.validate_graph` (rule `V-ADAPT-ALLOW`).
  - `make lint` runs `lint_adapter_allowlist.py` which also enforces allowlist use.

