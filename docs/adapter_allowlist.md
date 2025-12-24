## Adapter Allowlist (Strict DSL)

Strict DSL mode is the only mode. Any generic flow (Any/object) must pass through an explicit, allowlisted adapter. There is no legacy loose mode.

- All adapter steps (`meta.is_adapter=True`) must declare:
  - `adapter_id`: identifier present in `scripts/adapter_allowlist.json`
  - `adapter_allow`: token that matches the allowlist entry
- Validation errors:
  - `V-ADAPT-ALLOW`: adapter used without an allowlisted `adapter_id` or with mismatched token.
- Construction (no implicit defaults):
  - `Step.from_callable(..., is_adapter=True, adapter_id=..., adapter_allow=...)` **must** be provided; missing values raise at construction time.
- How to mark a custom adapter:
  ```python
  s = Step.from_callable(
      fn,
      name="my_adapter",
      is_adapter=True,
      adapter_id="my-boundary",
      adapter_allow="owner-token",
  )
  # Add {"my-boundary": "owner-token"} to scripts/adapter_allowlist.json
  ```
- Lint/validation:
  - Enforced during `Pipeline.validate_graph` (rule `V-ADAPT-ALLOW` + `V-A2-STRICT` for generic inputs without adapters).
  - `make lint` runs `lint_adapter_allowlist.py` which also enforces allowlist use.
- Observability:
  - Metrics track adapter invocations and allowlist usage; CI fails on new unapproved adapters or missing tokens.

## Agent Adapters (Code)

Agent adapters under `flujo/agents/adapters/` must register in
`tests/adapters/test_adapter_contracts.py` (`ADAPTER_CASES`) so shared usage and
cost semantics are enforced.
