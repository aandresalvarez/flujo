## Type Safety Observability & CI Gates

- Strict-only: `strict_dsl` and `context_strict_mode` are always enabled; there is no opt-out.
- Lints/validation that must stay green:
  - `V-A2-STRICT` for generic Any/object flows without explicit adapters.
  - `V-ADAPT-ALLOW` for adapters missing allowlist tokens.
  - `CTX-SCRATCHPAD` / `CTX-OUTPUT-KEYS` for context misuse.
  - Adapter allowlist lint + type-safety lint (`scripts/lint_type_safety.py`) to guard new `Any`/`cast` in core/DSL.
- Metrics to watch (CI dashboards/baselines):
  - Counts of `Any`/`cast` in core/DSL (from `lint_type_safety.py`).
  - Adapter invocations and allowlist coverage.
  - Scratchpad references in templates/sink targets.
- Commands:
  - `make lint` — runs adapter allowlist and type-safety lints.
  - `make test-fast` — quick regression check (strict validators included).
  - `make typecheck` — must pass before release.
  - `python scripts/codemods/scratchpad_to_typed.py --apply <paths>` — migrates scratchpad usage.
- Baselines must not regress; CI fails on any increase of `Any`/`cast` or new unapproved adapters/scratchpad writes.

