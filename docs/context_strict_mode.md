## Context Strict Mode (Phase 1)

Strict context mode is **always on**. There is no loose/scratchpad-compatible mode and no opt-out flags in CI or local runs.

- Scratchpad is reserved for framework metadata only. User data must be stored in typed context fields.
- Steps with `updates_context=True` must declare `output_keys` or `sink_to` (non-scratchpad) for persistence.
- Pipeline validation hard-fails:
  - `CTX-SCRATCHPAD` when writing to/reading from scratchpad in templated input or sink.
  - `CTX-OUTPUT-KEYS` when `updates_context` is set but no output target is declared.
- Use typed contexts:

  ```python
  from flujo.domain.context_mixins import BaseContext, typed_context

  class MyContext(BaseContext):
      counter: int = 0
      result: str | None = None

  Ctx = typed_context(MyContext)
  ```
- Codemod helper: `python scripts/codemods/scratchpad_to_typed.py --apply <paths>` rewrites `ctx.scratchpad["foo"]` to `ctx.foo` (conservative).
- Adapter allowlist: see `docs/adapter_allowlist.md` for marking adapters with `adapter_id`/`adapter_allow` and the required token.
- Observability: CI fails on new scratchpad writes or missing `output_keys`; `make lint` runs the context strict lints. Count-based metrics (scratchpad references, adapter usage, Any/cast in core) are tracked and must not regress relative to baseline.

### CI Enforcement

Strict mode is enforced automatically via:

1. **`make lint`**: Runs `scripts/lint_type_safety.py` and `scripts/lint_adapter_allowlist.py` which guard against baseline regressions.
2. **Architecture tests**: `tests/architecture/test_type_safety_compliance.py` verifies:
   - `Any` type usage stays within baseline
   - `cast()` usage stays within baseline  
   - Scratchpad allowlist is not expanded unexpectedly
3. **Pipeline validation**: Hard-fails on `CTX-SCRATCHPAD` and `CTX-OUTPUT-KEYS` violations at runtime.

No opt-out flags exist. Pull requests that increase type-safety debt must update baselines intentionally.
