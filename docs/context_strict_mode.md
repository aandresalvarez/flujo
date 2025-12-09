## Context Strict Mode (Phase 1)

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

