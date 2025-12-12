Expression Language (DSL) — Safe Conditions
===========================================

Flujo supports a small, safe expression language for declarative branching and loop exits in YAML via `condition_expression` and `exit_expression`.

Allowed Values and Names
- Literals: strings, numbers, booleans, null
- Names: `previous_step` (alias: `output`), `context`, `steps`, `resume_input`
- Attribute access: `obj.attr`
- Subscript with string keys: `obj['key']`
- Bool ops: `and`, `or`; unary `not`
- Comparisons: `==`, `!=`, `<`, `<=`, `>`, `>=`, `in`, `not in`

Allow‑listed Method Calls (Read‑Only)
- On `dict`: `get(key[, default])` — string keys only
- On `str`:
  - `lower()`, `upper()`, `strip()`
  - `startswith(prefix)`, `endswith(suffix)`

Everything else is disallowed and raises `Unsupported expression element: Call`.

Variables in Scope
- `previous_step`/`output`: the immediate prior step's output (for the first step, this is the pipeline input)
- `context`: a template-safe proxy of the pipeline context
- `steps`: map of previous step outputs, keyed by step name
- `resume_input`: the most recent HITL (human-in-the-loop) response (available only after a HITL step)

Examples
- Conditional by string prefix: `previous_step.lower().startswith('ok:')`
- Loop exit by status: `context.status.upper() == 'DONE'`
- Check membership: `'tool' in steps and steps['tool'].success == True`

Security Notes
- No arbitrary function calls, imports, attribute mutation, or I/O
- Only the allow‑listed string/dict methods above are callable

Error Messages
- Unsupported calls produce a clear `Unsupported expression element: Call` error
- Invalid names: `Unknown name: <name>`

Performance
- Simple expressions evaluate in microseconds on modern hardware
