---
title: OpenAPI Import (Dev)
---

# OpenAPI Import (Dev Command)

Flujo can generate Pydantic models (and optional agent wrappers) from an OpenAPI/Swagger spec.

## Command

```bash
flujo dev import-openapi <spec> --output generated_tools
```

Where:
- `<spec>` is a local path (JSON or YAML) or a URL (best-effort).
- `--output` is the output directory for generated files.

## What It Generates

- Pydantic models via `datamodel-code-generator` (optional dependency).
- Optional agent wrappers that use `httpx` + `make_agent_async` to expose OpenAPI operations as tools.
- A package `__init__.py` that re-exports the common entrypoints for “one-liner” imports.

## Options (high level)

- `--generate-agents/--skip-agents`: Toggle wrapper generation.
- `--agents-filename`: Name of the generated wrapper file (relative to the output dir).
- `--python-version`: Target Python version for model generation.

## Notes / Limitations

- `--output` must be a valid importable Python package name (e.g., `generated_tools`, not `generated-tools`).
- Wrapper generation supports local specs and URLs (fetched over HTTP).
- Response typing in generated wrappers is best-effort and may fall back to `dict`/`list` (some array responses map to `list[Model]` when `$ref` items are detected).
- If `datamodel-code-generator` is missing, the command exits with a friendly install hint.

## Importing the Generated Package

If you generate into `generated_tools/`, you can import directly:

```python
from generated_tools import make_openapi_agent, make_openapi_operation_agent

# Models are also re-exported from generated_models.py
from generated_tools import Pet
```
