# Caching Steps in YAML

This page shows how to cache step results in YAML using `kind: cache`, and when caching is appropriate.

## Overview

- `kind: cache` wraps an inner step and caches its output keyed by input.
- Use caching for expensive, deterministic operations where identical inputs recur (LLMs, remote APIs).

## Basic Syntax

```yaml
- kind: cache
  name: cached_stringify
  wrapped_step:
    kind: step
    name: stringify
    agent: { id: "flujo.builtins.stringify" }
```

## Example: Caching an LLM Call

```yaml
- kind: cache
  name: cached_summarize
  wrapped_step:
    kind: step
    name: summarize
    uses: agents.summarizer
    input: "{{ context.article_text }}"
```

## Best Practices

- Ensure the wrapped step is deterministic with respect to its input.
- Avoid caching steps that rely on external state unless such state is encoded into the input.
- Start with the default cache backend; add backend configuration later if needed.

## Troubleshooting

- If you don't observe hits on repeated runs with identical inputs, verify the wrapped step input really matches (no hidden fields or timestamps).
- For debugging, inspect `step_history[-1].metadata_["cache_hit"]` — it should be `True` for cache hits.
