# Centralized Cost & Budget Governance

This guide explains how administrators can define and enforce organization-wide budgets for all pipeline runs via `flujo.toml`.

## Overview

- Centralized: Budgets are configured once in `flujo.toml` and apply to all runs.
- Deterministic: Resolution follows exact name, then wildcard, then default.
- Restrictive by design: The final budget is the most restrictive combination of CLI/code-provided and TOML budgets.

## flujo.toml Schema

```toml
[budgets.default]
total_cost_usd_limit = 10.0
total_tokens_limit = 100000

[budgets.pipeline]
"analytics" = { total_tokens_limit = 200000 }
"team-*"   = { total_cost_usd_limit = 5.0 }
```

- `budgets.default`: Fallback when no pipeline-specific entry matches.
- `budgets.pipeline`: A map from pipeline names or glob patterns to `UsageLimits`.
  - Supports `*`, `?`, and character classes like `[a-z]`.

## Resolution Rules

Given a `pipeline_name`:
1. Exact match in `[budgets.pipeline]` wins.
2. First wildcard match in insertion order wins (e.g., `team-*`).
3. Fallback to `[budgets.default]`.

If no budget is found, the run is unlimited unless the developer provided limits in code/CLI.

## Precedence with Code/CLI

If a developer provides `UsageLimits` programmatically, the framework combines them with the TOML limits using the most restrictive rule for each field:

- `final.total_cost_usd_limit = min(code.total_cost_usd_limit, toml.total_cost_usd_limit)` (treating `None` as unlimited)
- `final.total_tokens_limit = min(code.total_tokens_limit, toml.total_tokens_limit)` (treating `None` as unlimited)

This allows developers to set tighter budgets without ever exceeding centralized policies.

## Inspecting Effective Budgets via CLI

Show how a budget resolves for a pipeline:

```bash
flujo budgets show my-pipeline
```

Example output:

```
Effective budget for 'my-pipeline':
  - total_cost_usd_limit: $5.00
  - total_tokens_limit: unlimited
Resolved from budgets.pipeline[team-*] in flujo.toml
```

## Operational Tips

- Track changes to `flujo.toml` in Git for auditability.
- Prefer specific pipeline names over broad wildcard patterns when possible.
- Start with conservative defaults, then carve-out larger budgets as needed.


