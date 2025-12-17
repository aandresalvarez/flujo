# Shadow Evaluations (LLM-as-Judge)

Flujo contains a “shadow evaluator” that can asynchronously score step outputs using an LLM-as-a-judge, without blocking the main user flow.

## What Exists in Code

- A scheduler and judge runner (`ShadowEvaluator`).
- Optional persistence to the state backend (`evaluations` table in SQLite/Postgres backends).
- A CLI viewer: `flujo lens evals`.

## Current Status

Shadow evaluation is implemented but defaults to disabled. Enable it via environment variables:

```bash
export FLUJO_SHADOW_EVAL_ENABLED=1
export FLUJO_SHADOW_EVAL_SAMPLE_RATE=0.1
export FLUJO_SHADOW_EVAL_TIMEOUT_S=30
export FLUJO_SHADOW_EVAL_JUDGE_MODEL=openai:gpt-4o-mini
export FLUJO_SHADOW_EVAL_SINK=telemetry   # or database
export FLUJO_SHADOW_EVAL_EVALUATE_ON_FAILURE=1  # optional
export FLUJO_SHADOW_EVAL_RUN_LEVEL=1  # optional
```

Notes:
- Sampling is cached per `run_id` (run-level sampling decision).
- If `FLUJO_SHADOW_EVAL_EVALUATE_ON_FAILURE=1`, only failed steps are evaluated.
- If `FLUJO_SHADOW_EVAL_RUN_LEVEL=1`, Flujo also schedules a single run-level evaluation saved under the pseudo step name `__run__`.

## Viewing Results

When enabled and using a DB-capable state backend:

```bash
flujo lens evals --limit 20
flujo lens evals --run-id <run_id>
```
