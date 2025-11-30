# Background Task Resumability

This document summarizes how background steps are persisted, monitored, and resumed.

## Configuration

`flujo.toml`

```toml
[background_tasks]
enable_state_tracking = true
enable_resumability = true
enable_quota = true
max_cost_per_task = 1.0
max_tokens_per_task = 10000
stale_task_timeout_hours = 24
```

All settings flow through `get_settings().background_tasks`; do not read env vars directly.

## Behavior

- Each background step runs with an isolated context and unique `run_id` linked via `parent_run_id`.
- State persists via `StateManager` with metadata: `task_id`, `step_name`, `is_background_task`, `background_error`.
- Quota: split from parent when available; otherwise uses per-task limits. Background runs gate on available quota.
- Control flow:
  - `PausedException` → state marked `paused`, category `control_flow`.
  - `PipelineAbortSignal` → state marked `failed`, category `control_flow`.
  - Other errors → classified via `ErrorClassifier`, persisted category and message.
- Telemetry/hooks:
  - Hook payloads carry `is_background`; OTEL/console tracers skip background events to reduce noise.
  - Tracing spans include `flujo.is_background=true` when applicable.
- Metrics:
  - Prometheus collector exports `flujo_background_tasks_by_status`.
  - Workflow stats include background status counts for dashboards.

## Runner APIs

- `get_failed_background_tasks(parent_run_id=None, hours_back=24)` → list failed tasks.
- `resume_background_task(task_id, new_data=None)` → runs failed task in foreground, updates state.
- `retry_failed_background_tasks(parent_run_id, max_retries=3)` → best-effort retries.
- `cleanup_stale_background_tasks(stale_hours=24)` → marks long-running background tasks as failed (timeout).
- `get_failed_background_tasks(parent_run_id=None, hours_back=24)` respects time windows and parent filtering.

## State Backend Support

- SQLite schema includes `metadata`, `is_background_task`, `parent_run_id`, `task_id`, `background_error` columns.
- Migrations validate new columns against a whitelist and safe column definitions.
- `list_background_tasks`, `get_failed_background_tasks`, `cleanup_stale_background_tasks` provide filtered views/maintenance.

## Testing

Integration tests: `tests/integration/test_background_resumability.py` cover resumability, hook flagging, stale cleanup, quota gating, pause persistence, and Prometheus background metrics. Benchmark: `tests/benchmarks/test_background_task_performance.py` guards overhead.

Benchmark marking: `@pytest.mark.benchmark` and `@pytest.mark.slow` to avoid default fast suites.
