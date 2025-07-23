# State Backends API

Flujo provides persistent state management for workflows, allowing them to be paused, resumed, and monitored across process restarts.

## Overview

The state management system has been optimized for production use cases with the following key features:

- **Durable Workflows**: Workflows can be paused and resumed across process restarts
- **Optimized SQLite Backend**: High-performance, indexed database for large-scale deployments
- **Admin Queries**: Built-in observability and monitoring capabilities
- **Automatic Migration**: Seamless upgrades for existing deployments

For detailed information about the optimized SQLite backend, see [State Backend Optimization](state_backend_optimization.md) and the comprehensive [SQLite Backend Guide](guides/sqlite_backend_guide.md).

## API Reference

::: flujo.state.models.WorkflowState

::: flujo.state.backends.base.StateBackend

::: flujo.state.backends.memory.InMemoryBackend

::: flujo.state.backends.file.FileBackend

::: flujo.state.backends.sqlite.SQLiteBackend

## Operational State Management (CLI)

Flujo provides robust CLI tools for managing workflow state, making it easy to resolve issues such as state/code mismatches or to clean up old runs:

- **Delete a specific workflow state:**
  ```bash
  flujo lens delete <RUN_ID>
  # Example:
  flujo lens delete cohort-clarification-run-123
  ```
  This is the recommended way to remove a problematic or stale workflow state.

- **Prune old/completed workflow states:**
  ```bash
  flujo lens prune --days-old 30
  # Optionally filter by status:
  flujo lens prune --days-old 90 --status completed --yes
  ```
  This command deletes all workflow states older than the specified number of days.

- **When you see a StateIncompatibilityError:**
  If you encounter an error like:
  ```
  StateIncompatibilityError: Cannot resume workflow due to a mismatch between the saved state and the current pipeline code.
  ...
  1. For Development & Testing:
     - Use the CLI: `flujo lens delete <RUN_ID>` to remove this workflow state.
     - Use the CLI: `flujo lens prune --days-old <N>` to clean up old or completed states.
     - Only as a last resort, delete the state database file (e.g., 'flujo_ops.db').
     - Or use a new, unique run_id for this execution.
  ...
  ```
  Use the CLI tools above for the safest and most robust solution.
