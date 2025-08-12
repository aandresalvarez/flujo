# State Backends API

Flujo provides persistent state management for workflows, allowing them to be paused, resumed, and monitored across process restarts.

## Overview

The state management system has been optimized for production use cases with the following key features:

- **Durable Workflows**: Workflows can be paused and resumed across process restarts
- **Optimized SQLite Backend**: High-performance, indexed database for large-scale deployments
- **Admin Queries**: Built-in observability and monitoring capabilities
- **Automatic Migration**: Seamless upgrades for existing deployments

For detailed information about the optimized SQLite backend, see [State Backend Optimization](state_backend_optimization.md) and the comprehensive [SQLite Backend Guide](../guides/sqlite_backend_guide.md).

## API Reference

::: flujo.state.models.WorkflowState

::: flujo.state.backends.base.StateBackend

::: flujo.state.backends.memory.InMemoryBackend

::: flujo.state.backends.file.FileBackend

::: flujo.state.backends.sqlite.SQLiteBackend
