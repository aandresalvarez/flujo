### **Functional Specification Document: FSD-08**

**Title:** Core Operational Persistence
**Project Lead:** AI Assistant
**Status:** Proposed
**Priority:** P1 - High
**Date:** 2025-07-14
**Version:** 1.0

---

### **1. Overview**

This document specifies the foundational layer for observability and resilience in the `flujo` framework. The primary goal is to provide all users with a robust, zero-configuration, local-first persistence layer for every pipeline run. This feature will automatically capture the full execution history, including inputs, outputs, and metadata for each step, into a structured, queryable local database.

This will be achieved by making the `SQLiteBackend` the default state-persistence mechanism and enhancing its schema to store granular run and step data. A new CLI command suite, `flujo lens`, will be introduced to provide developers with immediate, powerful tools to inspect, debug, and manage their pipeline runs directly from the terminal.

This FSD is the first of a three-part "FlujoLens" initiative and serves as the bedrock for all future observability features, including advanced tracing and production monitoring.

### **2. Problem Statement**

Currently, `flujo` pipeline persistence is an opt-in feature that requires manual configuration. When enabled, it saves only the latest state, overwriting previous step data. This presents several challenges:

1.  **Lack of Default History:** Developers have no out-of-the-box way to review the full execution history of a pipeline run, making debugging complex or failed runs difficult.
2.  **Brittle State Storage:** The current single-blob storage for context and output is inefficient for querying and does not scale well.
3.  **No Introspection Tools:** There are no built-in tools to easily list, filter, or inspect past or ongoing pipeline runs, which is a critical operational requirement.

This FSD addresses these problems by making structured, queryable persistence a default, core feature of the framework.

### **3. Functional Requirements (FR)**

| ID | Requirement | Justification |
| :--- | :--- | :--- |
| FR-21 | The `Flujo` runner **SHALL** default to using an instance of `SQLiteBackend` writing to `flujo_ops.db` in the current working directory if no `state_backend` is provided. | Ensures all runs are persisted by default, providing immediate value and improving the debugging experience. |
| FR-22 | The `SQLiteBackend` **SHALL** use a structured schema with separate tables for `runs` and `steps`. | Enables efficient, indexed querying of run history and step-level details, and lays the groundwork for advanced observability. |
| FR-23 | The `StateManager` component **SHALL** be responsible for persisting the final `PipelineResult`, including the full `step_history`, to the `SQLiteBackend` upon completion of a run. | Enforces separation of concerns: the runner manages execution, the manager handles state logic, and the backend handles storage. |
| FR-24 | A new CLI command group, `flujo lens`, **SHALL** be introduced. | Provides a dedicated, user-friendly namespace for all observability and operational commands. |
| FR-25 | The `flujo lens list` command **SHALL** display a tabular view of all persisted pipeline runs, with options to filter by status and pipeline name. | Offers a simple, powerful way for developers to get an overview of their pipeline activity. |
| FR-26 | The `flujo lens show <run_id>` command **SHALL** display a detailed summary of a specific run, including its final context and a list of all its steps with their inputs, outputs, and status. | Provides a crucial tool for debugging specific pipeline failures by showing the complete execution flow. |
| FR-27 | A new configuration utility **SHALL** be implemented to load the `StateBackend` URI from a `flujo.toml` file or a `FLUJO_STATE_URI` environment variable for use by the `flujo lens` CLI. | Creates a standard, ergonomic way for the standalone CLI tool to connect to the correct operational database. |

### **4. Non-Functional Requirements (NFR)**

| ID | Requirement | Justification |
| :--- | :--- | :--- |
| NFR-9 | The default persistence mechanism **MUST NOT** introduce more than a 5% performance overhead to a typical pipeline run compared to running with no backend. | Ensures the "always-on" persistence feature does not negatively impact development velocity. |
| NFR-10 | CLI commands (`list`, `show`) on a database with 10,000 runs **MUST** complete in under 500ms. | Guarantees that the operational tools remain snappy and useful even with a large history. |

### **5. Technical Design & Specification**

#### **5.1. `Flujo` Runner Modification**

The `Flujo` runner's constructor will be modified to instantiate `SQLiteBackend` by default.

```python
# In Flujo.__init__
if state_backend is None:
    # Use a default SQLite backend in the current directory.
    # The backend will handle creating the directory and file.
    from flujo.state.backends.sqlite import SQLiteBackend
    from pathlib import Path
    self.state_backend = SQLiteBackend(Path.cwd() / "flujo_ops.db")
```

#### **5.2. `SQLiteBackend` Schema and Implementation**

The `SQLiteBackend` will be updated to use a new, structured schema.

```sql
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    pipeline_name TEXT NOT NULL,
    pipeline_version TEXT NOT NULL,
    status TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT,
    total_cost REAL,
    final_context_blob TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_pipeline_name ON runs(pipeline_name);

CREATE TABLE IF NOT EXISTS steps (
    step_run_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    step_name TEXT NOT NULL,
    step_index INTEGER NOT NULL,
    status TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT,
    duration_ms INTEGER,
    cost REAL,
    tokens INTEGER,
    input_blob TEXT,
    output_blob TEXT,
    error_blob TEXT,
    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_steps_run_id ON steps(run_id);
```

The `StateBackend` interface will be extended to support structured data. The `StateManager` will use these new methods.

```python
class StateBackend(ABC):
    async def save_run_start(self, run_data: dict) -> None: ...
    async def save_step_result(self, step_data: dict) -> None: ...
    async def save_run_end(self, run_id: str, end_data: dict) -> None: ...
    async def get_run_details(self, run_id: str) -> Optional[dict]: ...
    async def list_run_steps(self, run_id: str) -> List[dict]: ...
```

#### **5.3. CLI Implementation (`flujo lens`)**

A new module will house the CLI logic with `list` and `show` commands.

### **6. Testing Plan**

A comprehensive suite of unit and integration tests will validate the new persistence layer and CLI tooling.

### **7. Implementation Plan**

1. **Phase 1: Backend & Schema** – Implement new tables and backend methods.
2. **Phase 2: Core Integration** – Update runner and state manager to use the new backend.
3. **Phase 3: CLI Tooling** – Add `flujo lens` commands and configuration helper.
4. **Phase 4: Documentation & Finalization** – Update documentation and ensure all tests pass.

### **8. Risks and Mitigation**

| Risk | Impact | Mitigation |
| :--- | :--- | :--- |
| **Performance Overhead** | Medium | Benchmark and optimize writes; allow disabling persistence. |
| **Schema Migration Bugs** | High | Thorough unit tests and fallback logic. |
| **CLI Configuration Complexity** | Low | Provide sensible defaults and clear docs. |

### **9. Future Considerations**

This FSD lays the groundwork for future tracing and monitoring features in the "FlujoLens" initiative.
