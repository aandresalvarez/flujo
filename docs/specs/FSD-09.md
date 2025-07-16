# Functional Specification Document: FSD-09

**Title:** Rich Internal Tracing and Visualization
**Project Lead:** AI Assistant
**Status:** Proposed
**Priority:** P1 - High
**Date:** 2025-07-14
**Version:** 1.0

---

## 1. Overview

This document specifies the second phase of the FlujoLens initiative, focusing on providing developers with a rich, local-first, zero-configuration tracing experience. Building upon the Core Operational Persistence layer from FSD-08, this feature will capture a detailed, hierarchical trace for every pipeline run, persist it to the operational database, and make it available for immediate inspection via a new, intuitive CLI command.

The core components of this FSD are:

1. **A default, internal `TraceManager` hook:** This component will be added to the `Flujo` runner by default and will be responsible for building a structured, in-memory representation of the execution trace as the pipeline runs.

2. **Persistence of Trace Data:** The `SQLiteBackend` will be enhanced with a new `traces` table to store span data, making traces durable and queryable.

3. **A Powerful CLI Visualization Tool:** The `flujo lens` command suite will be extended with a `trace` command that renders a rich, tree-based view of a pipeline's execution directly in the terminal, complete with timings, status, and metadata.

This feature will dramatically improve the debugging and performance analysis capabilities of `flujo`, allowing developers to understand the "why" behind their pipeline's behavior without needing to set up any external observability tools.

## 2. Problem Statement

While FSD-08 provides a history of *what* steps ran and their final outcomes, it lacks the hierarchical and contextual information needed to easily debug complex workflows. Developers currently cannot see:

1. **Parent-Child Relationships:** How nested steps (like those in a `LoopStep` or `ConditionalStep`) relate to their parent. A flat list of steps makes it difficult to understand the control flow.

2. **Precise Timings and Overlaps:** In parallel executions, it's impossible to see which branches ran concurrently or how their execution times overlapped.

3. **Granular Metadata:** Key decisions, such as which branch was taken in a `ConditionalStep` or why a `LoopStep` exited, are not explicitly captured in a structured way.

Without a built-in tracing mechanism, debugging remains a manual process of adding print statements or logs, which is inefficient and scales poorly.

## 3. Functional Requirements (FR)

| ID | Requirement | Justification |
| :--- | :--- | :--- |
| FR-28 | The `Flujo` runner **SHALL**, by default, include an internal `TraceManager` hook that is active on every run. | Ensures that detailed trace data is captured for all pipelines automatically, without requiring user configuration. |
| FR-29 | The `TraceManager` **SHALL** build a hierarchical tree of "span-like" objects in memory, mirroring the execution flow, including nested steps within loops and branches. | Captures the essential parent-child and timing relationships needed for effective debugging and performance analysis. |
| FR-30 | Each span object in the trace **SHALL** capture the step name, start/end times, final status, and relevant metadata (e.g., `executed_branch_key`, `iteration_number`). | Provides the granular detail needed to understand the execution of each step within its context. |
| FR-31 | The `SQLiteBackend` **SHALL** be enhanced with a `traces` table designed to store the structured span data captured by the `TraceManager`. | Makes the generated traces durable, queryable, and available for post-run analysis via the CLI. |
| FR-32 | The `StateManager` **SHALL** be responsible for persisting the full trace tree to the `traces` table upon completion of a run. | Maintains the clear separation of concerns where the `StateManager` handles all interactions with the `StateBackend`. |
| FR-33 | The `flujo lens` CLI **SHALL** be extended with a new `trace <run_id>` command. | Provides the primary user interface for accessing and visualizing the captured trace data. |
| FR-34 | The `flujo lens trace` command **SHALL** fetch the trace data for the specified `run_id` and render it as a `rich.Tree` in the console. | Offers an intuitive, powerful, and zero-setup visualization tool for developers to debug their pipelines. |

## 4. Non-Functional Requirements (NFR)

| ID | Requirement | Justification |
| :--- | :--- | :--- |
| NFR-11 | The `TraceManager` hook **MUST NOT** add more than a 5% performance overhead to a typical pipeline run. | Ensures that the default tracing feature is lightweight and does not hinder development velocity. |
| NFR-12 | The `flujo lens trace` command **MUST** render a trace for a pipeline with up to 100 spans in under 1 second. | Guarantees that the visualization tool is responsive and useful for complex but common pipeline sizes. |

## 5. Technical Design & Specification

### 5.1. `TraceManager` Hook Implementation

**File:** `flujo/tracing/manager.py` (New)

- A new `TraceManager` class will be implemented. It will not depend on OpenTelemetry.
- It will contain a `hook` method that implements the `HookCallable` protocol.
- **Internal State:** It will use a simple list (`self._span_stack: List[Span]`) to manage the current trace context.
- **Span Data Structure:** A simple `Span` dataclass will be defined within the module to hold span data (name, start_time, end_time, attributes, children).
- **Event Handling Logic:**
  - `pre_run`: Creates the root span and pushes it onto the stack.
  - `pre_step`: Creates a new child span, appends it to the current span on the stack, and pushes the new span onto the stack.
  - `post_step` / `on_step_failure`: Pops the current span from the stack, records its end time and status, and attaches metadata from the `StepResult`.
  - `post_run`: Attaches the completed root span (the full trace tree) to `payload.pipeline_result.trace_tree`.

### 5.2. `Flujo` Runner Modification

**File:** `flujo/application/runner.py`

- The `Flujo.__init__` method will be modified to instantiate and add the `TraceManager` hook to its `self.hooks` list by default.

### 5.3. `SQLiteBackend` Schema and Implementation

**File:** `flujo/state/backends/sqlite.py`

- **Schema Change:** A new table will be added in `_init_db`:
  ```sql
  CREATE TABLE IF NOT EXISTS traces (
      span_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL,
      parent_span_id TEXT,
      name TEXT NOT NULL,
      start_time TEXT NOT NULL,
      end_time TEXT NOT NULL,
      attributes_json TEXT,
      FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
  );
  CREATE INDEX IF NOT EXISTS idx_traces_run_id ON traces(run_id);
  ```
- **New Backend Methods:** The `StateBackend` interface and `SQLiteBackend` implementation will be extended with:
  ```python
  async def save_trace(self, trace_data: List[Dict[str, Any]]) -> None: ...
  async def get_trace(self, run_id: str) -> List[Dict[str, Any]]: ...
  ```
  The `save_trace` method will use `executemany` for efficient batch insertion of all spans in the trace tree.

### 5.4. `StateManager` Modification

**File:** `flujo/application/core/state_manager.py`

- The `StateManager.record_run_end` method will be updated to extract the `trace_tree` from the `PipelineResult` and call the new `backend.save_trace()` method.

### 5.5. CLI Implementation

**File:** `flujo/cli/lens.py`

- A new `trace` command will be added to the `lens_app`.
- It will call `backend.get_trace(run_id)` to fetch the flat list of spans.
- A helper function, `_reconstruct_and_render_tree(spans)`, will be implemented to:
  1. Rebuild the hierarchical `rich.Tree` from the flat list of spans using their `span_id` and `parent_span_id`.
  2. Format the display of each node to be readable, including status (✅/❌), duration, and key attributes.
  3. Print the final tree to the console.

## 6. Testing Plan

### 6.1. Unit Tests

- **`TraceManager`:**
  - Test that it correctly builds a nested trace tree for a sequence of simulated hook events.
  - Test that spans are correctly populated with metadata from `StepResult`.
  - Test that the stack is correctly managed, even with unclosed spans (e.g., due to an exception).
- **`SQLiteBackend`:**
  - Test `save_trace` and `get_trace` methods.
  - Test that deleting a run via `ON DELETE CASCADE` also deletes its associated traces.
- **`flujo lens trace` CLI:**
  - Test rendering of a simple linear trace.
  - Test rendering of a complex nested trace (loop inside a conditional).
  - Test graceful error handling when a `run_id` has no associated trace data.

### 6.2. Integration Tests

- **Test 1: Linear Pipeline Trace:**
  - Run a simple `step1 >> step2 >> step3` pipeline.
  - Execute `flujo lens trace <run_id>`.
  - Assert that the output is a non-nested tree showing the three steps in order.
- **Test 2: Nested Loop Trace:**
  - Run a pipeline containing a `LoopStep` that executes 3 times.
  - Execute `flujo lens trace <run_id>`.
  - Assert that the output tree shows the `LoopStep` as a parent node with three child nodes, one for each iteration.
- **Test 3: Conditional Branch Trace:**
  - Run a pipeline with a `ConditionalStep`.
  - Execute `flujo lens trace <run_id>`.
  - Assert that the output tree shows the `ConditionalStep` as a parent and that *only the executed branch* appears as a child node. The `executed_branch_key` should be an attribute on the parent span.

## 7. Implementation Plan

1. **Phase 1: `TraceManager` and Core Integration (2 days)**
   - [ ] Implement the `Span` dataclass and the `TraceManager` hook.
   - [ ] Modify the `Flujo` runner to include the `TraceManager` by default.
   - [ ] Add a `trace_tree` field to `PipelineResult`.
   - [ ] Write unit tests for the `TraceManager`.

2. **Phase 2: Database Schema and Persistence (1 day)**
   - [ ] Add the `traces` table to the `SQLiteBackend` schema.
   - [ ] Implement `save_trace` and `get_trace` in the `SQLiteBackend`.
   - [ ] Update `StateManager` to call `save_trace` on run completion.
   - [ ] Add unit tests for the new backend methods.

3. **Phase 3: CLI Implementation (1 day)**
   - [ ] Implement the `flujo lens trace` command.
   - [ ] Implement the `_reconstruct_and_render_tree` helper function using `rich.Tree`.
   - [ ] Write integration tests for the CLI command with various pipeline structures.

4. **Phase 4: Documentation & Finalization (1 day)**
   - [ ] Update documentation to explain the new default tracing and the `flujo lens trace` command.
   - [ ] Ensure all tests pass `make test`, `make testcov`, and `make all`.

## 8. Risks and Mitigation

| Risk | Impact | Mitigation |
| :--- | :--- | :--- |
| **Performance Overhead of Tracing:** The default hook could slow down very simple, fast pipelines. | Low-Medium | The `TraceManager` will be implemented with a focus on performance (using simple dataclasses, avoiding complex logic). NFR-11 will be used to benchmark and validate. The hook could be made disable-able via a `Flujo` constructor argument if needed. |
| **Complexity in Trace Rendering:** Rendering a deeply nested or very wide trace in the terminal can be complex and lead to a poor user experience. | Low | The `rich.Tree` object is well-suited for this. The implementation will start with a simple, readable format and can be enhanced with folding or truncation features later if needed. |
| **Large Trace Payloads:** For extremely long-running pipelines (e.g., thousands of loop iterations), the in-memory `trace_tree` could become large. | Low | This is an edge case. For now, the in-memory approach is sufficient. Future optimizations could stream spans directly to the database rather than holding the entire tree in memory until the end of the run. |
