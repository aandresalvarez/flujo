# Flujo 2.0 Roadmap: Operational Integrity Layer

**Status:** Draft v0.1
**Date:** November 20, 2025
**Target:** Transform Flujo from a developer framework into an "Application Server for AI Agents" with durable execution, forensic auditability, and side-effect gating.

---

## 1. Vision Recap

Flujo 2.0 unifies:
1.  **Durable Execution**: Pause, resume, replay.
2.  **Forensic Auditability**: Full context capture (Integrity Ledger).
3.  **Side-Effect Gating**: Policy-enforced actions.

**Core Promise**: "If an agent does something important, we can replay exactly what it did, explain why, and prove it respected policies."

---

## 2. Gap Analysis: Current vs. 2.0

| Feature | Current State (Flujo v1) | Target State (Flujo 2.0) | Gap / Action Required |
| :--- | :--- | :--- | :--- |
| **Data Model** | `workflow_state` (blob), `steps` (log), `traces` (telemetry). | **Integrity Ledger**: Structured `IntegrityEvent` (state_before, state_after, policy_checks, side_effects). | **High**: Design new `IntegrityEvent` schema. Migrate/Replace `SQLiteBackend` storage logic. |
| **Runtime** | `ExecutorCore` handles steps, caching, and basic retries. | **Runtime Kernel**: Emits `IntegrityEvent` for *every* transition. Deterministic replay via history. | **Medium**: Instrument `ExecutorCore` to emit events. Implement "Replay Mode" using ledger history. |
| **Policy** | `UsageLimits` (quota), basic `reasoning_precheck`. | **Policy Engine**: Static (deploy-time) & Dynamic (run-time) checks. `allow/deny` logic for tools/actions. | **High**: Build `PolicyEngine` component. Define Policy DSL (YAML/Python). |
| **Side-Effects** | Direct execution in `AgentStepExecutor` or tools. | **Side-Effect Adapters**: Wrappers (HTTP, DB, FS) that consult Policy Engine before execution. | **High**: Create `flujo.adapters` (HTTP, DB, FS). Refactor agents to use them. |
| **Telemetry** | `prometheus.py`, `otel_hook.py`. | **Telemetry Exporter**: First-class OTLP export of Integrity Events. | **Medium**: Map `IntegrityEvent` -> OTLP Spans. |

---

## 3. Implementation Roadmap

### Phase 1: The Integrity Kernel (Weeks 1-4)
**Goal**: Establish the Integrity Ledger and basic Runtime Kernel updates.

- [ ] **Define Data Model**:
  - Create `IntegrityEvent` Pydantic model (Run, Step, StateSnapshot, PolicyCheck, SideEffect).
  - Design `events` table schema for SQLite.
- [ ] **Refactor Storage**:
  - Update `SQLiteBackend` to support append-only `IntegrityEvent` logging.
  - Ensure atomic transactions for event writes.
- [ ] **Runtime Instrumentation**:
  - Update `ExecutorCore` to construct and emit `IntegrityEvent` at step boundaries.
  - Capture `state_before` and `state_after` snapshots.
- [ ] **Basic Replay**:
  - Implement `flujo replay` CLI to reconstruct state from the ledger.

### Phase 2: Policy Engine & Gating (Weeks 5-8)
**Goal**: Implement the "Firewall" for agents.

- [ ] **Policy Engine Core**:
  - Create `PolicyEngine` class.
  - Implement Static Checks (e.g., `deny_tools`, `max_cost` in `pipeline.yaml`).
  - Implement Dynamic Checks (input: state + action, output: allow/deny).
- [ ] **Side-Effect Adapters**:
  - Build `flujo.adapters.http` (wrapper around `httpx`).
  - Build `flujo.adapters.fs` (safe file system ops).
  - Integrate Adapters with `PolicyEngine`.
- [ ] **Agent Integration**:
  - Update `AgentStepExecutor` to use Adapters.
  - Enforce policy checks before tool execution.

### Phase 3: Developer Experience & Telemetry (Weeks 9-12)
**Goal**: Make it usable and observable.

- [ ] **DX / SDK**:
  - Typed `Context` API (`ctx.state`).
  - `@step` decorator improvements for 2.0 semantics.
  - `flujo validate` CLI for static policy checks.
- [ ] **Telemetry Exporter**:
  - Implement `IntegrityEvent` -> OTLP exporter.
  - Verify integration with LangSmith/Langfuse.
- [ ] **Compliance**:
  - "Audit Log" export command (JSON/CSV of decisions).

---

## 4. Architecture Changes

### 4.1 New Components
- `flujo.core.ledger`: Handles event definition and storage.
- `flujo.core.policy`: Handles rule evaluation.
- `flujo.adapters`: Contains side-effect wrappers.

### 4.2 Schema Evolution (SQLite)
New `events` table:
```sql
CREATE TABLE events (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    kind TEXT NOT NULL, -- 'step', 'policy', 'side_effect'
    payload JSON NOT NULL, -- The full IntegrityEvent
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
```

### 4.3 Execution Flow
1.  **Step Start**: Load `state_before`.
2.  **Action Request**: Agent requests Tool/LLM call.
3.  **Policy Check**: `PolicyEngine` evaluates request against `pipeline.yaml`.
4.  **Execution**: If allowed, `SideEffectAdapter` executes (or returns cached result if Replaying).
5.  **Step End**: Save `state_after` + `IntegrityEvent` (inputs, outputs, decisions) to Ledger.

---

## 5. Next Steps

1.  **Approve Roadmap**: Confirm this plan aligns with your vision.
2.  **Scaffold 2.0 Structure**: Create the new directory structure (`flujo/core/ledger`, `flujo/adapters`, etc.).
3.  **Prototype Ledger**: Implement the `IntegrityEvent` model and basic SQLite storage.
