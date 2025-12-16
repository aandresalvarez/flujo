# PRD v12: Granular Execution Mode (Strict, Idempotent, Policy-Driven)

| Metadata | Details |
| :--- | :--- |
| **Feature** | Granular Execution (Resumable Agents) |
| **Target Version** | Flujo 0.7.0 |
| **Status** | Approved for Implementation |
| **Priority** | High (Production Reliability) |
| **Dependencies** | `pydantic-ai >= 0.0.18`, `flujo.cost`, durable artifact store |

---

## 1. Problem Definition (Why)
Atomic agent execution loses work and can double-run non-idempotent tools after crashes. Quota accounting drifts when retries refund already-spent tokens. Resuming with altered prompts/tools causes divergence.

---

## 2. Goals / Non-Goals
- **Goals:** Crash-safe resume per turn; no double-execution; correct usage accounting; deterministic prompt/tool fingerprinting; bounded history with replay fidelity; policy-driven compliance with Flujo architecture.
- **Non-Goals:** Changing default atomic mode; altering runner control-flow semantics; introducing new state backends.

---

## 3. Solution Overview
- **Compiler Pattern:** `Step.granular(...)` → `Pipeline(LoopStep(GranularStep))`.
- **Policy:** `GranularAgentStepExecutor` executes exactly one turn; registered in policy registry (no `ExecutorCore` branching).
- **State:** `granular_state` persisted in `context.granular_state` with CAS guard + durable blobs.
- **First principles:** Prevent double-execution by construction (CAS + idempotency), never mis-account usage (single reconcile with truthful usage), never resume on divergent prompts/tools (mandatory fingerprint), never lose context needed for replay (blob offload + deterministic truncation).

---

## 4. State Schema (granular_state)
```python
class GranularState(TypedDict):
    turn_index: int                # committed turn count (0=start)
    history: List[Dict[str, Any]]  # PydanticAI-serialized messages
    is_complete: bool
    final_output: Any
    fingerprint: str               # SHA-256 of canonical run-shaping config
```

---

## 5. Execution Flow & Contracts (First-Principles Answers)
### 5.1 CAS & Idempotency (double-run prevention)
- `expected_index` = loop iteration; `stored_index` = state.turn_index.
- **stored_index > expected_index:** fail fast (`ResumeError`, recoverable); runner is behind—never execute. Runner action: re-fetch state and re-evaluate exit; if `max_turns` would be exceeded, abort pipeline to avoid spins.
- **stored_index == expected_index:** ghost-write case. Steps:
  - Re-validate fingerprint; if mismatch → `ResumeError(irrecoverable)`, do not run.
  - If match → **skip execution entirely** (no quota reserve/reconcile, no agent run), return stored state. Runner re-evaluates exit_condition on the returned state; if still incomplete, increment loop counter and re-check; abort if `max_turns` would be exceeded (no infinite loop).
- **stored_index < expected_index:** gap/missed commit → `ResumeError` (recoverable). Runner must re-fetch state and re-enter loop; never append over a gap.
- **Persist only when `incoming_turn_index == stored_index + 1`** using an atomic CAS/row-version check (DB compare-and-set on `turn_index` / optimistic concurrency token). On CAS violation, surface `ResumeError` (recoverable), do not merge, and require the runner to reload state before retrying. No concurrent append is allowed.

### 5.2 Prompt / Tool Fingerprint (deterministic resume)
- Canonical JSON (stable sort) of: `input_data` (dict keys sorted), `system_prompt`, `model_id`, `provider`, pricing plan, `tools` (names + SHA-256 of normalized signature derived from arg schema + fully-qualified name), `settings` (temperature, max_tokens, seed), `policy_version`, run-shaping toggles (idempotency enforcement, blob thresholds), and execution mode flags. Serialization is stable (sorted keys, UTF-8, no whitespace variance).
- Store on first turn; recompute every turn. Any mismatch → `ResumeError(irrecoverable=True)`; never run with mismatched fingerprint.
- Inputs to granular mode must be JSON-serializable; otherwise reject early with a clear error. Non-JSONable inputs fail fast with guidance.

### 5.3 Control-Flow Safety
- Re-raise `PausedException`, `PipelineAbortSignal`, `InfiniteRedirectError` unwrapped.
- LoopStep `on_failure="abort"`; inner Failure aborts loop and surfaces failure; no partial state merge.

### 5.4 Context Isolation
- Each turn runs in `ContextManager.isolate(frame.context)`; merge only on success. Failed/aborted turns must not mutate parent context.

---

## 6. Usage Accounting (Reserve → Execute → Reconcile)
- **Estimate:** model-aware input estimate (tiktoken or char heuristic) + output buffer; use provider/model pricing from `flujo.cost`.
- **Reserve:** fail fast on quota denial.
- **Track flags:** `capture_started` set when entering `capture_run_messages`; `network_attempted` set only after transport send is confirmed (post pre-flight checks, pre await).
- **Reconcile (single finally):**
  - If `capture_started` and messages exist → use provider usage (truth source).
  - If `network_attempted` but no messages (timeout/crash mid-flight) → charge baseline input (no refund); emit telemetry.
  - If no network attempt (pre-flight failure) → usage = 0.
  - One `reclaim(estimate, actual)`; never double-call. Avoid reserve/reconcile on CAS-skip path.
- Prefer provider usage when available; baseline is fallback. Consider capping refunds to avoid over-credit when provider usage is missing.

---

## 7. Side-Effect Safety (Idempotency Keys)
- Key: `sha256(f"{run_id}:{step_name}:{turn_index}")`.
- Inject into `RunContext.deps`.
- **Enforcement (first-class):**
  - Tools declare `requires_idempotency_key: bool` in their registration metadata; non-idempotent tools must set it to True or be rejected at registration.
  - If `enforce_idempotency=True` on Step **or** tool requires it, the executor inspects outgoing tool call payloads for an `idempotency_key` argument matching the injected key; if absent or mismatched, fail the turn with `ConfigurationError`.
  - Validation hook: pre-dispatch tool wrapper checks payload/kwargs and raises before side effects execute.

---

## 8. Blob Storage & History Truncation
### 8.1 Durable Blob Offloading (replay fidelity)
- Threshold default 20KB (configurable).
- Large tool return → move payload to **durable artifact store** (same persistence scope as state; e.g., DB-backed artifacts table or persisted object store) with retention policy (TTL + size caps). Replace message content with `<<FL_BLOB_REF:{blob_id}:size={n}>>`.
- On resume: resolve refs; if found, hydrate in-memory for the turn; if missing or store unavailable, **fail fast** by default. Degraded replay with markers is allowed only if the step opts in and the tool is idempotent, and the runner logs a clear warning.
- Artifact store durability across process restarts is required; define cleanup/retention and the explicit error surfaced on missing refs.

### 8.2 Middle-Out Truncation (Deterministic)
- Token estimator: `flujo.utils.token_counter` (tiktoken else char heuristic).
- If `history` exceeds `history_max_tokens`:
  - Keep: system prompt (index 0) + first user message.
  - Keep: tail messages that fit remaining budget (computed deterministically from remaining tokens after head + placeholder).
  - Replace middle with `ModelMessage(role='system', content='... [Context Truncated: N messages omitted] ...')` where `N` is exact count dropped.
- Head/tail token budgets are deterministic and tested; placeholder schema is fixed.

---

## 9. Loop Contract
- Loop exits only when `granular_state.is_complete` is true.
- On inner Failure: abort loop, surface failure, leave last committed state intact.
- On control-flow: propagate; runner persists as per Flujo control-flow semantics.
- CAS skip path: loop returns stored state without executing or reserving quota; exit_condition is reevaluated to avoid spins.

---

## 10. API Surface
### 10.1 `Step.granular`
```python
@classmethod
def granular(
    cls,
    name: str,
    agent: Any,
    input: Any,
    max_turns: int = 20,
    history_max_tokens: int = 128_000,
    blob_threshold_bytes: int = 20_000,
    enforce_idempotency: bool = False,
) -> Pipeline:
    ...
```

### 10.2 `GranularStep` (DSL)
```python
class GranularStep(Step):
    history_max_tokens: int
    blob_threshold_bytes: int = 20_000
    enforce_idempotency: bool = False
    meta: dict = {"policy": "granular_agent"}  # for registry routing
```

### 10.3 Policy Registration
- Map `GranularStep` → `GranularAgentStepExecutor` in policy registry; no `ExecutorCore` branching.

---

## 11. Testing Strategy (Deterministic-first)
| Test | Type | Objective |
| --- | --- | --- |
| `test_granular_cas_guard_skip` | Unit | stored_index == expected; fingerprint matches → skip execution, no double-append. |
| `test_granular_cas_gap_failure` | Unit | stored_index < expected → raise ResumeError; no write. |
| `test_granular_usage_baseline_on_timeout` | Unit | network_attempted True, no messages → baseline charged. |
| `test_granular_fingerprint_mismatch` | Unit | change tools/system/model → ResumeError(irrecoverable). |
| `test_idempotency_enforced_tool` | Unit | enforce_idempotency=True, tool omits key → ConfigurationError. |
| `test_blob_offload_and_rehydrate` | Integration | >20KB payload stored as blob ref; resume rehydrates. |
| `test_history_truncation_middle_out` | Unit | verify deterministic head/tail and placeholder. |
| `test_loop_abort_on_failure` | Integration | inner Failure aborts loop, state unchanged. |
| `test_cas_skip_no_quota_touch` | Unit | CAS skip path must not reserve/reclaim quota. |
| `test_blob_ref_missing_fails_fast` | Integration | missing blob ref triggers explicit failure unless caller opted into degraded mode. |

Real-LLM tests optional for acceptance; defaults to mocks for CI stability.

---

## 12. Implementation Checklist
1. DSL: Add `GranularStep` (`flujo/domain/dsl/granular.py`) with typed fields.
2. Factory: `Step.granular` builds LoopStep + GranularStep; sets `on_failure="abort"`.
3. Policy: `GranularAgentStepExecutor` in `flujo/application/core/policies/granular_policy.py`:
   - CAS guard with skip/abort rules and no-quota on skip.
   - Fingerprint validation (canonical JSON).
   - Isolation via `ContextManager.isolate()`.
   - Quota reserve→execute→reconcile (single reclaim).
   - Control-flow re-raise.
   - Idempotency key injection/enforcement.
   - Blob offload/rehydrate + truncation.
4. Blob store: durable artifact handling helper (shared with resume path) with retention/rehydration errors defined.
5. Utils: `_canonicalize_config`, `_calculate_local_token_usage`, `_hash_idempotency_key`.
6. Tests: implement table in §11 with typed fixtures (see `tests/test_types`).

---

## 13. Open Risks & Mitigations
- **Blob reference missing on resume:** fail loud with actionable error; continue with marker only if caller opts into degraded replay and tool is idempotent.
- **Over/under token estimate:** log discrepancy; prefer provider usage when available; baseline only when network_attempted; cap refunds to avoid over-credit.
- **Fingerprint drift for non-JSONable inputs:** require JSON-serializable inputs; otherwise reject with clear error.

---

## 14. Compliance with Flujo Team Guide
- Policy-driven routing only; no `ExecutorCore` branching.
- Control-flow exceptions re-raised, never wrapped.
- Context isolation and merge-on-success only.
- Pure quota pattern: Reserve → Execute → Reconcile.
- Centralized config: no direct env/TOML reads; use `ConfigManager`.

---

## 15. Developer Notes
- Prefer `make test-fast` for iteration; `make all` before PR.
- Use typed aliases (`JSONObject`, `TypedDict`) and mypy strict compliance.
- Avoid reactive “governor” patterns; quota-only.
