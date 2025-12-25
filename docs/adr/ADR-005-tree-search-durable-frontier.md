# ADR-005: Durable Tree Search Frontier

- **Status:** Accepted
- **Date:** 2025-09-06
- **Context:** A* search needs crash recovery and must avoid cross-branch context leakage while enforcing proactive quota limits.
- **Decision:** Introduce `TreeSearchStep` with policy-driven execution, persist the frontier in `context.tree_search_state` after every expansion, isolate each branch via `ContextManager.isolate()`, and enforce Reserve → Execute → Reconcile on proposer/evaluator calls. Apply LLM safeguards (goal pinning, stable-digest dedup, path summarization, candidate pre-filter, deterministic temperature).
- **Consequences:**
  - ✅ Search can pause/resume and survive process crashes without losing the open set.
  - ✅ Branch evaluation remains idempotent and safe from context poisoning.
  - ✅ Token usage is guarded proactively with quota reservations.
  - ❌ A* policies must keep state serialization stable and avoid non-deterministic branching.
