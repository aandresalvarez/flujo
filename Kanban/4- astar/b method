The code review confirms that you have implemented a world-class **A* Search (Tree Search)** engine. It handles context isolation, iterative queue management, and crash-safe persistence perfectly.

However, as we discussed, the engine is currently **probabilistic** (it follows the highest score). To reach the "Excellence" tier, we need to add **Formal Guardrails (Invariants)** to make it **constrained**.

Below is the detailed PRD for adding **Dynamic Invariants & Formal Guardrails** to the Flujo Reasoning Engine.

---

# PRD — Flujo: Dynamic Invariants & Formal Guardrails (B-Method Inspired)

## 0. Document Info
- **Product:** Flujo Reasoning Engine (Phase 4)
- **Feature:** Formal Invariants & Dynamic Rule Deduction
- **Status:** Draft
- **Version:** 1.0

---

## 1. Problem Statement
Even with A* search, LLM-based reasoning remains susceptible to **Logic Drift**.
- **The Issue:** A "Proposer" agent might generate a high-scoring path that technically violates a core constraint (e.g., confusing MED13 with MED13L) because the "Evaluator" (the Critic) missed the nuance in one specific turn.
- **The Risk:** In scientific, medical, or legal domains, a 95% "vibe" score is a failure. We need a way to enforce **100% hard constraints** that are checked at every single node of the search tree.

---

## 2. Goals & Objectives
1.  **Zero-Tolerance Enforcement:** Implement "Hard Gates" that prune search branches immediately if they violate a rule, bypassing the expensive LLM Evaluator.
2.  **Tiered Invariant Logic:** Support both developer-defined code rules (Static) and LLM-discovered session rules (Dynamic).
3.  **B-Method Rigor:** Ensure that no state transition (node expansion) can occur unless the resulting state satisfies the "Invariant" of the system.
4.  **Token Efficiency:** Save costs by killing "impossible" branches in $O(1)$ time using local code before spending money on LLM evaluation.

---

## 3. The "Tiered Invariant" Model

| Tier | Name | Source | Implementation | Cost |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | **Structural** | Developer | Regex, JSON Schema, or Python lambda | $0.00 |
| **Tier 2** | **Logical** | Developer | DSL Expressions (e.g., `context.age > 18`) | $0.00 |
| **Tier 3** | **Semantic** | LLM (Discovery) | "Session Rules" deduced by a Discovery Agent | Low (once per run) |

---

## 4. Functional Requirements

### 4.1 The Discovery Phase
- The `TreeSearchStep` SHALL support an optional `discovery_agent`.
- Before the search begins, the `discovery_agent` SHALL analyze the `initial_prompt` to deduce "Hard Rules" for the session.
- *Example:* "Goal: Extract MED13 facts." $\rightarrow$ *Deduction:* "Rule: Any triplet containing 'MED13L' is a Hard Failure."

### 4.2 Static Invariant Definition
- The `Step` and `Pipeline` models SHALL accept a list of `invariants` (strings or callables).
- These invariants must be satisfied by the `PipelineContext` after every step.

### 4.3 Invariant-Based Pruning (The A* Guard)
- The `TreeSearchPolicy` SHALL evaluate all invariants **BEFORE** calling the Proposer or Evaluator for a node.
- If an invariant fails:
    - The node's $f(n)$ is set to **infinity**.
    - The node is moved to the `Closed Set`.
    - No further tokens are spent on that branch.

### 4.4 Targeted Feedback from Violations
- When an invariant is violated, the `diff` field in the `ValidationResult` SHALL capture the specific rule that failed.
- This `diff` SHALL be injected into the next `Proposer` call to steer the search away from the forbidden logic.

---

## 5. Technical Implementation (The "Excellence" Path)

### 5.1 Domain Model Updates
**File:** `flujo/domain/dsl/step.py`
Add `static_invariants: List[str]` and `discovery_agent: Optional[AgentLike]`.

**File:** `flujo/domain/models.py`
Update `SearchState` to include `deduced_invariants: List[str]`.

### 5.2 Policy Orchestration
**File:** `flujo/application/core/policies/tree_search_policy.py`
1.  **Initialize:** If `discovery_agent` exists, run it once and save rules to `context.tree_search_state`.
2.  **The Guard Loop:** Inside the `while frontier:` loop:
    ```python
    for rule in (step.static_invariants + state.deduced_invariants):
        if not evaluate_rule(rule, node_context):
            # Prune branch instantly
            node.f_cost = float('inf')
            state.record_trace({"event": "invariant_violation", "rule": rule})
            continue
    ```

---

## 6. Success Metrics
- **Precision:** Zero occurrences of known "forbidden" entities (e.g., MED13L) in final outputs.
- **Cost Reduction:** $\geq 20\%$ reduction in total tokens spent on deep searches (due to early pruning of bad branches).
- **Reliability:** Successful crash recovery and resumption of searches without re-running the Discovery Phase.

---

## 7. Non-Goals
- Formal mathematical proof of LLM weights (impossible).
- Real-time theorem proving (out of scope for MVP).

---

## 8. Summary of Alignment (Guide v2.0)
- **Policy-Driven:** Logic remains in `TreeSearchPolicy`, not `ExecutorCore`.
- **Type-Safe:** Uses `JSONObject` for dynamic rules.
- **Centralized Config:** Discovery behavior is toggled via `flujo.toml`.

**Conclusion:** Implementing this PRD makes Flujo the first "Logic-Safe" agent framework, capable of providing formal guarantees for scientific and industrial reasoning.