# PRD — Flujo Evolution: The "Robust Reasoning" Engine

## 0. Document Info
- **Product:** Flujo Framework Extensions
- **Goal:** Unify "Industrial Reliability" (Poetiq) with "Deep Reasoning" (A* Search) into a single coherent roadmap.
- **Status:** Draft
- **Version:** 2.0

---

## 1. Problem Statement & Strategic Intent

We face two distinct but related problems in current LLM orchestration:

1.  **Reliability (The "Body"):** Agents are brittle. They produce inconsistent outputs, fail silent checks, and lack structured mechanisms to "lock in" winning strategies. We need robust error correction, consensus, and versioning.
2.  **Reasoning Depth (The "Brain"):** Agents are linear. They blindly follow a chain-of-thought. If they make a logic error early on, they cannot backtrack; they just hallucinate to justify the error. We need non-linear exploration.

**The Solution:**
We will evolve Flujo in two distinct phases.
*   **Phase 1** builds the **Infrastructure Layer** (Feedback, Diffing, Consensus) to ensure agents can evaluate themselves accurately.
*   **Phase 2** builds the **Search Layer** (A*) which *uses* the infrastructure from Phase 1 to navigate complex solution spaces efficiently.

---

## 2. Phase 1: The Infrastructure Layer (Poetiq-Inspired)
**Objective:** Enable "Test-Time Compute" for **Refinement**. The agent should be able to check its own work and fix it, or ask a panel of experts.

### 2.1 Core Capability: The Feedback & Diff Engine
We move beyond simple boolean validators (`is_valid: True/False`).
*   **Structured Diffs:** Implement a processor that compares `candidate_output` vs. `schema/reference` and returns a JSON patch or structured delta (e.g., "Missing keys: [A, B]", "Value X is out of range").
*   **Flujo Implementation:**
    *   Extend `ValidationResult` to support a `diff` field.
    *   Create `DiffProcessor` in `flujo/processors` that utilizes generic diff tools (DeepDiff) or LLM-based semantic diffs.

### 2.2 Core Capability: Consensus (The Panel)
Run multiple agents and pick the winner. This acts as a primitive "Beam Search" with depth=1.
*   **Mechanism:**
    *   Run `ParallelStep` with $N$ branches.
    *   Introduce a **Reducer Strategy** (Voting/Consensus).
*   **Flujo Implementation:**
    *   Update `ParallelStep` to support a `reduce` callable that accepts `List[StepResult]` and returns a single `StepResult`.
    *   Implement standard reducers: `majority_vote`, `code_consensus` (run all, see which output is identical), and `judge_selection` (LLM picks best).

### 2.3 Core Capability: Frozen Solvers
Move from "prompt engineering" to "versioned software artifacts."
*   **Mechanism:**
    *   A "Solver" is a specific combination of: `Blueprint (YAML)` + `Prompts` + `Config` + `Commit Hash`.
*   **Flujo Implementation:**
    *   Leverage `flujo/infra/registry.py`.
    *   Add a lock-file mechanism (`flujo.lock`) that records the exact hashes of all skills and prompts used in a successful pipeline run.

---

## 3. Phase 2: The Search Layer (A* / Tree Search)
**Objective:** Enable "Test-Time Compute" for **Exploration**. The agent can backtrack, branch, and find optimal paths using the *Feedback Engine* from Phase 1 as its compass.

### 3.1 New Primitive: `TreeSearchStep`
We introduce a specialized step that encapsulates the complexity of the A* algorithm. It does **not** use the standard `Pipeline` graph for internal nodes to avoid database bloat.

*   **Inputs:**
    *   `Proposer`: Agent that generates $N$ next steps.
    *   `Evaluator` (The Phase 1 Feedback Engine): Returns a score ($h$) and feedback.
    *   `CostFunction`: Calculates $g$ (depth/tokens).

*   **Internal State (The Frontier):**
    *   Maintains a Priority Queue (Open Set) of `SearchNodes`.
    *   Maintains a Closed Set of visited logical states.
    *   State is serialized to `context.scratchpad.search_state` for crash recovery, but intermediate nodes are *not* persisted as full database rows until the search concludes.

### 3.2 The Algorithm: Beam-A*
We implement a hybrid to balance optimality with token costs.
1.  **Expand:** Proposer generates $k$ candidates from the current best node.
2.  **Evaluate:** Phase 1 Evaluator scores them ($h(n) = 1.0 - score$).
3.  **Cost:** Calculate $f(n) = g(n) + h(n)$.
4.  **Prune:** Keep only the top $W$ nodes (Beam Width) in the priority queue.
5.  **Backtrack:** If the current path score drops below the score of a previous fork, the Priority Queue naturally pops the previous fork. **This is the "magic" that linear pipelines lack.**

---

## 4. Phase 3: Convergence (The Meta-System)
**Objective:** The system learns from Phase 1 and Phase 2 traces to become faster and cheaper.

### 4.1 Heuristic Tuning
*   **Problem:** A* is only as good as its heuristic ($h$). If the Evaluator is wrong, the search fails.
*   **Solution:** Use the "Shadow Evaluator" (already in Flujo) to review A* traces.
    *   Identify nodes where the Evaluator gave a high score, but the branch led to a dead end.
    *   Use `flujo improve` to update the Evaluator's prompt: "You gave this a 0.9, but it failed because X. In the future, check for X."

### 4.2 Distillation (Path-to-Prompt)
*   **Problem:** A* is expensive (many tokens).
*   **Solution:**
    *   Take the *winning path* from a successful A* run.
    *   Feed it into the `SelfImprovementAgent`.
    *   Generate a "Few-Shot Example" or a specialized prompt that encourages the model to follow that specific successful path *linearly* next time, bypassing the search.

---

## 5. Technical Comparison & Trade-offs

| Feature | Phase 1 (Poetiq/Linear) | Phase 2 (A*/Tree) |
| :--- | :--- | :--- |
| **Logic** | Iterative Refinement (Do -> Fix) | Tree Search (Plan -> Branch -> Backtrack) |
| **Cost Profile** | Linear ($O(N)$ attempts) | Exponential ($O(B^D)$), mitigated by Beam pruning |
| **Best Use Case** | Code generation, Data Mapping, Extract-Transform-Load | Complex Planning, Math, novel Hypothesis Generation |
| **Failure Mode** | "Spiral of Death" (fix breaks something else) | "Analysis Paralysis" (search executes forever without converging) |
| **State Storage** | Full Context History | Lightweight Node References |

## 6. Implementation Roadmap

### Milestone 1: The Reliable Operator (Weeks 1-2)
*   [ ] Implement `DiffProcessor` (JSON Patching).
*   [ ] Update `ParallelStep` to support a `consensus` reducer.
*   [ ] **Deliverable:** A pipeline that runs 3 agents, votes on the answer, and auto-corrects schema errors using diffs.

### Milestone 2: The Search Primitive (Weeks 3-4)
*   [ ] Implement `TreeSearchStep` class.
*   [ ] Implement the Priority Queue logic (iterative loop, not recursive).
*   [ ] **Deliverable:** A pipeline that solves a "Game of 24" or logic puzzle by backtracking, proving it can escape dead ends.

### Milestone 3: The Integration (Weeks 5-6)
*   [ ] Wire the Milestone 1 `DiffProcessor` as the Heuristic ($h$) for Milestone 2's `TreeSearchStep`.
*   [ ] **Deliverable:** A biomedical research agent that proposes hypotheses, scores them against literature (Heuristic), backtracks if a hypothesis is contradicted, and outputs the optimal reasoning chain.