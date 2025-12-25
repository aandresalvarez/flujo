# PRD — Flujo Evolution: The "Robust Reasoning" Engine

## 0. Document Info
- **Product:** Flujo Framework Extensions
- **Goal:** Unify "Industrial Reliability" (Poetiq) with "Deep Reasoning" (A* Search) into a single coherent roadmap.
- **Status:** Draft
- **Version:** 2.4 (LLM Robustness Layer)

---

## 1. Problem Statement & Strategic Intent

We face two distinct but related problems in current LLM orchestration:

1.  **Reliability (The "Body"):** Agents are brittle. They produce inconsistent outputs, fail silent checks, and lack structured mechanisms to "lock in" winning strategies. We need robust error correction, consensus, and versioning.
2.  **Reasoning Depth (The "Brain"):** Agents are linear. They blindly follow a chain-of-thought. If they make a logic error early on, they cannot backtrack; they just hallucinate to justify the error. We need non-linear exploration.

**The Solution:**
We will evolve Flujo in two distinct phases.
*   **Phase 1** builds the **Infrastructure Layer** (Evaluator, Diffing, Consensus) to ensure agents can evaluate themselves accurately.
*   **Phase 2** builds the **Search Layer** (A*) which *uses* the infrastructure from Phase 1 to navigate complex solution spaces efficiently.

---

## 2. Phase 1: The Infrastructure Layer (Poetiq-Inspired)
**Objective:** Enable "Test-Time Compute" for **Refinement**. The agent should be able to check its own work and fix it, or ask a panel of experts.

### 2.1 Core Capability: The Multi-Signal Evaluator & Feedback Engine
To solve LLM "vibe check" calibration errors, we move beyond simple boolean/numeric scorers.
*   **The Problem:** LLMs are poor at raw numeric scoring ($0.0 - 1.0$) and often exhibit sycophancy or bias.
*   **The Solution (Multi-Signal):** The Evaluator generates a score derived from a **Verification Pipeline**:
    *   **Rubric-Based Verification:** Agents must fill out a specific `Checklist` (e.g., "Is it valid JSON?", "Does it satisfy Constraint X?"). The score is the percentage of passed checks.
    *   **Objective Signals:** Integration of programmatic checks (Regex, Pydantic validation, Unit tests) which carry higher weights.
    *   **Critic-Judge Loop:** A "Critic" agent identifies flaws, and a "Judge" agent uses the Critic's report to finalize the rubric.
*   **Structured Diffs:** Implementation of a `DiffProcessor` that returns JSON patches or structured deltas for auto-correction.

### 2.2 Core Capability: Consensus (The Panel)
Run multiple agents and pick the winner. This acts as a primitive "Beam Search" with depth=1.
*   **Mechanism:**
    *   Run `ParallelStep` with $N$ branches.
    *   Introduce a **Reducer Strategy** (Voting/Consensus).
*   **Flujo Implementation:**
    *   Update `ParallelStep` to support a `reduce` callable that accepts `List[StepResult]` and returns a single `StepResult`.
    *   Implement standard reducers: `majority_vote`, `code_consensus` (run all, see which output is identical), and `judge_selection` (using the Multi-Signal Evaluator).

### 2.3 Core Capability: Frozen Solvers
Move from "prompt engineering" to "versioned software artifacts."
*   **Mechanism:**
    *   A "Solver" is a specific combination of: `Blueprint (YAML)` + `Prompts` + `Config` + `Commit Hash`.
*   **Flujo Implementation:**
    *   Leverage `flujo/infra/registry.py`.
    *   Add a lock-file mechanism (`flujo.lock`) that records the exact hashes of all skills and prompts used in a successful pipeline run.

---

## 3. Phase 2: The Search Layer (A* / Tree Search)
**Objective:** Enable "Test-Time Compute" for **Exploration** with full crash-recovery and quota safety.

### 3.1 New Primitive: `TreeSearchStep`
We introduce a specialized step that encapsulates the complexity of the A* algorithm. To remain compliant with Flujo v2.0, it must adhere to the following persistence and safety rules:

*   **Inputs:**
    *   `Proposer`: Agent that generates $N$ next steps.
    *   `Evaluator`: The **Multi-Signal Evaluator** from Phase 1. Returns a score ($h$) and an `EvaluationReport`.
    *   `CostFunction`: Calculates $g$ (depth/tokens).

*   **Durable Frontier (Persistence):**
    *   **The Rule:** Intermediate state must be durable (Guide Section 8).
    *   **Implementation:** The Priority Queue (Open Set) and metadata for the Closed Set are stored in `context.tree_search_state`. 
    *   **Crash Recovery:** The policy updates this field after every node expansion. If the system crashes, Flujo's `resume` logic re-reads the state and pick up the search from the last frontier.

*   **Node Isolation (Idempotency):**
    *   **The Rule:** Path A must not "poison" Path B (Guide Section 3.5).
    *   **Implementation:** Every `SearchNode` contains its own `PipelineContext`, created via `ContextManager.isolate(parent_context)`. This ensures each branch of the search is mathematically independent.
    *   **Merging:** Only the context of the *winning path* is merged back into the main context upon completion.

### 3.2 The Algorithm: Quota-Aware Beam-A*
We implement a hybrid to balance optimality with token costs.

1.  **Reserve:** Before expansion, calculate a `UsageEstimate` and **Reserve** tokens from the `active_quota`.
2.  **Expand:** Proposer generates $k$ candidates from the current best node.
3.  **Isolate:** For each candidate, call `ContextManager.isolate()` to create a pristine execution environment.
4.  **Evaluate:** Run the **Multi-Signal Evaluator**. The heuristic $h(n) = 1.0 - (\text{Rubric\_Score})$.
    *   *Pruning:* If a "Hard Check" (objective signal) fails, $h = \infty$ immediately.
5.  **Cost:** Calculate $f(n) = g(n) + h(n)$.
6.  **Reconcile:** Refund unused tokens to the quota if expansion was cheaper than estimated.
7.  **Prune & Save:** Keep only top $W$ nodes (Beam Width) and update `context.tree_search_state`.
8.  **Control Flow (Pausing):** If a node triggers a `PausedException` (e.g. HITL), the policy must save the frontier and **re-raise** the exception (Guide Lesson 14).

### 3.3 LLM Robustness Layer (Production-Hardening)
LLMs have fundamental limitations that can cause A* to become a "token-burning machine." The following **5 safeguards** are mandatory:

#### 3.3.1 Context Window Overflow (The "Amnesia" Problem)
*   **The Risk:** As the search deepens, the Proposer needs the "path so far" which can exceed context limits.
*   **The Solution:** Use `HistoryManager` to produce a **Path Summary**.
*   **Implementation:** Before calling the Proposer, the policy calls `HistoryManager.summarize(path_nodes)` to keep the path representation under 2,000 tokens regardless of depth.

#### 3.3.2 Proposer Repetition (The "Infinite Loop" Problem)
*   **The Risk:** LLMs tend to propose similar or identical paths. Without deduplication, the search wastes tokens.
*   **The Solution:** Use `flujo.utils.hash.stable_digest` for **State Hashing**.
*   **Implementation:** Every candidate node is hashed before evaluation. If the hash exists in the `ClosedSet`, the candidate is **instantly killed** without calling the expensive Evaluator.

#### 3.3.3 Goal Drift (The "Forgetting the Mission" Problem)
*   **The Risk:** Deep searches may lose track of the original objective.
*   **The Solution:** **Goal Pinning** in every Proposer and Evaluator prompt.
*   **Implementation:** The prompt builder must always start with: `Primary Objective: {{ context.initial_prompt }}`. This is mandatory for all search agents.

#### 3.3.4 Proposer Quality (The "Wasted Expansion" Problem)
*   **The Risk:** Malformed or empty candidates waste evaluation cycles.
*   **The Solution:** **Candidate Pre-Filter** (lightweight, objective checks).
*   **Implementation:** Each candidate must pass a Python validator (e.g., `len(candidate) > 0`, schema check) *before* the Multi-Signal Evaluator runs.

#### 3.3.5 Non-Determinism (The "Flaky Search" Problem)
*   **The Risk:** `temperature > 0` makes searches non-reproducible and hard to debug.
*   **The Solution:** Mandate `temperature=0` for Proposer and Evaluator by default.
*   **Implementation:** The `TreeSearchStep` policy overrides the `StepConfig` of its internal agents to force `temperature=0`. A full **Search Trace** is logged for replay/debugging.

---

## 4. Phase 3: Convergence (The Meta-System)
**Objective:** The system learns from Phase 1 and Phase 2 traces to become faster and cheaper.

### 4.1 Heuristic Tuning
*   **Problem:** A* is only as good as its heuristic ($h$). If the Evaluator is wrong, the search fails.
*   **Solution:** Use the "Shadow Evaluator" (already in Flujo) to review A* traces.
    *   Identify nodes where the Evaluator gave a high score, but the branch led to a dead end.
    *   Use `flujo improve` to update the Evaluator's **Rubric** or prompt: "You gave this a 0.9, but it failed because X. In the future, check for X in the rubric."

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
| **Best Use Case** | Code generation, Data Mapping, ETL | Complex Planning, Math, novel Hypotheses |
| **State Storage** | Full Context History | **Durable Frontier State** in context |
| **Context** | Shared object (Retry isolation) | **Mandatory Isolation** per node |
| **Budget** | Reactive checks | **Proactive Quota Reservation** |
| **Scoring** | Single Agent Scorer | **Multi-Signal Verification Pipeline** |
| **LLM Safety** | Standard retry/fallback | **5-Layer Robustness** (Dedup, Goal Pin, Pre-Filter, Summarize, Determinism) |

---

## 6. Implementation Roadmap

### Milestone 1: The Reliable Operator (Weeks 1-2)
*   [ ] Implement `DiffProcessor` (JSON Patching) using `JSONObject`.
*   [ ] Build the **Multi-Signal Evaluator** framework (Rubrics + Objective Checks).
*   [ ] Update `ParallelStep` to support a `consensus` reducer using the new Evaluator.
*   [ ] **Validation:** Add unit tests for `DiffProcessor` and `ParallelStep` consensus logic.
*   [ ] **Deliverable:** A pipeline that runs 3 agents, votes, and auto-corrects schema errors using diffs.

### Milestone 2: The Durable Search Primitive (Weeks 3-4)
*   [ ] **Model Definition:** Create `SearchNode` and `SearchState` models using `JSONObject`.
*   [ ] **Policy Implementation:** Build `DefaultTreeSearchStepExecutor` in `flujo/application/core/policies/tree_search_policy.py`.
*   [ ] **State Hook:** Implement Save-on-Iteration logic to update `context.tree_search_state`.
*   [ ] **Isolation Logic:** Integrate `ContextManager.isolate()` into the expansion loop.
*   [ ] **Quota Guard:** Implement the **Reserve -> Execute -> Reconcile** pattern in the search loop.
*   [ ] **LLM Robustness Layer:**
    *   [ ] **2.1 Frontier Deduplication:** Implement `ClosedSet` using `stable_digest` hashing.
    *   [ ] **2.2 Goal Pinning:** Inject `context.initial_prompt` into every Proposer/Evaluator call.
    *   [ ] **2.3 Path Summarization:** Integrate `HistoryManager.summarize()` for deep searches.
    *   [ ] **2.4 Candidate Pre-Filter:** Add lightweight Python validators before LLM evaluation.
    *   [ ] **2.5 Deterministic Search:** Force `temperature=0` and log full Search Traces.
*   [ ] **Engineering Excellence (Tests):**
    *   [ ] **Unit Tests:** Validate Priority Queue sorting and `SearchNode` serialization.
    *   [ ] **Integration Test (Crash Recovery):** Run search, kill process, resume, and verify frontier parity.
    *   [ ] **Idempotency Test:** Assert that parallel search branches have zero context leakage.
    *   [ ] **Quota Test:** Force a limit breach and verify precise "Reserve -> Reconcile" behavior.
    *   [ ] **Deduplication Test:** Verify that repeated candidates are killed without LLM calls.
    *   [ ] **Goal Drift Test:** Verify that `initial_prompt` is present in all search agent prompts.
*   [ ] **Deliverable:** A search pipeline that can be paused, resumed, respects cost limits, and is immune to LLM drift/repetition.

### Milestone 3: The Integration (Weeks 5-6)
*   [ ] Wire the Milestone 1 `DiffProcessor` as the Heuristic ($h$) for Milestone 2's `TreeSearchStep`.
*   [ ] **Validation:** End-to-end test of A* with backtracking on a logic puzzle (e.g. "Game of 24").
*   [ ] **Deliverable:** A biomedical research agent that proposes hypotheses, scores them, backtracks if contradicted, and outputs optimal reasoning.

### Milestone 4: Documentation & Knowledge Transfer (Week 7)
*   [ ] **Technical Documentation:** Document the `TreeSearchStep` YAML schema and configuration in `docs/user_guide/pipeline_dsl.md`.
*   [ ] **Architectural ADR:** Record the A* state persistence, context isolation, and LLM robustness strategies in `docs/adr/`.
*   [ ] **Cookbook:** Create `docs/cookbook/reasoning_with_astar.md` showing the "Game of 24" implementation.
*   [ ] **Developer Guide:** Update `FLUJO_TEAM_GUIDE.md` with search layer patterns.
