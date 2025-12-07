````{"variant":"document","title":"PRD — Poetiq-Inspired Self-Improving Reasoning Layer","id":"82137"}
# PRD — Poetiq-Inspired Self-Improving Reasoning Layer

## 0. Document Info

- **Product:** Self-Improving Reasoning Layer on Top of LLMs  
- **Author:** (TBD)  
- **Owner / DRI:** (TBD)  
- **Version:** v1.0  
- **Status:** Draft  

---

## 1. Problem Statement

Modern LLM-based systems are powerful but:

- **Inconsistent**: Same task can get very different answers.
- **Hard to trust**: Hallucinations and missing checks are common.
- **Expensive to improve**: Typical improvement (RLHF / fine-tuning) requires massive data and compute.
- **Hard to “lock in” wins**: When a good reasoning pattern is discovered, it’s not trivial to freeze and reuse it as a stable solver.

We need a **structured, self-improving reasoning layer** that:

- Sits **on top of** existing LLMs (no retraining).
- **Learns from experience** by analyzing how tasks were solved.
- Provides **rich feedback loops** so the system can refine its own solutions.
- Supports **frozen, versioned solvers** for critical workflows.
- Enables **multi-expert consensus** for high-stakes tasks.

---

## 2. Goals & Non-Goals

### 2.1 Goals

1. **Test-time self-improvement for individual tasks**  
   - Implement an iterative loop: “propose → execute → diff → refine”, inspired by Poetiq’s ARC solver.

2. **Reusable “Frozen Solvers”**  
   - Provide a way to package a successful configuration (strategy + prompts + scoring + panel settings) into a versioned solver that can be reused and deployed.

3. **Multi-Expert Panel & Consensus**  
   - Allow multiple expert configurations to run in parallel on the same task and produce a consensus answer (with diversity preserved when needed).

4. **Meta-System for Strategy Learning (MVP)**  
   - Collect detailed reasoning traces and performance metrics.
   - Use them to:
     - pick strategies per task type,
     - propose improved strategies offline.

5. **Rich Feedback & Diff Engine**  
   - Create a generic engine to compute structured metrics and diffs between candidate outputs and desired behavior (or reference signals), and translate them into LLM-friendly feedback.

6. **Safety / Quality via Constraints**  
   - Enforce basic invariants (e.g., schema correctness, no empty output, domain-specific checks).
   - Guard actions (e.g., prevent finalization when checks fail).

### 2.2 Non-Goals (for now)

- Training or fine-tuning new LLMs.
- Building a full GUl for non-technical end users (initially this is a backend / infra product for internal tools).
- Formal theorem-proving level guarantees (B / TLA+ models may be aspirational, not in MVP).
- Replacing all existing agents / pipelines — initial integration will be for a subset of workflows.

---

## 3. Target Users & Use Cases

### 3.1 Primary Users

- **AI / Data Engineers**  
  - Build and maintain LLM-powered workflows (e.g., OMOP mapping, QC pipelines, analysis assistants).
- **Applied Researchers / Domain Experts**  
  - Consume stable, more reliable solvers for specific tasks.
  - Provide feedback on quality / failure modes.

### 3.2 Representative Use Cases

1. **Concept Mapping / Cohort Definition**
   - Generate and refine OMOP concept sets from textual definitions.
   - Run mappings against test data; evaluate coverage and error.
   - Use feedback to iteratively improve mappings.

2. **Cohort QC Explainer**
   - Generate explanations of why cohort counts differ between versions.
   - Use metrics/diffs to drive refinement of explanations and queries.

3. **Literature / Evidence Synthesis**
   - Produce structured summaries with evidence links.
   - Detect missing key references or contradictions.

4. **General Analytical Tasks**
   - Multi-step reasoning (decomposition, retrieval, synthesis).
   - Benefit from panel mode for consensus and robustness.

---

## 4. High-Level Concept

We introduce a **Reasoning Orchestrator** that sits between:

- **Requests** (tasks),
- **LLMs + tools** (underlying capabilities),
- **Evaluators** (tests, critics, humans).

Key ideas:

1. Treat the LLM as a **black-box expert**.
2. Represent reasoning as a **sequence of structured steps**.
3. Implement a **per-task iterative improvement loop** using execution feedback & diffs.
4. Allow running **multiple experts** in parallel and aggregating via consensus.
5. Continuously log **traces** and use them offline to improve strategies and configs.

---

## 5. System Components

### 5.1 Reasoning Orchestrator

**Responsibilities**

- Execute reasoning **strategies**:
  - call LLMs and tools,
  - maintain intermediate state,
  - enforce invariants & guards.
- Integrate with Feedback Engine.
- Support both:
  - **single-expert** mode, and
  - **multi-expert** panel mode.

**Requirements**

- Ability to run multiple steps:
  - Decompose → Retrieve → Generate → Critic → Refine → Finalize.
- Support explicit state object (`ReasoningState`) that is:
  - serializable,
  - logged per step,
  - type-checked.

---

### 5.2 Reasoning State

A shared structure, available to all strategies:

**Fields (MVP)**

- `task_id`: unique identifier.
- `task_spec`: canonical representation of the user request.
- `inputs`: raw data / examples / context.
- `intermediate_artifacts`:
  - candidate outputs,
  - evidence,
  - notes.
- `feedback_history`:
  - attempts,
  - metrics,
  - textual feedback,
  - timestamps.
- `final_output`: chosen final answer (once done).
- `meta`:
  - strategy_id,
  - expert_id,
  - panel_id (if applicable),
  - token usage, runtime, etc.

**Constraints (initial invariants)**

- `final_output` must conform to a contract (schema) when present.
- Every attempt must have an associated metric set (even if partial).
- ReasoningState must always be serializable to JSON for logging.

---

### 5.3 Strategy DSL & Strategy Registry

**Strategy DSL**

A simple configuration format (YAML/JSON) that describes:

- **When** to use the strategy:
  - domains, task tags.
- **How** to solve tasks:
  - sequences / patterns of operators (high level).
- **Iterative behavior**:
  - max attempts,
  - which feedback types to use,
  - thresholds for success.
- **Evaluation configuration**:
  - which tests to run,
  - which metrics to compute.

**Strategy Registry**

- Stores versioned strategies:
  - `strategy_id`: e.g., `omop_mapping_v1.3`.
  - metadata: description, owner, status (experimental, production).
- Provides an API to:
  - list strategies,
  - fetch by ID / tag,
  - mark deprecated or archived.

---

### 5.4 Operators (Reasoning Steps)

Operators encapsulate single steps the Orchestrator can take.

**Examples**

- `DECOMPOSE`: call LLM to break down task into sub-tasks.
- `RETRIEVE`: call DBs/APIs for data.
- `GENERATE`: call LLM to produce content / code / mappings.
- `CRITIC`: call LLM in critic mode to evaluate outputs.
- `REFINE`: call LLM to improve outputs given feedback.
- `EVAL`: run Evaluation Engine.
- `STOP`: finalize answer if conditions are met.

**Requirements**

- Each operator defines:
  - `guard(state) -> bool`: when allowed.
  - `execute(state, config) -> state`: deterministic wrapper around LLM/tools.
- Orchestrator must:
  - enforce guards,
  - log each operator invocation and result.

---

### 5.5 Feedback & Diff Engine

The heart of Poetiq-inspired improvement.

**Responsibilities**

- Given:
  - a **candidate output** (or program / mapping / explanation),
  - **reference signals** (e.g., expected outputs, baseline results, known constraints),
- Compute:
  - **numeric metrics** (soft scores, coverage, accuracy, etc.),
  - **structured diffs** (what changed / what’s wrong),
  - **LLM-ready feedback text**.

**Requirements**

- Must be pluggable by domain:
  - e.g., `omop_mapping_feedback`, `cohort_qc_feedback`, `generic_text_feedback`.
- Must produce:
  - a machine-readable summary for meta-system,
  - a textual summary for LLM refine prompts.
- Examples of feedback formats:
  - “You mis-mapped X… coverage is Y% vs target Z%… you missed codes A, B, C…”
  - “Your explanation contradicts known fact F… missing discussion of risk R…”

---

### 5.6 Iterative Self-Refinement Loop

Poetiq-style “propose → execute → diff → refine” at the task level.

**Flow**

1. **Initial Attempt**
   - Strategy triggers LLM to produce a first solution (e.g., mapping, code, explanation).
2. **Execution**
   - Run solution through relevant tools / pipelines.
3. **Feedback**
   - Feedback Engine computes metrics & diffs.
4. **Check Thresholds**
   - If all success criteria satisfied, finalize.
   - If not, and attempts < max_attempts:
     - store attempt + feedback in `feedback_history`.
     - generate a refine prompt using selected prior attempts + their feedback.
5. **Refinement**
   - LLM proposes improved solution.
6. **Repeat** until:
   - success, or
   - max attempts reached.

**Requirements**

- Strategy must be able to configure:
  - max attempts,
  - when to refine vs restart vs stop,
  - which prior attempts to show to LLM (e.g., random subset, best scoring).

---

### 5.7 Multi-Expert Panel & Consensus

Inspired by Poetiq’s multi-expert ARC solver.

**Concept**

- For high-stakes tasks, run **multiple experts** in parallel, each with:
  - different strategy config,
  - or different model, temperature, seed.
- Aggregate their outputs to produce:
  - a consensus answer,
  - or a ranked set of candidate answers.

**Requirements**

- Ability to define a `panel` config:
  - list of `expert_configs` (each referencing a strategy + model config),
  - evaluation rules for consensus.
- For each task:
  - run experts concurrently (within budget).
- Group results:
  - group by identical / near-identical `final_output` “signature”.
- Ranking:
  - rank groups by:
    - group size (votes),
    - average evaluation score,
    - stability (e.g., fewer errors).
- Output:
  - final answer = best group’s representative,
  - optionally: exposure of secondary candidates + justification.

---

### 5.8 Frozen Solvers

A “frozen solver” is a **versioned, ready-to-use configuration bundle** produced by the meta-system (or manually) for a specific domain/task.

**Contents**

- Fixed:
  - strategy (or set of strategies),
  - operator set,
  - panel configuration,
  - evaluation config,
  - LLM/model IDs & parameters.
- Metadata:
  - domain, description,
  - training / evaluation benchmarks,
  - version, changelog.

**Requirements**

- Ability to:
  - register a Frozen Solver in a registry,
  - call it via a simple API: `run_solver(solver_id, task_payload)`,
  - guarantee stable behavior across calls (absent upstream LLM drift).

---

### 5.9 Meta-System (Strategy Learning, v1)

Initial meta-system: not fully automatic, but enough to learn from traces.

**Responsibilities**

- Collect:
  - all traces (ReasoningStates over time),
  - per-strategy performance metrics.
- Provide:
  - dashboards / reports of strategy performance by domain & task type.
- Simple **strategy selection logic**:
  - e.g., for domain D and task type T, select top-K strategies by past performance.
- Offline **strategy evolution**:
  - periodic jobs that:
    - call LLM to propose improved strategies,
    - test them on archived tasks,
    - update registry / label winners.

**Requirements**

- Data schema for trace storage.
- APIs for querying performance by strategy, domain, time.
- Hooks for offline experimentation pipelines.

---

## 6. Functional Requirements

### 6.1 Task Intake & Dispatch

1. The system SHALL accept a task with:
   - `task_id`,
   - `domain` / `task_type`,
   - input payload (structured or unstructured).
2. The system SHALL:
   - look up applicable Frozen Solver or strategy set for the domain/task_type.
   - dispatch to:
     - single solver, or
     - panel, depending on configuration.

---

### 6.2 Reasoning Execution

1. The Orchestrator SHALL execute strategies step-by-step, enforcing operator guards and invariants.
2. The Orchestrator SHALL update `ReasoningState` after each operation.
3. The Orchestrator SHALL stop under any of these conditions:
   - success criteria met,
   - attempts or search budget exhausted,
   - hard error flagged (e.g., repeated invariant violation).

---

### 6.3 Feedback Engine

1. The Feedback Engine SHALL support pluggable feedback modules per domain.
2. It SHALL compute:
   - numerical scores (e.g., soft scores, coverage),
   - structured diffs where applicable,
   - textual feedback suitable for an LLM refine prompt.
3. It SHALL expose APIs like:
   - `compute_feedback(domain, candidate_output, reference_context) -> FeedbackResult`.

---

### 6.4 Self-Refinement Loop

1. Strategies SHALL be able to:
   - configure a max number of refine attempts.
2. For each attempt:
   - the Orchestrator SHALL call the Feedback Engine.
   - if feedback indicates failure but promising (score above “hopeless” threshold), Orchestrator SHALL attempt refinement.
3. The system SHALL log all refine iterations with:
   - attempts,
   - feedback,
   - metrics,
   - whether final success was achieved.

---

### 6.5 Multi-Expert Panel

1. A panel definition SHALL contain:
   - list of expert configurations,
   - evaluation/consensus rules.
2. The Orchestrator SHALL run experts concurrently (subject to resource limits).
3. The system SHALL:
   - group expert results based on output signature,
   - rank groups by configured criteria (votes, scores),
   - select a final group and representative result.
4. The system SHALL optionally:
   - return metadata about all experts’ outputs to allow inspection.

---

### 6.6 Frozen Solver Execution

1. A Frozen Solver SHALL be invocable via:
   - `run_solver(solver_id, task_payload)`.
2. The system SHALL:
   - resolve solver config from registry,
   - run the solver without dynamic strategy changes.
3. The system SHALL:
   - record metrics per solver version,
   - support rollout strategies (e.g., canary old vs new solver).

---

### 6.7 Meta-System & Telemetry

1. The system SHALL log:
   - per-operator step data,
   - per-iteration feedback & metrics,
   - per-task final outcomes.
2. The system SHALL expose:
   - an API/DB to query performance by strategy, solver, domain and time.
3. The meta-system (MVP) SHALL:
   - compute basic statistics:
     - mean success rate per strategy/domain,
     - average # attempts, tokens, runtime.
   - support manual dashboards for analysis.

---

## 7. Non-Functional Requirements

- **Reliability**
  - No task should be dropped silently.
  - Logs must be sufficient to reconstruct any run.
- **Performance**
  - Single-expert runs should complete within configurable targets (e.g., < N seconds for standard tasks).
  - Panel runs should degrade gracefully (configurable max experts, timeouts).
- **Scalability**
  - Must support parallel runs across tasks and panels.
- **Observability**
  - Structured logs and metrics for:
    - per-step latency,
    - failure modes,
    - tokens usage.
- **Security**
  - No sensitive data should leak in logs.
  - Feedback Engine must be careful when using external tools (PHI, etc.).
- **Extensibility**
  - Easy to add:
    - new domains,
    - new feedback modules,
    - new strategies.

---

## 8. Data & Schemas (MVP Sketch)

### 8.1 ReasoningState (JSON-ish)

```json
{
  "task_id": "string",
  "domain": "string",
  "task_type": "string",
  "strategy_id": "string",
  "expert_id": "string",
  "panel_id": "string | null",
  "inputs": {...},
  "intermediate_artifacts": [...],
  "feedback_history": [
    {
      "attempt_index": 0,
      "candidate_output": {...},
      "metrics": {"score": 0.85, "...": "..."},
      "feedback_text": "string",
      "timestamp": "ISO8601"
    }
  ],
  "final_output": {...} ,
  "meta": {
    "tokens_used": 1234,
    "runtime_ms": 5678,
    "status": "success | fail | partial"
  }
}
```

---

### 8.2 Strategy Definition (YAML sketch)

```yaml
id: "omop_mapping_v1"
description: "Mapping condition text to OMOP concepts with refinement."
domain: ["omop", "mapping"]
status: "experimental"

operators:
  - name: DECOMPOSE
    when: "initial"
  - name: GENERATE_MAPPING
    when: "after_decompose"
  - name: EVAL_MAPPING
    when: "after_generate"
  - name: REFINE
    when: "eval_failed"

iteration_policy:
  max_attempts: 5
  refine_threshold:
    min_score: 0.3     # below => hopeless, stop
    target_score: 0.9  # above => success

feedback_config:
  module: "omop_mapping_feedback"
  include_examples: 3
```

---

## 9. Rollout & Phasing

### Phase 0 — Design & Scaffolding

- Define core interfaces:
  - ReasoningState, Strategy, Operator, FeedbackEngine, Panel, Solver.
- Implement simple single-expert loop with **one** strategy and **one** feedback module.

### Phase 1 — Feedback & Self-Refinement

- Implement Feedback Engine for 1–2 concrete domains (e.g., OMOP mapping, small QC task).
- Enable iterative refine loop with logging.
- Evaluate quality vs. existing baseline prompts.

### Phase 2 — Panels & Frozen Solvers

- Implement multi-expert panel execution + consensus.
- Define “Frozen Solver” schema and registry.
- Create and test first frozen solver for a well-bounded use case.

### Phase 3 — Meta-System MVP

- Build trace logging & telemetry.
- Add simple strategy performance dashboards.
- Implement basic strategy selection and offline evolution workflows.

### Phase 4 — Hardening & Expansion

- Expand to more domains and tasks.
- Strengthen invariants & guards.
- Optimize performance and cost.

---

## 10. Open Questions

- How aggressively should we use **different models** in panel mode (e.g., mixing vendors) vs. staying with one?
- Which **initial domain** gives the clearest win (OMOP mapping, QC, or something simpler)?
- How much of the meta-system should be:
  - automated vs
  - a tool for humans to make decisions (semi-automatic tuning)?

---

## 11. Success Metrics (v1)

- **Quality**
  - ≥ X% improvement in task success rate vs baseline system for targeted workflows.
- **Reliability**
  - ≥ Y% reduction in “obvious hallucination” incidents in evaluated tasks.
- **Efficiency**
  - Self-refinement converges to success in ≤ N attempts on ≥ P% of tasks.
- **Adoption**
  - At least K workflows migrated to Frozen Solvers within T months.
- **Learning**
  - Evidence that performance improves over time with more traces, without retraining base LLMs.

---

This PRD defines the scope, components, and behavior of a Poetiq-inspired self-improving reasoning layer that can sit on top of your existing LLM stack, adding structured search, feedback-driven refinement, multi-expert consensus, and a simple meta-system for continuous improvement.
````
