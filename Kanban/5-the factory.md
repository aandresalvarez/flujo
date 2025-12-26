 

# Phase 3: The Production Factory (Hardened)

**Objective:** Scale MED13 extraction to 10,000+ papers. 
**Primary SLO:** $\leq 0.1\%$ gene-identity mismatch (Target: $99.9\%$ Precision).
**Efficiency Target:** Reduce cost from $\$0.08$ to $\approx \$0.006$ per abstract.

---

## 1. Operational Definitions (Machine-Checkable)

To ensure zero ambiguity during execution and shadow-evaluation:

*   **Critical Fields:** `subject_gene` (must be "MED13"), `relation`, `object_entity`, and `evidence_quote`.
*   **The "Purity" Evidence Gate:** A triplet is rejected if the `evidence_quote`:
    *   Does NOT contain a token-boundary match for `\bMED13\b` (regex).
    *   CONTAINS a token-boundary match for `\bMED13L\b` (even if MED13 is also present).
    *   Does not match a verbatim substring of the source document at `evidence_span: {start, end}`.
*   **Critical Disagreement:** A comparison between a Factory run ($A$) and a Shadow A* run ($B$) returns `True` if:
    *   `A.subject_gene != B.subject_gene`.
    *   `A.object_entity` has a semantic similarity to `B.object_entity` $< 0.8$.
    *   The Factory run failed to extract a triplet that the Shadow run confirmed with high certainty.
*   **SLO Math:**
    *   **Scale Target:** $\leq 0.1\%$ mismatch ($1/1000$).
    *   **Rolling SLO Window:** Last 2,000 shadow samples.
    *   **Incident Trigger:** $\geq 1$ Critical Disagreement in a rolling window of 200 (immediate brake).

---

## 2. Deterministic Escalation Triggers

Instead of relying on model-provided "confidence," the pipeline automatically escalates from "Cheap" to "Deep" mode based on these signals:

1.  **Dual-Gene Presence:** Text contains both `\bMED13\b` AND `\bMED13L\b` anywhere in the document.
2.  **The L-Trap Proximity:** Regex detection of `MED13L` within 100 characters of a `MED13` mention.
3.  **Guardrail Conflict:** The lightweight "Binary Classifier" agent flags the paper as "MED13L" while the "Extractor" agent proposes a "MED13" triplet.
4.  **Schema/AROS Warnings:** Any coercion or validation warning triggered during the Factory run.

---

## 3. High-Assurance Input Integrity (The Lockfile++)

The `flujo.lock` prevents logic drift by signing the entire reasoning environment:

*   **Prompts:** Hashes of system, user, and discovery templates.
*   **Regex Configs:** Patterns for L-Trap detection and boundary markers.
*   **Normalization Code:** Python code hashes for text cleaning and triplet flattening.
*   **Retrieval/Tools:** JSON Schemas for all skills and version hashes for retrieval corpora.
*   **Hyper-parameters:** Pinned `model_id` (with date-suffixes), `temperature: 0`, and `seed`.

---

## 4. Systematic Trace-to-Policy Distillation

1.  **Gold Dataset:** 1,000 curated abstracts (500 MED13+, 500 Adversarial MED13L traps).
2.  **Teacher Run:** Execute the full `TreeSearchStep` (A* Search) on the Gold set to build the **Reference Trace Library**.
3.  **Synthesis:** A `SelfImprovementAgent` analyzes the library to produce a **Distilled Policy** (Guardrail Agent + Extractor Agent).
4.  **Stop Condition:** Deploy to production ONLY if the distilled policy achieves:
    *   **Precision:** $\geq 99.9\%$ (max 1 error in 1,000) on the Gold Set.
    *   **Recall:** $\geq 95\%$ of the A* Search baseline.
    *   **Schema Validity:** $100\%$ over the entire set.

---

## 5. Operations & The Durable Ledger

Every run is recorded in a **Durable Run Ledger** for forensic auditing and drift analysis.

**Ledger Schema:**
`idempotency_key | input_hash | lock_hash | escalated (bool) | escalation_reason[] | schema_valid (bool) | output_hash | total_cost | shadow_run_id | critical_disagreement (bool)`

```python
from flujo.client import TaskClient
from flujo.utils.hash import stable_digest

async def run_med13_factory_stream(paper_stream):
    async with TaskClient() as client:
        # Bounded concurrency to manage rate limits and DB pressure
        semaphore = asyncio.Semaphore(20) 
        
        async def process_paper(paper):
            async with semaphore:
                # 1. Idempotency (Skip if hash exists in StateBackend)
                key = stable_digest(paper.text)
                
                # 2. State-Aware Execution
                # The pipeline handles Two-Tier Routing (Escalation) internally.
                # Every 1/100 runs is automatically shadowed via TaskClient config.
                return await client.run_task(
                    pipeline="med13_factory_v1",
                    input_data=paper.text,
                    idempotency_key=key,
                    metadata={"doc_id": paper.id}
                )
```

---

## Next Immediate Tasks (The "V0" Sprint)

1.  **Draft the Evaluation Function:** Code the `critical_disagreement(a_star_res, distilled_res) -> bool` logic in `evaluate_results.py`.
2.  **Expand the Gold Set:** Increase the sample size to 100 MED13 and 100 MED13L-heavy abstracts.
3.  **Baseline Generation:** Run the current A* engine over this set to produce the "Teacher" traces.
4.  **Hardened Purity Check:** Replace the `med13_purity_invariant` with the regex-based token boundary check defined in Section 1.

**By strictly defining these gates, you remove "faith" from the equation and replace it with "proof."**
=========


This is the most important question for the **Excellence Roadmap**. To be a true "World Class" system, we must not build a "MED13 tool"—we must build a **General Reasoning Framework** and use MED13 as the "Stress Test."

Here is the breakdown of how this plan splits between **Flujo Core** (General) and the **MED13 Example** (Specific), including the missing **Documentation** requirement.

---

### 1. Framework (General) vs. Example (Specific)

| Feature | Where it lives | Generalization Logic |
| :--- | :--- | :--- |
| **`flujo.lock`** | **Flujo Core** | Works for any YAML pipeline. Ensures that if you change a prompt or model in *any* project, the system detects it. |
| **Two-Tier Routing** | **Flujo Core** | A new `Step` property: `on_failure: escalate_to_search`. Any step can now "fail up" to a deeper reasoning model. |
| **Shadow Evaluator** | **Flujo Core** | A built-in hook. Users just provide a `scoring_function` and a `sampling_rate`. |
| **L-Trap / Purity Gate**| **MED13 Example** | This is a **Domain-Specific Invariant**. The *ability* to run invariants is Core; the *logic* of the gene-match is the Example. |
| **Durable Ledger** | **Flujo Core** | Part of the `StateBackend`. Every Flujo run will now record its "Reasoning Path" for audit. |

---

### 2. Generalizing for Other Problems
The architecture we are building for MED13 is actually a **Template for High-Precision Tasks**.

**How a user would apply this to "Legal Contract Analysis":**
1.  **Stage A (Consensus):** 3 models agree on the "Liability Clause" text.
2.  **Tier 1 (Cheap):** A fast prompt extracts the expiration date.
3.  **The Invariant:** "Date must be in the future."
4.  **Escalation:** If the date is 1999, the Invariant fails, and the system **Escalates** to a **TreeSearchStep** to find if the date was scanned incorrectly or if the contract is expired.

---

### 3. The Implementation Plan (Including Documentation)

To achieve "Excellence," we are adding a **Workstream 0: The Manual** to the roadmap.

#### **Workstream 1: Flujo Core Features (General)**
1.  **Implement `flujo.lock`**: Add a CLI command to generate hashes for all YAML-defined prompts and model settings.
2.  **Implement `ShadowEvaluator`**: Add an async hook that samples runs and executes a "Gold Policy" (A* Search) to detect drift.
3.  **Harden `TaskClient`**: Add `idempotency_key` support to the `run_task` method to prevent duplicate spending at scale.

#### **Workstream 2: MED13 "Factory" Implementation (Specific)**
1.  **Refine `pipeline.yaml`**: Implement the two-tier routing (Linear Distilled Prompt $\rightarrow$ A* Fallback).
2.  **Implement the Purity Validator**: Create a robust regex-based Python check for token-boundary gene matches.

#### **Workstream 3: Detailed Documentation (The "Excellence" Requirement)**
As per **Section 2.1** of the Team Guide, we will add three core documents:
1.  **`docs/guides/robust_reasoning.md`**: A general guide on how to use A*, Consensus, and Invariants for *any* problem.
2.  **`docs/reference/lockfile.md`**: Technical spec on how Flujo ensures reproducibility.
3.  **`examples/med13_high_assurance/README.md`**: A step-by-step tutorial (the "Lab Manual") explaining how this specific project achieves 99.9% precision.

---

### 4. Updated Checklist for the "V0" Sprint

*   [ ] **Code:** Enhance `ValidationResult` with `score` and `diff` (General).
*   [ ] **Code:** Add `idempotency_key` to `TaskClient` (General).
*   [ ] **Example:** Build the MED13 `distilled_policy` (Specific).
*   [ ] **Documentation:** Write the `robust_reasoning.md` guide (General).

### Verdict
By building the features this way, you aren't just solving the MED13 problem. You are giving every Flujo user a **"Reasoning Kit"** they can use to build high-assurance apps for Finance, Law, and Engineering.

**Is this the level of generalization you were looking for?** If so, I will prepare the technical spec for the `flujo.lock` implementation.

-=======


These technical specifications define the transition of Flujo from a research-oriented agent runner to a **High-Assurance Reasoning System**. 

The work is divided into **Core Framework Primitives** (General) and the **MED13 Production Pipeline** (Specific).

---

# 1. Framework: The `flujo.lock` Specification
**Goal:** Guarantee logical reproducibility by detecting "Silent Drift" in prompts, models, and tools.

### 1.1 Lockfile Schema (`flujo.lock`)
The lockfile is a JSON/YAML file generated via `flujo lock`. It stores the state of the "Reasoning Environment."
```yaml
version: "1.0"
pipeline_hash: "sha256_..."
solvers:
  med13_extraction:
    model: "openai:gpt-4o-2024-08-06"
    params: { temperature: 0, seed: 42 }
    prompt_hash: "sha256_system_prompt_text"
    tool_hashes:
      pubmed_api: "sha256_json_schema_definition"
    invariant_hashes:
      purity_check: "sha256_python_source_code"
```

### 1.2 Implementation Logic
*   **Generation:** A new CLI command `flujo lock` traverses the `pipeline.yaml` and any registered `agents`. It hashes the text of prompts and the structural schemas of tools.
*   **Enforcement:** During `flujo run`, if a lockfile exists:
    *   **Strict Mode:** If current prompt/model != lock, the engine raises `RuntimeDriftError` and halts.
    *   **Audit Mode:** The mismatch is recorded in the `StateBackend` for forensic analysis.

---

# 2. Framework: Two-Tier Routing (The "Escalation" Policy)
**Goal:** Minimize cost by using "System 1" (Fast) and only invoking "System 2" (A* Search) when needed.

### 2.1 Escalation Logic
Add a new `on_failure` variant to the `Step` model: `escalate_to_search`.

```python
# flujo/domain/dsl/step.py
class StepFailureMode(str, Enum):
    ABORT = "abort"
    SKIP = "skip"
    ESCALATE_TO_SEARCH = "escalate_to_search" # New
```

### 2.2 Execution Policy
When a Tier 1 Step returns `success: False` or triggers an **Invariant Violation**:
1.  The `ExecutorCore` checks if an `escalation_step` (usually a `TreeSearchStep`) is defined.
2.  If yes, it executes the search using the **Parent Node's Context**.
3.  The `StepResult` records `escalated: True` for the ledger.

---

# 3. Framework: The Shadow Evaluator (Drift Detection)
**Goal:** Verify the precision of distilled (cheap) prompts against the "Teacher" (A* Search).

### 3.1 The Shadow Hook
A built-in `ShadowEvalHook` is added to `flujo/application/core/runtime/shadow_evaluator.py`.

*   **Config:** `sampling_rate: 0.01` (1%), `slo_threshold: 0.001` (0.1% mismatch).
*   **Operation:** 
    1.  When a run completes, the hook samples it.
    2.  It re-runs the same input through the **Gold Standard** (A* Search).
    3.  It compares results using the `Critical Disagreement` function.
    4.  If the disagreement rate exceeds the SLO, it triggers a system-wide `ConstraintBreachEvent`.

---

# 4. Specific: The MED13 "Factory" Pipeline
**Goal:** A hardened implementation of the gene-extraction project using the new primitives.

### 4.1 The Purity Validator (The Guardrail)
A robust Python-based validator is implemented as a standalone skill.

```python
# med13_project/skills/purity.py
import re

def med13_purity_check(output, context) -> ValidationResult:
    # Rule 1: Token Boundary Match
    if not re.search(r'\bMED13\b', output.evidence_quote):
        return ValidationResult(is_valid=False, score=0, feedback="Missing MED13")
        
    # Rule 2: Negative Constraint (The L-Trap)
    if re.search(r'\bMED13L\b', output.evidence_quote, re.IGNORECASE):
        return ValidationResult(is_valid=False, score=0, feedback="MED13L Contamination")
        
    return ValidationResult(is_valid=True, score=1.0)
```

### 4.2 The Orchestration (`pipeline.yaml`)
```yaml
steps:
  - name: verify_source
    kind: parallel
    reduce: consensus # Stage A
    branches: [gpt_check, claude_check]
    
  - name: fast_extract
    kind: step
    uses: "agents.distilled_extractor"
    on_failure: escalate_to_search # Two-Tier Routing
    escalation_step: "deep_search"
    
  - name: deep_search
    kind: tree_search # Stage B
    proposer: "agents.triplet_proposer"
    evaluator: "skills.purity_check"
    max_cost_usd: 1.0
```

---

# 5. Documentation Requirements
Per the **Excellence Standard**, the following manuals must be produced:

### 5.1 `docs/guides/robust_reasoning.md`
*   **Theory:** How A* ($f=g+h$) applies to text.
*   **Pattern:** The "Teacher-Student" loop (Search $\rightarrow$ Distill $\rightarrow$ Lock).
*   **Safety:** How to write B-Method invariants for LLMs.

### 5.2 `docs/reference/lockfile.md`
*   Technical spec of the hashing algorithm.
*   CLI usage guide (`flujo lock`, `flujo verify`).
*   How to handle "Authorized Drift" when intentionally updating a prompt.

### 5.3 `examples/med13_high_assurance/README.md`
*   Detailed walkthrough of the MED13 case study.
*   Instructions on how to run the Gold Set benchmark.
*   Example of a "Shadow Eval" incident report.

---

# 6. Implementation Schedule (Sprint "Excellence")

1.  **Workstream A (Domain):** Add `score` and `diff` to `ValidationResult`. (Day 1)
2.  **Workstream B (CLI):** Implement `flujo lock` command. (Day 2)
3.  **Workstream C (Policy):** Update `TreeSearchPolicy` to handle `escalation` logic. (Day 3)
4.  **Workstream D (Project):** Build the MED13 "Factory" YAML. (Day 4)
5.  **Workstream E (Docs):** Complete the Robust Reasoning manual. (Day 5)

**This plan officially moves Flujo into the "Logic-Safe" tier of AI engineering.** Ready to begin?