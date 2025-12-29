This updated PRD for **Phase 3** reflects the transition from a "LLM-only" approach to a **Neuro-Symbolic Factory**. It ensures that even when you scale for speed and cost, your precision is maintained by hard code sensors and mathematical proofs rather than LLM "vibes."

---

# Phase 3: The Production Factory (Hardened & Neuro-Symbolic)

**Objective:** Scale MED13 extraction to 10,000+ papers.
**Precision SLO:** $\leq 0.1\%$ gene-identity mismatch (Target: $99.9\%$ Precision).
**Architecture:** Hybrid Distillation (Compressed LLM + Persistent Python Guardrails).

---

## 1. Neuro-Symbolic Operational Definitions

Precision is achieved by deriving scores from **Deterministic Sensors** rather than LLM opinions.

*   **Critical Fields (The Identity Triad):**
    1.  `gene_uid`: Unique ID from a canonical database (e.g., NCBI:9969).
    2.  `relation_type`: Verified against a fixed Bio-Ontology.
    3.  `evidence_span`: The `{start, end}` character offsets in the source PDF.
*   **The Scoring Formula (The Proof):**
    The framework calculates the score ($h$) using a weighted sum of Boolean Sensors:
    $$Score = (0.5 \times \text{ID\_Match}) + (0.3 \times \text{Span\_Verify}) + (0.2 \times \text{Logic\_Sensor})$$
    *   **ID\_Match (Code):** `True` if the entity resolves to the target MED13 UID via Tool/DB.
    *   **Span\_Verify (Code):** `True` if `source_text[start:end]` verbatim matches the `evidence_quote`.
    *   **Logic\_Sensor (LLM):** `True` if a high-reasoning agent confirms the sentence structure supports the relation.
*   **Critical Disagreement:** A comparison between Factory ($A$) and Shadow A* ($B$) results is `True` if `A.gene_uid != B.gene_uid`.

---

## 2. Deterministic Escalation Triggers

The system automatically upgrades from "Cheap" (Linear) to "Deep" (A* Search) based on these objective signals:

1.  **Tool Ambiguity (Primary Trigger):** If the Entity Resolution tool returns $>1$ possible ID or a "Low Confidence" flag for a gene mention.
2.  **L-Trap Proximity:** Regex detection of `MED13L` within 100 characters of a `MED13` mention.
3.  **Dual-Gene Presence:** Verifiable detection of both `NCBI:9969` (MED13) and `NCBI:55186` (MED13L) within the same context chunk.
4.  **Sensor Failure:** Any "False" return from the **ID\_Match** or **Span\_Verify** code sensors in the cheap tier.

---

## 3. The Lockfile++ (Environment Integrity)

The `flujo.lock` prevents "Silent Logic Drift" by signing the **complete reasoning environment**:

*   **Reasoning Logic:** Hashes of all system prompts and `STRICT_INVARIANTS` source code.
*   **Knowledge State:** Hashes of the `gene_ontology.json` and any local knowledge graph snapshots used for resolution.
*   **Tool Interface:** JSON Schemas for external APIs (PubMed, UniProt).
*   **Hyper-parameters:** Pinned `model_id`, `temperature: 0`, `seed`, and `max_tokens`.

---

## 4. Hybrid Distillation Strategy

We do not distill "Hard Rules" into prompts. We distill only the **extraction intuition**.

*   **The Distilled Policy:** A three-step sequence:
    1.  **Agent (Fast):** "Identify the MED13 relationship and provide the quote." (Cheap LLM).
    2.  **Validator (Code):** Verify the quote is verbatim and extract character offsets. (Python).
    3.  **Refiner (Tool):** Resolve the entity to a UID via Database. (Symbolic Tool).
*   **Stop Condition:** Deploy only if this hybrid policy matches $\geq 99.9\%$ of the **Teacher (A* Search)** gene-identity assignments on a 1,000-sample validation set.

---

## 5. Operations & Shadow SLOs

The **Durable Run Ledger** provides a forensic record of every state transition.

*   **Sampling Policy:** Randomly sample 1% of successful factory runs.
*   **Safety Mode (Brakes):** If **Disagreement Rate $> 0.2\%$** (Warning) or **$> 0.5\%$** (Incident) in the rolling window of 2,000 samples:
    1.  **Auto-Escalate:** Force all Tier 1 traffic to Tier 2 (A* Search).
    2.  **Snapshot:** Generate a diff of the `flujo.lock` to check for model-weight drift from the provider.
    3.  **Audit:** Record the `diff` field from the Shadow Evaluator to pinpoint which sensor failed.

---

## 6. Documentation (The Excellence Manuals)

*   **`docs/guides/neuro_symbolic_search.md`**: How to build "Calculated Heuristics" using Boolean Sensors.
*   **`docs/reference/idempotency_and_ledger.md`**: Technical spec for the `TaskClient` durable ledger and state-aware deduplication.
*   **`examples/med13_production/LAB_MANUAL.md`**: Step-by-step instructions on setting up the Gene Identity Tool and running the 1,000-sample validation gate.

---

### Implementation Schedule

1.  **Sprint 1 (Symbolic):** Implement the `gene_uid` resolution tool and the `verbatim_span` code validator.
2.  **Sprint 2 (Reasoning):** Update `TreeSearchPolicy` to use the **Weighted Sum Formula** for node prioritization.
3.  **Sprint 3 (Integrity):** Add Database/Ontology hashing to the `flujo.lock` CLI.
4.  **Sprint 4 (Factory):** Build the Hybrid Distilled Policy and run the 1,000-sample benchmark.

**Summary:** This PRD moves Flujo from "Prompting" to "Verification." By treating the LLM as one of many sensors and calculating the score in code, you guarantee the precision required for scientific research.

To ensure "Excellence" and architectural purity, we must strictly separate the **Framework (The Engine)** from the **Example (The Fuel)**. 

Below is the added section for the **Phase 3 PRD**, clarifying how this implementation benefits every future Flujo project while keeping the MED13 logic isolated.

---

## 7. Separation of Concerns: Core vs. Project

### 7.1 Flujo Core Implementation (General Framework Value)
The following features are implemented directly in the Flujo source code. They are **domain-agnostic** and provide immediate value to any project (Legal, Finance, DevOps, etc.).

| Feature | Core Component | General Utility |
| :--- | :--- | :--- |
| **Calculated Heuristics** | `TreeSearchPolicy` | Allows A* to use any numeric `score` from a `ValidationResult` as the $h$ value. |
| **`flujo lock` CLI** | `flujo.cli.lens` | Provides a command to hash prompts, model versions, and tool schemas to prevent logic drift in any project. |
| **Escalation Policy** | `Step` & `ExecutorCore` | Adds `on_failure: escalate` logic, allowing any step to trigger a deeper search if a "Sensor" returns False. |
| **Shadow Eval Hook** | `application.runtime` | A background worker that samples $X\%$ of runs and executes a "Gold Policy" to measure production drift. |
| **Durable Ledger** | `StateBackend` | Standardizes the logging of `input_hash`, `lock_hash`, and `disagreement_flags` for forensic auditing. |
| **Custom Object Hashing** | `utils.hash` | Exposes a hook for users to include external files (like `.json` ontologies) in the pipeline's cryptographic lock. |

### 7.2 MED13 Example Implementation (Domain-Specific Logic)
The following code lives only inside the `examples/med13_high_assurance/` directory. It defines the "Rules of the World" for molecular biology.

| Feature | Location | Specific Logic |
| :--- | :--- | :--- |
| **The Purity Invariants** | `skills/purity_check.py` | The **Regex logic** for `\bMED13\b` vs `\bMED13L\b` and the token-boundary rules. |
| **Weighted Sum Formula** | `pipeline.yaml` | The specific weights assigned to sensors: `(0.5 * ID + 0.3 * Span + 0.2 * Logic)`. |
| **Identity Resolver** | `skills/gene_resolver.py` | The Python code that calls the **UniProt/NCBI API** to fetch a `gene_uid`. |
| **Critical Disagreement** | `eval/comparisons.py` | The specific rule stating that a mismatch in `gene_uid` is a "Critical" incident. |
| **Discovery Prompt** | `agents/discovery.py` | The prompt instructing the LLM to look for **Paralog Contamination** specifically. |

---

## 8. Generalization Example: Using the Core for "LegalTech"

To prove this is a "General Reasoning System," consider a user building a **Legal Contract Analyzer**. Because you implemented the features in Section 7.1, the user only has to provide the "Example" logic:

1.  **Project Logic:** A skill that checks if a "Liability Cap" is a number (The Sensor).
2.  **Project Formula:** `Score = (0.8 * CapFound) + (0.2 * JurisdictionMatch)`.
3.  **Core Feature:** The `flujo lock` ensures the "California Law" prompt doesn't drift.
4.  **Core Feature:** If the "Cheap" extractor misses the cap, the `escalate_to_search` policy triggers an A* Search to find the cap in the "Miscellaneous" section of the contract.

---

## 9. Revised Roadmap with Documentation Workstream

To comply with **Team Guide v2.0 (Section 2.1)**, we add a dedicated documentation workstream.

### **Workstream: The "Excellence" Manuals**
*   **CORE DOC (`docs/guides/robust_reasoning.md`):** Explains how to use the new `TreeSearchStep` and `ValidationResult.score` for any domain.
*   **CORE DOC (`docs/reference/lockfile.md`):** Explains the math behind the `flujo.lock` and how to verify environment integrity.
*   **EXAMPLE DOC (`examples/med13_production/README.md`):** A specialized "Lab Guide" on how to achieve 99.9% precision using the Neuro-Symbolic Factory pattern.

### **Checklist for Deployment:**
- [ ] **Core:** Update `ValidationResult` model (score/diff).
- [ ] **Core:** Add `escalate` to Step execution policy.
- [ ] **Example:** Write the MED13 `purity_check.py` using regex-token boundaries.
- [ ] **Manual:** Finalize the `neuro_symbolic_search.md` guide.

**Conclusion:** By isolating the "Engine" (Section 7.1) from the "MED13 Logic" (Section 7.2), you ensure Flujo remains a general-purpose tool while proving its power in the most difficult domain possible.