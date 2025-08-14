 

### **Next FSD: The Flujo Compiler Linter (FSD-015)**

This directly addresses **Gap #9 (YAML DSL safety at compile time)** by focusing on pre-run validation. It's the perfect next step because it leverages the stable core to provide immediate value and safety to developers.

#### **Functional Specification Document: The Flujo Compiler Linter (FSD-015)**

**Author:** Alvaro
**Date:** 2023-10-27
**Status:** Proposed
**JIRA/Ticket:** FLUJO-130

**1. Overview**
This document specifies the creation of an enhanced pre-run validation system, or "linter," for Flujo pipelines. Currently, `pipeline.validate_graph()` checks for basic structural issues, but many logical errors (e.g., merge conflicts, impossible budgets) are only caught at runtime. This FSD proposes expanding the validation capabilities to catch these errors *before* a pipeline is executed, improving developer experience and preventing entire classes of runtime failures.

**2. Rationale & Goals**
*   **Goal:** Provide developers with fast, actionable feedback on pipeline design flaws.
*   **Goal:** Prevent logically invalid pipelines from ever starting a run.
*   **Goal:** Improve the safety and predictability of the DSL.
*   **Goal:** Lay the groundwork for a future YAML-to-Pipeline compilation step.

**3. Functional Requirements & Design**

**Task 3.1: Enhance `ValidationReport` and `ValidationFinding`**
*   **Location:** `flujo/domain/pipeline_validation.py`
*   **Details:** Add a `suggestion: Optional[str]` field to `ValidationFinding` to provide users with concrete advice on how to fix a detected issue.

**Task 3.2: Implement Advanced Static Checks in `validate_graph`**
*   **Location:** `flujo/domain/dsl/pipeline.py`
*   **Details:** Add new checks to the `validate_graph` method:
    1.  **Context Merge Conflict Detection:** For every `ParallelStep`, analyze the branches. If two or more branches update the same context field and the `merge_strategy` is the default `CONTEXT_UPDATE` without a `field_mapping` for that field, add a `ValidationFinding` error.
        *   *Suggestion:* `"Set an explicit MergeStrategy (e.g., OVERWRITE) or provide a field_mapping for the conflicting key '...'' on ParallelStep '...'."*
    2.  **Unbound Output Warning:** Traverse the pipeline graph. If a step produces an output but the next step does not consume it (e.g., its input type is `None` or it doesn't match), and the step does not have `updates_context=True`, add a `ValidationFinding` warning.
        *   *Suggestion:* `"The output of step '...' is not used by the next step. Did you mean to set updates_context=True or add an adapter step?"*
    3.  **Incompatible Fallback Signature Check:** For any step with a `fallback_step`, check that the fallback's input type is compatible with the original step's input type. If not, add a `ValidationFinding` error.

**Task 3.3: Integrate Linter into the CLI**
*   **Location:** `flujo/cli/main.py`
*   **Details:**
    1.  The `flujo validate` command will now output the richer error and warning messages, including suggestions.
    2.  The `flujo run` command will automatically call `pipeline.validate_graph(raise_on_error=True)` after loading the pipeline. If validation fails, the run will be aborted with a clear error message before any execution begins.

**4. Testing Strategy**
*   **Unit Tests (`make test-fast`):**
    *   Create new unit tests in `tests/domain/dsl/test_pipeline.py` for each new validation rule.
    *   For example, `test_validate_graph_detects_merge_conflict` will create a `ParallelStep` with a known conflict and assert that the correct `ValidationFinding` is produced.
*   **E2E Tests (`make all`):**
    *   Create CLI tests in `tests/cli/test_main.py`.
    *   Test `flujo run` on an invalid pipeline file and assert that it exits with a non-zero status code and prints the expected validation error.
    *   Test `flujo validate` on the same file and check for the correct output.

