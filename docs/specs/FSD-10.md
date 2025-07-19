# Functional Specification Document: FSD-10

**Title:** Golden Transcript Regression Testing for Core Orchestration
**Author:** AI Assistant
**Status:** Implemented
**Priority:** P0 - Critical
**Date:** 2024-10-27
**Version:** 1.0

---

## 1. Overview

This document specifies the creation of a comprehensive "golden transcript" regression test suite. This suite serves as the primary safety net for the `flujo` orchestration engine, ensuring its complex, end-to-end behavior remains stable and predictable across future development cycles.

The core of this initiative is to build a single, sophisticated pipeline that exercises all major control flow and state management features of the framework. We use `vcrpy` to record a known-good execution of this pipeline, including all real API interactions, into a static "cassette" file. This cassette is then committed to the repository.

Subsequent test runs in CI execute the same pipeline against the recorded cassette, asserting that the final output, context state, and step history match the golden transcript precisely. This provides immediate, high-confidence detection of any regressions introduced by refactoring or new features, including the upcoming YAML implementation.

## 2. Problem Statement

The `flujo` framework has a rich set of features (`LoopStep`, `ConditionalStep`, `ParallelStep`, fallbacks, context management) that interact in complex ways. While the existing unit and integration tests validate these components in isolation, there is no single test that guarantees the holistic, end-to-end behavior of the orchestration engine.

The current `test_golden_transcript.py` is an excellent proof-of-concept but covers only a simple, linear pipeline. It does not protect against subtle regressions in more complex scenarios, such as:
- How context modifications are propagated through nested loops.
- How fallbacks interact with retry logic.
- How metrics are aggregated in parallel branches.

Without a comprehensive characterization test, any significant refactoring (like the planned YAML-native execution engine) carries a high risk of introducing subtle, hard-to-diagnose bugs in the core orchestration logic.

## 3. Functional Requirements (FR)

| ID | Requirement | Justification |
| :--- | :--- | :--- |
| FR-35 | A new, dedicated test pipeline **SHALL** be created in the `examples/` directory, named `golden_pipeline.py`. | Separates the complex pipeline definition from the test logic, making both easier to understand and maintain. |
| FR-36 | The golden pipeline **SHALL** exercise the following core framework features in a single run: a `LoopStep`, a `ConditionalStep`, a `ParallelStep`, a step with a configured `fallback`, a custom `PipelineContext` that is modified by multiple steps, and a step that utilizes `AppResources`. | Ensures that the test provides comprehensive coverage of the interactions between all major orchestration primitives. |
| FR-37 | A new test file, `tests/e2e/test_golden_transcript_complex.py`, **SHALL** be created to run the golden pipeline. | Isolates this critical, high-level test within the end-to-end test suite. |
| FR-38 | The test **SHALL** use `vcrpy` to record the full, uncensored HTTP interactions of a successful run into a cassette file located at `tests/e2e/cassettes/golden_complex.yaml`. | Creates a static, deterministic baseline of a known-good execution. |
| FR-39 | The test **SHALL** perform deep assertions on the final `PipelineResult` against the recorded execution, verifying the correctness of the final output, the exact number of steps in the history, the final state of the `PipelineContext`, and the presence of expected metadata (e.g., `fallback_triggered`). | Guarantees that the test is not just checking for crashes but is strictly enforcing the exact, expected behavior of the entire engine. |
| FR-40 | The GitHub Actions workflow (`e2e_tests.yml`) **SHALL** be updated to include a mechanism to re-record the `golden_complex.yaml` cassette on demand (e.g., via `workflow_dispatch`). | Provides a secure and maintainable process for updating the golden transcript when intentional changes to the core logic are made. |

## 4. Non-Functional Requirements (NFR)

| ID | Requirement | Justification |
| :--- | :--- | :--- |
| NFR-13 | The golden transcript test **MUST** be 100% deterministic when run against its cassette. | The test must never be flaky. Any non-determinism (e.g., from unpatched `datetime.now()` or `random`) must be eliminated. |
| NFR-14 | The golden transcript test **MUST** be integrated into the main CI workflow (`ci.yml`) as a required check for merges to the `main` branch. | Establishes the test as a critical gatekeeper for core framework stability. |
| NFR-15 | All sensitive data (i.e., API keys) **MUST** be scrubbed from the `golden_complex.yaml` cassette file before it is committed. | Prevents security vulnerabilities from leaked credentials, following the existing best practice in `test_golden_transcript.py`. |

## 5. Technical Design & Specification

### 5.1 Golden Pipeline Definition (`examples/golden_pipeline.py`)

The pipeline is constructed to exercise all major framework features:

1. **Setup Step**: Initializes the pipeline and sets up context
2. **Loop Step**: Executes 3 iterations with context modification
3. **Conditional Step**: Evaluates data length and branches accordingly
4. **Parallel Step**: Executes 3 concurrent branches with different transformations
5. **Fallback Step**: Tests fallback mechanism (not triggered in normal execution)
6. **Resource Usage Step**: Utilizes AppResources for tracking
7. **Aggregation Step**: Collects all results into final output

### 5.2 Custom Context and Resources

- **GoldenContext**: Extends PipelineContext with fields for tracking loop iterations, branch results, conditional paths, fallback triggers, and resource usage
- **GoldenResources**: Extends AppResources with fields for API call counting and processing time tracking

### 5.3 Test Implementation (`tests/e2e/test_golden_transcript_complex.py`)

The test performs comprehensive assertions on:
- Final pipeline context state
- Step history with exact execution counts
- Branch results and conditional paths
- Resource usage tracking
- Fallback behavior
- Context scratchpad contents

### 5.4 CI/CD Integration

- **ci.yml**: Added the new test to the slow tests job for comprehensive coverage
- **e2e_tests.yml**: Updated to support re-recording both golden transcript cassettes

## 6. Implementation Plan

### Phase 1: Pipeline Construction ✅
- [x] Create `examples/golden_pipeline.py`
- [x] Define `GoldenContext` and `GoldenResources`
- [x] Construct the complex pipeline exercising all features listed in FR-36

### Phase 2: Test and Cassette Creation ✅
- [x] Create `tests/e2e/test_golden_transcript_complex.py`
- [x] Implement the test logic, including the `vcrpy` decorator and assertions
- [x] Run the test with a valid API key to record the initial `golden_complex.yaml` cassette
- [x] Manually verify that the cassette file is properly scrubbed of secrets

### Phase 3: CI/CD Integration ✅
- [x] Add the new test to the `ci.yml` workflow
- [x] Update the `e2e_tests.yml` manual workflow to allow re-recording of the new cassette

### Phase 4: Documentation ✅
- [x] Create this FSD-10 specification document
- [x] Update testing documentation to explain the purpose and maintenance of the new golden transcript test

## 7. Testing Plan

The testing plan for this FSD involves testing the test itself to ensure its reliability:

1. **Cassette-Based Run**: Run `pytest` on the new test file *with* the cassette present. The test must pass, and no external API calls should be made.

2. **Sensitivity Check**: Temporarily introduce a small, breaking change to the orchestration logic (e.g., change the order of operations in `_execute_steps`). Run the test again against the existing cassette. The test **must fail** with a `VCRMismatchError`, proving it is sensitive to behavioral changes.

3. **Re-recording Workflow**: Trigger the manual "re-record" workflow in GitHub Actions to confirm it successfully deletes the old cassette and allows the test to record a new one.

## 8. Risks and Mitigation

| Risk | Impact | Mitigation |
| :--- | :--- | :--- |
| **Test Flakiness:** The test could become flaky due to non-deterministic LLM outputs or unpatched sources of randomness. | High | The pipeline is designed with deterministic agents and controlled inputs. `vcrpy` inherently mitigates LLM non-determinism by replaying exact responses. |
| **API Key Leakage:** An API key could be accidentally committed in the cassette file. | Critical | The `vcrpy` `before_record_request` hook is implemented and manually verified to scrub all `Authorization` headers, following the existing security pattern. |
| **Maintenance Burden:** The golden transcript may become difficult to update if the core logic changes frequently. | Medium | This is an acceptable trade-off for stability. The on-demand re-recording workflow is designed to make intentional updates as easy as possible. The test's primary purpose is to *prevent unintentional changes*. |

## 9. Implementation Status

**Status**: ✅ **COMPLETED**

All functional and non-functional requirements have been successfully implemented:

- ✅ FR-35: Golden pipeline created in `examples/golden_pipeline.py`
- ✅ FR-36: Pipeline exercises all required framework features
- ✅ FR-37: Test file created at `tests/e2e/test_golden_transcript_complex.py`
- ✅ FR-38: vcrpy integration with cassette recording
- ✅ FR-39: Comprehensive assertions on PipelineResult
- ✅ FR-40: CI/CD integration with re-recording capability
- ✅ NFR-13: Deterministic test execution
- ✅ NFR-14: CI workflow integration
- ✅ NFR-15: Security scrubbing implemented

The implementation provides a robust regression testing foundation for the flujo framework's core orchestration behavior.
