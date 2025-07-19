# FSD-10: Comprehensive Golden Transcript Test Suite

## Overview

This specification defines a comprehensive suite of golden transcript tests for the Flujo framework, designed to provide maximum confidence and feedback with minimum maintenance overhead. Instead of one monolithic test, we implement a small suite of focused tests, each with a clear purpose aligned with distinct architectural layers.

## Motivation

The previous approach of a single monolithic golden transcript test was fragile and hard to debug. This new strategy provides:

1. **Isolation and Debuggability**: If a test fails, you know exactly which part of the framework is broken
2. **Clarity and Maintainability**: Each test file has a clear, single purpose
3. **Architectural Alignment**: The test structure mirrors the framework architecture
4. **Robustness**: Each test can be perfectly deterministic for its specific purpose

## Test Suite Structure

### 1. Core Orchestration Transcript (`test_golden_transcript_core.py`)

**Purpose**: Lock in the behavior of the fundamental, low-level control flow primitives and their interactions with context, resources, and resilience features.

**Pipeline Content**: The "Definitive Golden Pipeline" that tests:
- `Step.loop_until`
- `Step.branch_on`
- `Step.parallel`
- `Step.fallback` and `StepConfig(max_retries=...)`
- `PipelineContext` modification and propagation
- `AppResources` injection

**What it Guarantees**: That the fundamental building blocks of any custom pipeline work as expected.

### 2. Agentic Loop Recipe Transcript (`test_golden_transcript_agentic_loop.py`)

**Purpose**: Lock in the behavior of the most important high-level recipe, `make_agentic_loop_pipeline`.

**Pipeline Content**:
- A pipeline created with `make_agentic_loop_pipeline`
- A deterministic `StubAgent` for the planner that emits a sequence of commands
- A simple `agent_registry` with `StubAgent`s as tools
- Tests for final state, command log, and resume functionality

**What it Guarantees**: That the user-facing `AgenticLoop` recipe works as documented.

### 3. Refinement Loop Recipe Transcript (`test_golden_transcript_refine.py`)

**Purpose**: Lock in the behavior of the `Step.refine_until` recipe.

**Pipeline Content**:
- A `Step.refine_until` step
- A deterministic `generator_pipeline` using a `StubAgent`
- A deterministic `critic_pipeline` using a `StubAgent` that returns `RefinementCheck` objects
- Tests for loop termination and final output

**What it Guarantees**: That the generator-critic pattern works as expected.

### 4. Dynamic Parallel Router Transcript (`test_golden_transcript_dynamic_parallel.py`)

**Purpose**: Test the `Step.dynamic_parallel_branch` primitive.

**Pipeline Content**:
- A `Step.dynamic_parallel_branch` step
- A deterministic `router_agent` that returns a list of branch names
- Tests for selective branch execution and result aggregation

**What it Guarantees**: That the runtime branch selection and execution logic works correctly.

## Implementation Requirements

### Core Orchestration Test

```python
# tests/e2e/test_golden_transcript_core.py
@pytest.mark.asyncio
async def test_golden_transcript_core():
    """Test the core orchestration primitives with deterministic behavior."""
    # Uses the definitive golden pipeline
    # Tests both branches A and B
    # Verifies all fundamental primitives work correctly
```

### Agentic Loop Test

```python
# tests/e2e/test_golden_transcript_agentic_loop.py
@pytest.mark.asyncio
async def test_golden_transcript_agentic_loop():
    """Test the agentic loop recipe with deterministic behavior."""
    # Uses make_agentic_loop_pipeline
    # Tests command execution and state management
    # Verifies resume functionality
```

### Refinement Loop Test

```python
# tests/e2e/test_golden_transcript_refine.py
@pytest.mark.asyncio
async def test_golden_transcript_refine():
    """Test the refinement loop recipe with deterministic behavior."""
    # Uses Step.refine_until
    # Tests generator-critic feedback flow
    # Verifies termination conditions
```

### Dynamic Parallel Test

```python
# tests/e2e/test_golden_transcript_dynamic_parallel.py
@pytest.mark.asyncio
async def test_golden_transcript_dynamic_parallel():
    """Test the dynamic parallel router with deterministic behavior."""
    # Uses Step.dynamic_parallel_branch
    # Tests runtime branch selection
    # Verifies result aggregation
```

## Quality Requirements

### Determinism

- All tests must be perfectly deterministic
- No manual state setting or forcing of expected values
- Framework must naturally produce the expected state

### Isolation

- Each test focuses on a specific architectural layer
- Clear separation between core primitives and high-level recipes
- Independent test execution without shared state

### Maintainability

- Simple, focused test pipelines
- Clear assertions that verify specific behaviors
- Comprehensive documentation of test purposes

### Robustness

- Tests should catch regressions in their specific domain
- Clear error messages when tests fail
- Easy debugging and root cause analysis

## Success Criteria

1. **All tests pass consistently**: No flaky tests or intermittent failures
2. **Clear failure isolation**: When a test fails, it's obvious which framework component is broken
3. **Comprehensive coverage**: All major framework features are tested
4. **Fast execution**: Tests run quickly and efficiently
5. **Easy maintenance**: Tests are simple to understand and modify

## Migration Plan

1. **Phase 1**: Implement `test_golden_transcript_core.py` (highest priority)
2. **Phase 2**: Implement `test_golden_transcript_agentic_loop.py`
3. **Phase 3**: Implement `test_golden_transcript_refine.py`
4. **Phase 4**: Implement `test_golden_transcript_dynamic_parallel.py`
5. **Phase 5**: Update CI/CD to run all golden transcript tests
6. **Phase 6**: Deprecate the old monolithic test

## Benefits

This approach provides:

- **Maximum confidence**: Each test is focused and deterministic
- **Minimum maintenance**: Simple, clear test structure
- **Professional quality**: Aligns with industry best practices
- **Scalable architecture**: Easy to add new tests as framework evolves

The golden transcript test suite will serve as the foundation for ensuring Flujo framework reliability and correctness.
