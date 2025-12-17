"""Unit tests for ExecutionManager state handling (FSD 3 of 4)."""

import pytest
from unittest.mock import Mock
from typing import Any

from flujo.application.core.execution_manager import ExecutionManager
from flujo.application.core.step_coordinator import StepCoordinator
from flujo.application.core.executor_core import ExecutorCore
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.step import Step
from flujo.domain.models import (
    PipelineResult,
    StepResult,
    UsageLimits,
    BaseModel,
)
from flujo.exceptions import UsageLimitExceededError
from flujo.infra.backends import LocalBackend

# Avoid xdist worker crashes observed in CI by running tests in this module serially
pytestmark = pytest.mark.serial


class MockContext(BaseModel):
    """Mock context for testing."""

    value: str = "test"


class MockStep(Step):
    """Mock step for testing."""

    def __init__(self, name: str, success: bool = True, cost: float = 0.1, tokens: int = 10):
        super().__init__(name=name)
        # Store test-specific attributes in a way that doesn't conflict with Step model
        self._test_success = success
        self._test_cost = cost
        self._test_tokens = tokens


class SequencedAgentExecutor:
    """Agent executor override that returns a predefined StepResult sequence."""

    def __init__(self, step_results: list[StepResult]) -> None:
        self._step_results = step_results
        self._idx = 0

    async def execute(
        self,
        _core: object,
        _step: Any,
        _data: Any,
        _context: Any,
        _resources: Any,
        _limits: Any,
        _stream: bool,
        _on_chunk: Any,
        _cache_key: Any,
        _fallback_depth: int,
    ) -> StepResult:
        if self._idx >= len(self._step_results):
            raise AssertionError("SequencedAgentExecutor exhausted")
        sr = self._step_results[self._idx]
        self._idx += 1
        return sr


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline with test steps."""
    steps = [
        MockStep("step1", success=True, cost=0.1, tokens=10),
        MockStep("step2", success=True, cost=0.2, tokens=20),
        MockStep("step3", success=True, cost=0.3, tokens=30),
    ]
    return Mock(spec=Pipeline, steps=steps)


@pytest.fixture
def usage_limits():
    """Create usage limits for testing."""
    return UsageLimits(total_cost_usd_limit=0.5, total_tokens_limit=50)


@pytest.fixture
def execution_manager(mock_pipeline):
    """Create an ExecutionManager instance for testing."""
    return ExecutionManager(pipeline=mock_pipeline)


@pytest.mark.asyncio
async def test_execution_manager_updates_state_before_usage_check():
    """Test that ExecutionManager updates state before checking usage limits."""
    # Create a pipeline with steps that will exceed limits
    steps = [
        MockStep("step1", success=True, cost=0.3, tokens=25),  # Will exceed cost limit
        MockStep("step2", success=True, cost=0.2, tokens=20),
    ]
    pipeline = Mock(spec=Pipeline, steps=steps)

    # Create step results that will trigger usage limit breach
    step_results = [
        StepResult(
            name="step1",
            output="result1",
            success=True,
            cost_usd=0.3,
            token_counts=25,
        ),
        StepResult(
            name="step2",
            output="result2",
            success=True,
            cost_usd=0.2,
            token_counts=20,
        ),
    ]

    # Create usage limits that will be exceeded
    usage_limits = UsageLimits(total_cost_usd_limit=0.4, total_tokens_limit=40)

    # Create execution manager using quota-based limits
    core = ExecutorCore()
    core.agent_step_executor = SequencedAgentExecutor(step_results)
    execution_manager = ExecutionManager(
        pipeline=pipeline,
        backend=LocalBackend(core),
        usage_limits=usage_limits,
    )

    # Create initial pipeline result
    result = PipelineResult(
        step_history=[],
        total_cost_usd=0.0,
    )

    # Execute steps - this should raise UsageLimitExceededError
    with pytest.raises(UsageLimitExceededError) as exc_info:
        async for _ in execution_manager.execute_steps(
            start_idx=0,
            data="test_input",
            context=None,
            result=result,
        ):
            pass

    # Verify that the exception contains the correct step history
    error = exc_info.value
    assert error.result is not None
    print(f"DEBUG: Step history length: {len(error.result.step_history)}")
    print(f"DEBUG: Step history: {[s.name for s in error.result.step_history]}")
    print(f"DEBUG: Total cost: {error.result.total_cost_usd}")

    # The breaching step should be in the history
    assert len(error.result.step_history) == 2  # Both steps should be present
    # The first step should be the one before the breach, the second is the breaching step
    assert error.result.step_history[0].name == "step1"
    assert error.result.step_history[1].name == "step2"
    assert error.result.step_history[0].cost_usd == 0.3
    assert error.result.step_history[1].cost_usd == 0.2
    assert error.result.step_history[0].token_counts == 25
    assert error.result.step_history[1].token_counts == 20
    # Use approximate comparison for floating point
    assert abs(error.result.total_cost_usd - 0.5) < 1e-10


@pytest.mark.asyncio
async def test_execution_manager_handles_multiple_steps_before_breach():
    """Test that ExecutionManager correctly handles multiple steps before a breach."""
    # Create a pipeline with steps that will eventually exceed limits
    steps = [
        MockStep("step1", success=True, cost=0.1, tokens=10),
        MockStep("step2", success=True, cost=0.2, tokens=15),  # This will breach
        MockStep("step3", success=True, cost=0.1, tokens=10),
    ]
    pipeline = Mock(spec=Pipeline, steps=steps)

    # Create step results
    step_results = [
        StepResult(
            name="step1",
            output="result1",
            success=True,
            cost_usd=0.1,
            token_counts=10,
        ),
        StepResult(
            name="step2",
            output="result2",
            success=True,
            cost_usd=0.2,
            token_counts=15,  # This will breach the token limit
        ),
        StepResult(
            name="step3",
            output="result3",
            success=True,
            cost_usd=0.1,
            token_counts=10,
        ),
    ]

    # Create usage limits that will be exceeded by step2
    usage_limits = UsageLimits(total_cost_usd_limit=1.0, total_tokens_limit=20)

    # Create execution manager using quota-based limits
    core = ExecutorCore()
    core.agent_step_executor = SequencedAgentExecutor(step_results)
    execution_manager = ExecutionManager(
        pipeline=pipeline,
        backend=LocalBackend(core),
        usage_limits=usage_limits,
    )

    # Create initial pipeline result
    result = PipelineResult(
        step_history=[],
        total_cost_usd=0.0,
    )

    # Execute steps - this should raise UsageLimitExceededError
    with pytest.raises(UsageLimitExceededError) as exc_info:
        async for _ in execution_manager.execute_steps(
            start_idx=0,
            data="test_input",
            context=None,
            result=result,
        ):
            pass

    # Verify that the exception contains the correct step history
    error = exc_info.value
    assert error.result is not None
    # Should contain steps up to the breaching step
    assert len(error.result.step_history) >= 2
    assert error.result.step_history[0].name == "step1"
    assert error.result.step_history[1].name == "step2"  # The breaching step
    # Use approximate comparison for floating point
    assert abs(error.result.total_cost_usd - 0.3) < 1e-10


@pytest.mark.asyncio
async def test_execution_manager_no_usage_limits():
    """Test that ExecutionManager works correctly when no usage limits are configured."""
    # Create a pipeline with steps
    steps = [
        MockStep("step1", success=True, cost=0.1, tokens=10),
        MockStep("step2", success=True, cost=0.2, tokens=20),
    ]
    pipeline = Mock(spec=Pipeline, steps=steps)

    # Create step results
    step_results = [
        StepResult(
            name="step1",
            output="result1",
            success=True,
            cost_usd=0.1,
            token_counts=10,
        ),
        StepResult(
            name="step2",
            output="result2",
            success=True,
            cost_usd=0.2,
            token_counts=20,
        ),
    ]

    # Create execution manager without usage limits
    core = ExecutorCore()
    core.agent_step_executor = SequencedAgentExecutor(step_results)
    execution_manager = ExecutionManager(pipeline=pipeline, backend=LocalBackend(core))

    # Create initial pipeline result
    result = PipelineResult(
        step_history=[],
        total_cost_usd=0.0,
    )

    # Execute steps - this should complete successfully
    step_count = 0
    async for _ in execution_manager.execute_steps(
        start_idx=0,
        data="test_input",
        context=None,
        result=result,
    ):
        step_count += 1

    # Verify that all steps were executed and added to history
    assert len(result.step_history) == 2
    # Use approximate comparison for floating point
    assert abs(result.total_cost_usd - 0.3) < 1e-10


@pytest.mark.asyncio
async def test_execution_manager_failed_step_handling():
    """Test that ExecutionManager correctly handles failed steps."""
    # Create a pipeline with a step that fails
    steps = [
        MockStep("step1", success=False, cost=0.1, tokens=10),
    ]
    pipeline = Mock(spec=Pipeline, steps=steps)

    # Create step result for failed step
    step_results = [
        StepResult(
            name="step1",
            output=None,
            success=False,
            cost_usd=0.1,
            token_counts=10,
            feedback="Step failed",
        ),
    ]

    # Create execution manager
    core = ExecutorCore()
    core.agent_step_executor = SequencedAgentExecutor(step_results)
    execution_manager = ExecutionManager(pipeline=pipeline, backend=LocalBackend(core))

    # Create initial pipeline result
    result = PipelineResult(
        step_history=[],
        total_cost_usd=0.0,
    )

    # Execute steps - this should complete after the failed step
    step_count = 0
    async for _ in execution_manager.execute_steps(
        start_idx=0,
        data="test_input",
        context=None,
        result=result,
    ):
        step_count += 1

    # Verify that the failed step was added to history
    assert len(result.step_history) == 1
    assert result.step_history[0].success is False
    assert result.step_history[0].feedback == "Step failed"
    assert result.total_cost_usd == 0.1  # Should still accumulate cost


@pytest.mark.asyncio
async def test_execution_manager_step_coordinator_integration():
    """Test that ExecutionManager correctly uses StepCoordinator for state updates."""
    # Create a pipeline with steps
    steps = [
        MockStep("step1", success=True, cost=0.1, tokens=10),
        MockStep("step2", success=True, cost=0.2, tokens=20),
    ]
    pipeline = Mock(spec=Pipeline, steps=steps)

    # Create step results
    step_results = [
        StepResult(
            name="step1",
            output="result1",
            success=True,
            cost_usd=0.1,
            token_counts=10,
        ),
        StepResult(
            name="step2",
            output="result2",
            success=True,
            cost_usd=0.2,
            token_counts=20,
        ),
    ]

    # Create execution manager with custom step coordinator
    step_coordinator = StepCoordinator()
    core = ExecutorCore()
    core.agent_step_executor = SequencedAgentExecutor(step_results)
    execution_manager = ExecutionManager(
        pipeline=pipeline,
        step_coordinator=step_coordinator,
        backend=LocalBackend(core),
    )

    # Create initial pipeline result
    result = PipelineResult(
        step_history=[],
        total_cost_usd=0.0,
    )

    # Execute steps
    async for _ in execution_manager.execute_steps(
        start_idx=0,
        data="test_input",
        context=None,
        result=result,
    ):
        pass

    # Verify that the step coordinator was used to update the pipeline result
    assert len(result.step_history) == 2
    assert result.step_history[0].name == "step1"
    assert result.step_history[1].name == "step2"
    # Use approximate comparison for floating point
    assert abs(result.total_cost_usd - 0.3) < 1e-10
