"""Unit tests for parallel step execution strategies."""

import asyncio
import logging
from typing import Any, Dict

import pytest

from flujo.application.core.executor_core import ExecutorCore
from flujo.domain.models import BaseModel
from flujo.domain.dsl.parallel import ParallelStep
from flujo.domain.dsl.step import Step, BranchFailureStrategy, MergeStrategy
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.models import (
    StepResult,
    UsageLimits,
)
from flujo.exceptions import UsageLimitExceededError
from flujo.infra.monitor import global_monitor
from flujo.testing.utils import StubAgent


class MockContext(BaseModel):
    """Mock context for testing, inherits from BaseModel for proper validation."""

    # Required PipelineContext attributes
    initial_prompt: str = "test_prompt"
    scratchpad: Dict[str, Any] = {}

    # Additional data storage for tests that expect it
    data: Dict[str, Any] = {}

    model_config = {"extra": "allow", "arbitrary_types_allowed": True}

    def __init__(self, data: Dict[str, Any] = None, **kwargs):
        # Merge data and kwargs
        merged_data = dict(data) if data is not None else {}
        merged_data.update(kwargs)

        # Extract known fields
        initial_prompt = merged_data.pop("initial_prompt", "test_prompt")
        scratchpad = merged_data.pop("scratchpad", {})

        # Store original data for tests that expect it
        data_field = dict(data) if data is not None else {}
        data_field.update(kwargs)

        # Initialize BaseModel with all fields
        super().__init__(
            initial_prompt=initial_prompt, scratchpad=scratchpad, data=data_field, **merged_data
        )

    @classmethod
    def model_validate(cls, data: Dict[str, Any]):
        return cls(data)


class TestParallelStepExecution:
    """Test parallel step execution with different strategies."""

    @pytest.fixture
    def mock_step_executor(self):
        """Create a mock step executor."""

        async def executor(step, input_data, context, resources, breach_event=None):
            # Simulate successful step execution
            # Handle both Step and Pipeline objects
            if hasattr(step, "name"):
                step_name = step.name
            else:
                # For Pipeline objects, use a default name
                step_name = "pipeline_step"

            return StepResult(
                name=step_name,
                output=input_data,  # Return input_data as expected by tests
                success=True,
                attempts=1,
                latency_s=0.1,
                token_counts=10,
                cost_usd=0.01,
            )

        return executor

    @pytest.fixture
    def mock_context_setter(self):
        """Create a mock context setter."""

        def setter(result, context):
            pass

        return setter

    @pytest.fixture
    def parallel_step(self):
        """Create a basic parallel step."""
        return ParallelStep(
            name="test_parallel",
            branches={
                "branch1": Pipeline(steps=[Step(name="step1", agent=StubAgent(["output1"]))]),
                "branch2": Pipeline(steps=[Step(name="step2", agent=StubAgent(["output2"]))]),
            },
            merge_strategy=MergeStrategy.NO_MERGE,
            on_branch_failure=BranchFailureStrategy.PROPAGATE,
        )

    @pytest.mark.asyncio
    async def test_basic_parallel_execution_no_merge(
        self, parallel_step, mock_step_executor, mock_context_setter
    ):
        """Test basic parallel execution with NO_MERGE strategy."""
        context = MockContext({"key": "value"})

        executor = ExecutorCore()
        result = await executor._handle_parallel_step(
            parallel_step=parallel_step,
            data="test_input",
            context=context,
            resources=None,
            limits=None,
            breach_event=asyncio.Event(),
            context_setter=mock_context_setter,
        )

        assert result.success
        assert result.name == "test_parallel"
        assert isinstance(result.output, dict)
        assert "branch1" in result.output
        assert "branch2" in result.output

    @pytest.mark.asyncio
    async def test_parallel_execution_with_context_include_keys(
        self, mock_step_executor, mock_context_setter
    ):
        """Test parallel execution with context include keys."""
        parallel_step = ParallelStep(
            name="test_parallel",
            branches={
                "branch1": Pipeline(steps=[Step(name="step1", agent=StubAgent(["output1"]))])
            },
            merge_strategy=MergeStrategy.NO_MERGE,
            on_branch_failure=BranchFailureStrategy.PROPAGATE,
            context_include_keys=["key1", "key2"],
        )

        context = MockContext({"key1": "value1", "key2": "value2", "key3": "value3"})

        executor = ExecutorCore()
        result = await executor._handle_parallel_step(
            parallel_step=parallel_step,
            data="test_input",
            context=context,
            resources=None,
            limits=None,
            breach_event=asyncio.Event(),
            context_setter=mock_context_setter,
        )

        assert result.success

    @pytest.mark.asyncio
    async def test_parallel_execution_no_context(
        self, parallel_step, mock_step_executor, mock_context_setter
    ):
        """Test parallel execution without context."""
        executor = ExecutorCore()
        result = await executor._handle_parallel_step(
            parallel_step=parallel_step,
            data="test_input",
            context=None,
            resources=None,
            limits=None,
            breach_event=asyncio.Event(),
            context_setter=mock_context_setter,
        )

        assert result.success

    @pytest.mark.asyncio
    async def test_parallel_execution_with_usage_limits(
        self, parallel_step, mock_step_executor, mock_context_setter
    ):
        """Test parallel execution with usage limits."""
        usage_limits = UsageLimits(total_cost_usd_limit=0.05, total_tokens_limit=50)

        executor = ExecutorCore()
        result = await executor._handle_parallel_step(
            parallel_step=parallel_step,
            data="test_input",
            context=None,
            resources=None,
            limits=usage_limits,
            breach_event=asyncio.Event(),
            context_setter=mock_context_setter,
        )

        assert result.success

    @pytest.mark.asyncio
    async def test_parallel_execution_cost_limit_breach(
        self, mock_step_executor, mock_context_setter
    ):
        """Test parallel execution with cost limit breach."""

        # Create a step executor that returns high cost
        async def high_cost_executor(step, input_data, context, resources, breach_event=None):
            return StepResult(
                name=step.name if hasattr(step, "name") else "test",
                success=True,
                output=input_data,
                latency_s=0.1,
                cost_usd=1.0,  # High cost
                token_counts=10,
                attempts=1,
            )

        parallel_step = ParallelStep(
            name="test_parallel",
            branches={
                "branch1": Pipeline(steps=[Step(name="step1", agent=StubAgent(["output1"]))])
            },
            merge_strategy=MergeStrategy.NO_MERGE,
            on_branch_failure=BranchFailureStrategy.PROPAGATE,
        )

        usage_limits = UsageLimits(total_cost_usd_limit=0.5)  # Low limit

        with pytest.raises(UsageLimitExceededError):
            executor = ExecutorCore()
            await executor._handle_parallel_step(
                parallel_step=parallel_step,
                data="test_input",
                context=None,
                resources=None,
                limits=usage_limits,
                breach_event=asyncio.Event(),
                context_setter=mock_context_setter,
                step_executor=high_cost_executor,
            )

    @pytest.mark.asyncio
    async def test_parallel_execution_token_limit_breach(
        self, mock_step_executor, mock_context_setter
    ):
        """Test parallel execution with token limit breach."""

        # Create a step executor that returns high token count
        async def high_token_executor(step, input_data, context, resources, breach_event=None):
            return StepResult(
                name=step.name if hasattr(step, "name") else "test",
                success=True,
                output=input_data,
                latency_s=0.1,
                cost_usd=0.01,
                token_counts=100,  # High token count
                attempts=1,
            )

        parallel_step = ParallelStep(
            name="test_parallel",
            branches={
                "branch1": Pipeline(steps=[Step(name="step1", agent=StubAgent(["output1"]))])
            },
            merge_strategy=MergeStrategy.NO_MERGE,
            on_branch_failure=BranchFailureStrategy.PROPAGATE,
        )

        usage_limits = UsageLimits(total_tokens_limit=50)  # Low limit

        with pytest.raises(UsageLimitExceededError):
            executor = ExecutorCore()
            await executor._handle_parallel_step(
                parallel_step=parallel_step,
                data="test_input",
                context=None,
                resources=None,
                limits=usage_limits,
                breach_event=asyncio.Event(),
                context_setter=mock_context_setter,
                step_executor=high_token_executor,
            )

    @pytest.mark.asyncio
    async def test_parallel_execution_branch_failure_propagate(self, mock_context_setter):
        """Test parallel execution with branch failure and PROPAGATE strategy."""

        # Create a step executor that fails
        async def failing_executor(step, input_data, context, resources, breach_event=None):
            return StepResult(
                name=step.name if hasattr(step, "name") else "test",
                success=False,
                output=None,
                feedback="Test failure",
                latency_s=0.1,
                cost_usd=0.01,
                token_counts=10,
                attempts=1,
            )

        parallel_step = ParallelStep(
            name="test_parallel",
            branches={
                "branch1": Pipeline(steps=[Step(name="step1", agent=StubAgent(["output1"]))]),
                "branch2": Pipeline(steps=[Step(name="step2", agent=StubAgent(["output2"]))]),
            },
            merge_strategy=MergeStrategy.NO_MERGE,
            on_branch_failure=BranchFailureStrategy.PROPAGATE,
        )

        executor = ExecutorCore()
        result = await executor._handle_parallel_step(
            parallel_step=parallel_step,
            data="test_input",
            context=None,
            resources=None,
            limits=None,
            breach_event=asyncio.Event(),
            context_setter=mock_context_setter,
            step_executor=failing_executor,
        )

        # Verify that the result indicates failure
        assert not result.success
        assert "failed" in result.feedback.lower()

    @pytest.mark.asyncio
    async def test_parallel_execution_branch_failure_ignore(
        self, mock_step_executor, mock_context_setter
    ):
        """Test parallel execution with branch failure and IGNORE strategy."""
        # Create a step executor that fails for one branch
        branch_results = {"branch1": True, "branch2": False}

        async def conditional_failing_executor(
            step, input_data, context, resources, breach_event=None
        ):
            step_name = step.name if hasattr(step, "name") else "test"
            success = branch_results.get(step_name, True)
            return StepResult(
                name=step_name,
                success=success,
                output=input_data if success else None,
                feedback="Test failure" if not success else None,
                latency_s=0.1,
                cost_usd=0.01,
                token_counts=10,
                attempts=1,
            )

        parallel_step = ParallelStep(
            name="test_parallel",
            branches={
                "branch1": Pipeline(steps=[Step(name="branch1", agent=StubAgent(["output1"]))]),
                "branch2": Pipeline(steps=[Step(name="branch2", agent=StubAgent(["output2"]))]),
            },
            merge_strategy=MergeStrategy.NO_MERGE,
            on_branch_failure=BranchFailureStrategy.IGNORE,
        )

        executor = ExecutorCore()
        result = await executor._handle_parallel_step(
            parallel_step=parallel_step,
            data="test_input",
            context=None,
            resources=None,
            limits=None,
            breach_event=asyncio.Event(),
            context_setter=mock_context_setter,
        )

        # Should succeed even with branch failure
        assert result.success

    @pytest.mark.asyncio
    async def test_parallel_execution_merge_overwrite(
        self, mock_step_executor, mock_context_setter
    ):
        """Test parallel execution with OVERWRITE merge strategy."""
        parallel_step = ParallelStep(
            name="test_parallel",
            branches={
                "branch1": Pipeline(steps=[Step(name="step1", agent=StubAgent(["output1"]))])
            },
            merge_strategy=MergeStrategy.OVERWRITE,
            on_branch_failure=BranchFailureStrategy.PROPAGATE,
        )
        # Provide a scratchpad in the context data
        context = MockContext({"key": "value", "scratchpad": {}})
        executor = ExecutorCore()
        result = await executor._handle_parallel_step(
            parallel_step=parallel_step,
            data="test_input",
            context=context,
            resources=None,
            limits=None,
            breach_event=asyncio.Event(),
            context_setter=mock_context_setter,
        )
        assert result.success

    @pytest.mark.asyncio
    async def test_parallel_execution_merge_scratchpad(
        self, mock_step_executor, mock_context_setter
    ):
        """Test parallel execution with MERGE_SCRATCHPAD strategy."""
        parallel_step = ParallelStep(
            name="test_parallel",
            branches={
                "branch1": Pipeline(steps=[Step(name="step1", agent=StubAgent(["output1"]))])
            },
            merge_strategy=MergeStrategy.MERGE_SCRATCHPAD,
            on_branch_failure=BranchFailureStrategy.PROPAGATE,
        )

        context = MockContext({"key": "value"})
        context.scratchpad = {"existing": "data"}

        executor = ExecutorCore()
        result = await executor._handle_parallel_step(
            parallel_step=parallel_step,
            data="test_input",
            context=context,
            resources=None,
            limits=None,
            breach_event=asyncio.Event(),
            context_setter=mock_context_setter,
        )

        assert result.success

    @pytest.mark.asyncio
    async def test_parallel_execution_merge_scratchpad_no_scratchpad(
        self, mock_step_executor, mock_context_setter
    ):
        """Test parallel execution with MERGE_SCRATCHPAD strategy on context without scratchpad."""
        parallel_step = ParallelStep(
            name="test_parallel",
            branches={
                "branch1": Pipeline(steps=[Step(name="step1", agent=StubAgent(["output1"]))])
            },
            merge_strategy=MergeStrategy.MERGE_SCRATCHPAD,
            on_branch_failure=BranchFailureStrategy.PROPAGATE,
        )

        # Use a context with no scratchpad attribute at all
        class NoScratchpadContext(BaseModel):
            initial_prompt: str = "test_prompt"
            data: Dict[str, Any] = {}

            model_config = {"extra": "allow", "arbitrary_types_allowed": True}

            def __init__(self, data: Dict[str, Any] = None, **kwargs):
                merged_data = dict(data) if data is not None else {}
                merged_data.update(kwargs)

                initial_prompt = merged_data.pop("initial_prompt", "test_prompt")

                super().__init__(initial_prompt=initial_prompt, data=merged_data, **merged_data)
                # Note: deliberately not setting scratchpad - the framework should create it

            @classmethod
            def model_validate(cls, data):
                return cls(data)

        context = NoScratchpadContext({"key": "value"})

        # The improved framework should create a scratchpad if it doesn't exist
        executor = ExecutorCore()
        result = await executor._handle_parallel_step(
            parallel_step=parallel_step,
            data="test_input",
            context=context,
            resources=None,
            limits=None,
            breach_event=asyncio.Event(),
            context_setter=mock_context_setter,
        )

        # Verify the pipeline completed successfully
        assert result.success

        # Verify that a scratchpad was created on the context
        assert hasattr(context, "scratchpad")
        assert isinstance(context.scratchpad, dict)

    @pytest.mark.asyncio
    async def test_parallel_execution_custom_merge_strategy(
        self, mock_step_executor, mock_context_setter
    ):
        """Test parallel execution with custom merge strategy."""

        def custom_merge_strategy(context, branch_context):
            context.data["merged"] = True

        parallel_step = ParallelStep(
            name="test_parallel",
            branches={
                "branch1": Pipeline(steps=[Step(name="step1", agent=StubAgent(["output1"]))])
            },
            merge_strategy=custom_merge_strategy,
            on_branch_failure=BranchFailureStrategy.PROPAGATE,
        )

        context = MockContext({"key": "value"})

        executor = ExecutorCore()
        result = await executor._handle_parallel_step(
            parallel_step=parallel_step,
            data="test_input",
            context=context,
            resources=None,
            limits=None,
            breach_event=asyncio.Event(),
            context_setter=mock_context_setter,
        )

        assert result.success
        assert context.data["merged"] is True

    @pytest.mark.asyncio
    async def test_parallel_execution_exception_handling(self, mock_context_setter):
        """Test parallel execution with exception handling."""

        # Create a step executor that raises exceptions
        async def exception_executor(step, input_data, context, resources, breach_event=None):
            raise ValueError("Test exception")

        parallel_step = ParallelStep(
            name="test_parallel",
            branches={
                "branch1": Pipeline(steps=[Step(name="step1", agent=StubAgent(["output1"]))]),
                "branch2": Pipeline(steps=[Step(name="step2", agent=StubAgent(["output2"]))]),
            },
            merge_strategy=MergeStrategy.NO_MERGE,
            on_branch_failure=BranchFailureStrategy.PROPAGATE,
        )

        executor = ExecutorCore()
        result = await executor._handle_parallel_step(
            parallel_step=parallel_step,
            data="test_input",
            context=None,
            resources=None,
            limits=None,
            breach_event=asyncio.Event(),
            context_setter=mock_context_setter,
            step_executor=exception_executor,
        )

        # Verify that the result indicates failure
        assert not result.success
        assert "failed" in result.feedback.lower()

    @pytest.mark.asyncio
    async def test_parallel_execution_task_cancellation(self, mock_context_setter):
        """Test parallel execution with task cancellation."""

        # Create a step executor that takes time
        async def slow_executor(step, input_data, context, resources, breach_event=None):
            # Simulate slow execution
            await asyncio.sleep(0.1)
            return StepResult(
                name=step.name if hasattr(step, "name") else "test",
                success=True,
                output=input_data,
                latency_s=0.1,
                cost_usd=0.01,
                token_counts=10,
                attempts=1,
            )

        parallel_step = ParallelStep(
            name="test_parallel",
            branches={
                "branch1": Pipeline(steps=[Step(name="step1", agent=StubAgent(["output1"]))])
            },
            merge_strategy=MergeStrategy.NO_MERGE,
            on_branch_failure=BranchFailureStrategy.PROPAGATE,
        )

        usage_limits = UsageLimits(total_cost_usd_limit=0.001)  # Very low limit

        with pytest.raises(UsageLimitExceededError):
            executor = ExecutorCore()
            await executor._handle_parallel_step(
                parallel_step=parallel_step,
                data="test_input",
                context=None,
                resources=None,
                limits=usage_limits,
                breach_event=asyncio.Event(),
                context_setter=mock_context_setter,
                step_executor=slow_executor,
            )

    @pytest.mark.asyncio
    async def test_parallel_execution_merge_context_update(
        self, mock_step_executor, mock_context_setter
    ):
        """Test parallel execution with context update merge strategy."""

        # Create a context that can be updated
        class TestContext(BaseModel):
            initial_prompt: str = "test_prompt"
            value: str = "initial"
            scratchpad: Dict[str, Any] = {}

            model_config = {"extra": "allow", "arbitrary_types_allowed": True}

            def __init__(self, value: str = "initial", **kwargs):
                super().__init__(initial_prompt="test_prompt", value=value, scratchpad={}, **kwargs)

            @classmethod
            def model_validate(cls, data):
                if isinstance(data, dict):
                    return cls(value=data.get("value", "initial"))
                return cls(value=str(data))

        initial_context = TestContext("initial")

        parallel_step = ParallelStep(
            name="test_parallel",
            branches={
                "branch1": Pipeline(steps=[Step(name="step1", agent=StubAgent(["output1"]))]),
                "branch2": Pipeline(steps=[Step(name="step2", agent=StubAgent(["output2"]))]),
            },
            merge_strategy=MergeStrategy.CONTEXT_UPDATE,
            on_branch_failure=BranchFailureStrategy.PROPAGATE,
        )

        executor = ExecutorCore()
        result = await executor._handle_parallel_step(
            parallel_step=parallel_step,
            data="test_input",
            context=initial_context,
            resources=None,
            limits=None,
            breach_event=asyncio.Event(),
            context_setter=mock_context_setter,
            step_executor=mock_step_executor,
        )

        # Verify that the result is successful
        assert result.success
        # The output should contain the branch outputs
        assert "branch1" in result.output
        assert "branch2" in result.output


@pytest.mark.asyncio
async def test_parallel_usage_limit_enforced_atomically(caplog):
    """Test that usage limits are enforced atomically under true concurrency in parallel steps."""
    caplog.set_level(logging.INFO)
    global_monitor.calls.clear()

    # Create a step that returns a cost/token increment after a small delay
    class CostlyAgent:
        def __init__(self, cost, tokens, delay=0.05):
            self.cost = cost
            self.tokens = tokens
            self.delay = delay
            self.called = False

        async def run(self, *_args, breach_event=None, **_kwargs):
            self.called = True
            # Simulate a longer operation with periodic cancellation checks
            for _ in range(50):  # More iterations for more frequent checks
                if breach_event is not None and breach_event.is_set():
                    # Simulate early exit on cancellation
                    return StepResult(
                        name="costly_step",
                        output="cancelled",
                        success=False,
                        latency_s=self.delay,
                        cost_usd=0.0,
                        token_counts=0,
                        attempts=1,
                        feedback="Cancelled early due to usage limit breach",
                    )
                await asyncio.sleep(self.delay / 50)  # Shorter sleep for more frequent checks
            return StepResult(
                name="costly_step",
                output="ok",
                success=True,
                latency_s=self.delay,
                cost_usd=self.cost,
                token_counts=self.tokens,
                attempts=1,
            )

    # Set up N branches, each with a cost that will cumulatively breach the limit
    N = 5
    usage_limits = UsageLimits(total_cost_usd_limit=1.0, total_tokens_limit=250)
    # Calculate values that will definitely exceed the limit
    cost_per_branch = (usage_limits.total_cost_usd_limit / N) + 0.01  # Ensure we exceed the limit
    token_per_branch = (usage_limits.total_tokens_limit // N) + 1  # Ensure we exceed the limit
    # Create agents with different delays to simulate staggered execution
    delays = [0.01, 0.02, 0.03, 0.04, 0.05]  # Different delays for each agent
    agents = [CostlyAgent(cost_per_branch, token_per_branch, delay=delays[i]) for i in range(N)]
    branches = {f"b{i}": Pipeline(steps=[Step(name=f"s{i}", agent=agents[i])]) for i in range(N)}
    parallel_step = ParallelStep(
        name="test_parallel_race",
        branches=branches,
        merge_strategy=MergeStrategy.NO_MERGE,
        on_branch_failure=BranchFailureStrategy.PROPAGATE,
    )

    # Robust step_executor that forwards breach_event
    async def step_executor(step, input_data, context, resources, breach_event=None):
        return await step.agent.run(input_data, breach_event=breach_event)

    # Run the parallel step logic and expect UsageLimitExceededError
    from flujo.application.core.executor_core import ExecutorCore

    # Execute the parallel step
    breach_event = asyncio.Event()
    with pytest.raises(UsageLimitExceededError):
        executor = ExecutorCore()
        await executor._handle_parallel_step(
            parallel_step=parallel_step,
            data="test_input",
            context=None,
            resources=None,
            limits=usage_limits,
            breach_event=breach_event,
            context_setter=lambda result, context: None,
        )

    # All agents should have been called (they all start before the breach is detected)
    called_count = sum(a.called for a in agents)
    assert called_count == N, f"Expected all branches to start, got {called_count}/{N}"

    # The UsageLimitExceededError should have been raised
    # This verifies that the breach detection is working correctly
