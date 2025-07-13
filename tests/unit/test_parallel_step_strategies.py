"""Unit tests for parallel step execution strategies."""

import asyncio
from typing import Any, Dict

import pytest

from flujo.application.core.step_logic import _execute_parallel_step_logic
from flujo.domain.dsl.parallel import ParallelStep
from flujo.domain.dsl.step import Step
from flujo.domain.models import (
    StepResult,
    UsageLimits,
)
from flujo.domain.dsl.step import BranchFailureStrategy, MergeStrategy
from flujo.domain.dsl.pipeline import Pipeline
from flujo.testing.utils import StubAgent
from flujo.exceptions import UsageLimitExceededError


class MockContext:
    """Mock context for testing, mimics a Pydantic model with flexible construction and attribute access."""

    def __init__(self, data: Dict[str, Any] = None, **kwargs):
        # Accept both dict and kwargs for flexible construction
        self.__dict__["data"] = dict(data) if data is not None else {}
        self.__dict__["data"].update(kwargs)
        # Set all keys as attributes
        for k, v in self.__dict__["data"].items():
            setattr(self, k, v)
        # Only set scratchpad if present, else default
        if hasattr(self, "scratchpad"):
            pass
        else:
            self.scratchpad = self.__dict__["data"].get("scratchpad", {})

    def model_dump(self) -> Dict[str, Any]:
        out = self.__dict__["data"].copy()
        if hasattr(self, "scratchpad"):
            out["scratchpad"] = self.scratchpad
        return out

    @classmethod
    def model_validate(cls, data: Dict[str, Any]):
        return cls(data)

    def __getattr__(self, item):
        # Avoid recursion for 'data'
        if item == "data":
            return self.__dict__["data"]
        if "data" in self.__dict__ and item in self.__dict__["data"]:
            return self.__dict__["data"][item]
        raise AttributeError(f"MockContext has no attribute '{item}'")


class TestParallelStepExecution:
    """Test parallel step execution with different strategies."""

    @pytest.fixture
    def mock_step_executor(self):
        """Create a mock step executor."""

        async def executor(step, input_data, context, resources):
            return StepResult(
                name=step.name if hasattr(step, "name") else "test",
                success=True,
                output=input_data,
                latency_s=0.1,
                cost_usd=0.01,
                token_counts=10,
                attempts=1,
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

        result = await _execute_parallel_step_logic(
            parallel_step=parallel_step,
            parallel_input="test_input",
            context=context,
            resources=None,
            step_executor=mock_step_executor,
            context_model_defined=True,
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

        result = await _execute_parallel_step_logic(
            parallel_step=parallel_step,
            parallel_input="test_input",
            context=context,
            resources=None,
            step_executor=mock_step_executor,
            context_model_defined=True,
            context_setter=mock_context_setter,
        )

        assert result.success

    @pytest.mark.asyncio
    async def test_parallel_execution_no_context(
        self, parallel_step, mock_step_executor, mock_context_setter
    ):
        """Test parallel execution without context."""
        result = await _execute_parallel_step_logic(
            parallel_step=parallel_step,
            parallel_input="test_input",
            context=None,
            resources=None,
            step_executor=mock_step_executor,
            context_model_defined=True,
            context_setter=mock_context_setter,
        )

        assert result.success

    @pytest.mark.asyncio
    async def test_parallel_execution_with_usage_limits(
        self, parallel_step, mock_step_executor, mock_context_setter
    ):
        """Test parallel execution with usage limits."""
        usage_limits = UsageLimits(total_cost_usd_limit=0.05, total_tokens_limit=50)

        result = await _execute_parallel_step_logic(
            parallel_step=parallel_step,
            parallel_input="test_input",
            context=None,
            resources=None,
            step_executor=mock_step_executor,
            context_model_defined=True,
            usage_limits=usage_limits,
            context_setter=mock_context_setter,
        )

        assert result.success

    @pytest.mark.asyncio
    async def test_parallel_execution_cost_limit_breach(
        self, mock_step_executor, mock_context_setter
    ):
        """Test parallel execution with cost limit breach."""

        # Create a step executor that returns high cost
        async def high_cost_executor(step, input_data, context, resources):
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
            await _execute_parallel_step_logic(
                parallel_step=parallel_step,
                parallel_input="test_input",
                context=None,
                resources=None,
                step_executor=high_cost_executor,
                context_model_defined=True,
                usage_limits=usage_limits,
                context_setter=mock_context_setter,
            )

    @pytest.mark.asyncio
    async def test_parallel_execution_token_limit_breach(
        self, mock_step_executor, mock_context_setter
    ):
        """Test parallel execution with token limit breach."""

        # Create a step executor that returns high token count
        async def high_token_executor(step, input_data, context, resources):
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
            await _execute_parallel_step_logic(
                parallel_step=parallel_step,
                parallel_input="test_input",
                context=None,
                resources=None,
                step_executor=high_token_executor,
                context_model_defined=True,
                usage_limits=usage_limits,
                context_setter=mock_context_setter,
            )

    @pytest.mark.asyncio
    async def test_parallel_execution_branch_failure_propagate(self, mock_context_setter):
        """Test parallel execution with branch failure and PROPAGATE strategy."""

        # Create a step executor that fails
        async def failing_executor(step, input_data, context, resources):
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

        result = await _execute_parallel_step_logic(
            parallel_step=parallel_step,
            parallel_input="test_input",
            context=None,
            resources=None,
            step_executor=failing_executor,
            context_model_defined=True,
            context_setter=mock_context_setter,
        )

        assert not result.success
        assert "Branch 'branch1' failed" in result.feedback

    @pytest.mark.asyncio
    async def test_parallel_execution_branch_failure_ignore(
        self, mock_step_executor, mock_context_setter
    ):
        """Test parallel execution with branch failure and IGNORE strategy."""
        # Create a step executor that fails for one branch
        branch_results = {"branch1": True, "branch2": False}

        async def conditional_failing_executor(step, input_data, context, resources):
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

        result = await _execute_parallel_step_logic(
            parallel_step=parallel_step,
            parallel_input="test_input",
            context=None,
            resources=None,
            step_executor=conditional_failing_executor,
            context_model_defined=True,
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
        result = await _execute_parallel_step_logic(
            parallel_step=parallel_step,
            parallel_input="test_input",
            context=context,
            resources=None,
            step_executor=mock_step_executor,
            context_model_defined=True,
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

        result = await _execute_parallel_step_logic(
            parallel_step=parallel_step,
            parallel_input="test_input",
            context=context,
            resources=None,
            step_executor=mock_step_executor,
            context_model_defined=True,
            context_setter=mock_context_setter,
        )

        assert result.success

    @pytest.mark.asyncio
    async def test_parallel_execution_merge_scratchpad_no_scratchpad(
        self, mock_step_executor, mock_context_setter
    ):
        """Test parallel execution with MERGE_SCRATCHPAD strategy but no scratchpad."""
        parallel_step = ParallelStep(
            name="test_parallel",
            branches={
                "branch1": Pipeline(steps=[Step(name="step1", agent=StubAgent(["output1"]))])
            },
            merge_strategy=MergeStrategy.MERGE_SCRATCHPAD,
            on_branch_failure=BranchFailureStrategy.PROPAGATE,
        )

        # Use a context with no scratchpad attribute at all
        class NoScratchpadContext:
            def __init__(self, data):
                self.data = data

            def model_dump(self):
                return self.data.copy()

            @classmethod
            def model_validate(cls, data):
                return cls(data)

        context = NoScratchpadContext({"key": "value"})
        with pytest.raises(
            ValueError,
            match="MERGE_SCRATCHPAD strategy requires context with 'scratchpad' attribute",
        ):
            await _execute_parallel_step_logic(
                parallel_step=parallel_step,
                parallel_input="test_input",
                context=context,
                resources=None,
                step_executor=mock_step_executor,
                context_model_defined=True,
                context_setter=mock_context_setter,
            )

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

        result = await _execute_parallel_step_logic(
            parallel_step=parallel_step,
            parallel_input="test_input",
            context=context,
            resources=None,
            step_executor=mock_step_executor,
            context_model_defined=True,
            context_setter=mock_context_setter,
        )

        assert result.success
        assert context.data["merged"] is True

    @pytest.mark.asyncio
    async def test_parallel_execution_exception_handling(self, mock_context_setter):
        """Test parallel execution with exception handling."""

        # Create a step executor that raises an exception
        async def exception_executor(step, input_data, context, resources):
            raise ValueError("Test exception")

        parallel_step = ParallelStep(
            name="test_parallel",
            branches={
                "branch1": Pipeline(steps=[Step(name="step1", agent=StubAgent(["output1"]))])
            },
            merge_strategy=MergeStrategy.NO_MERGE,
            on_branch_failure=BranchFailureStrategy.PROPAGATE,
        )
        result = await _execute_parallel_step_logic(
            parallel_step=parallel_step,
            parallel_input="test_input",
            context=None,
            resources=None,
            step_executor=exception_executor,
            context_model_defined=True,
            context_setter=mock_context_setter,
        )
        assert not result.success
        # The feedback is set by the propagate logic, but the branch result should have the error
        assert "failed. Propagating failure" in result.feedback
        assert "Branch execution error" in result.output["branch1"].feedback

    @pytest.mark.asyncio
    async def test_parallel_execution_task_cancellation(self, mock_context_setter):
        """Test parallel execution with task cancellation."""

        # Create a step executor that takes time
        async def slow_executor(step, input_data, context, resources):
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
                "branch1": Pipeline(steps=[Step(name="step1", agent=StubAgent(["output1"]))]),
                "branch2": Pipeline(steps=[Step(name="step2", agent=StubAgent(["output2"]))]),
            },
            merge_strategy=MergeStrategy.NO_MERGE,
            on_branch_failure=BranchFailureStrategy.PROPAGATE,
        )

        # Test with usage limits that will trigger cancellation
        usage_limits = UsageLimits(total_cost_usd_limit=0.001)  # Very low limit

        with pytest.raises(UsageLimitExceededError):
            await _execute_parallel_step_logic(
                parallel_step=parallel_step,
                parallel_input="test_input",
                context=None,
                resources=None,
                step_executor=slow_executor,
                context_model_defined=True,
                usage_limits=usage_limits,
                context_setter=mock_context_setter,
            )
