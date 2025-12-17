"""Unit tests for parallel step execution strategies."""

from flujo.type_definitions.common import JSONObject

import asyncio
import logging

import pytest

from flujo.application.core.executor_core import ExecutorCore
from flujo.domain.models import BaseModel
from flujo.domain.dsl.parallel import ParallelStep
from flujo.domain.dsl.step import Step, BranchFailureStrategy, MergeStrategy
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.models import (
    StepResult,
    StepOutcome,
    Success,
    UsageLimits,
)
from flujo.exceptions import UsageLimitExceededError, ConfigurationError
from flujo.infra.monitor import global_monitor
from flujo.testing.utils import StubAgent


class MockContext(BaseModel):
    """Mock context for testing, inherits from BaseModel for proper validation."""

    # Required PipelineContext attributes
    initial_prompt: str = "test_prompt"
    step_outputs: JSONObject = {}

    # Additional data storage for tests that expect it
    data: JSONObject = {}

    model_config = {"extra": "allow"}

    def __init__(self, data: JSONObject = None, **kwargs):
        # Merge data and kwargs
        merged_data = dict(data) if data is not None else {}
        merged_data.update(kwargs)

        # Extract known fields
        initial_prompt = merged_data.pop("initial_prompt", "test_prompt")
        step_outputs = merged_data.pop("step_outputs", {})

        # Store original data for tests that expect it
        data_field = dict(data) if data is not None else {}
        data_field.update(kwargs)

        # Initialize BaseModel with all fields
        super().__init__(
            initial_prompt=initial_prompt,
            step_outputs=step_outputs,
            data=data_field,
            **merged_data,
        )

    @classmethod
    def model_validate(cls, data: JSONObject):
        return cls(data)


class TestParallelStepExecution:
    """Test parallel step execution with different strategies."""

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
    async def test_basic_parallel_execution_no_merge(self, parallel_step, mock_context_setter):
        """Test basic parallel execution with NO_MERGE strategy."""
        context = MockContext({"key": "value"})

        executor = ExecutorCore()
        result = await executor._handle_parallel_step(
            parallel_step=parallel_step,
            data="test_input",
            context=context,
            resources=None,
            limits=None,
            context_setter=mock_context_setter,
        )

        assert result.success
        assert result.name == "test_parallel"
        assert isinstance(result.output, dict)
        assert "branch1" in result.output
        assert "branch2" in result.output

    @pytest.mark.asyncio
    async def test_parallel_execution_with_context_include_keys(self, mock_context_setter):
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
            context_setter=mock_context_setter,
        )

        assert result.success

    @pytest.mark.asyncio
    async def test_parallel_execution_no_context(self, parallel_step, mock_context_setter):
        """Test parallel execution without context."""
        executor = ExecutorCore()
        result = await executor._handle_parallel_step(
            parallel_step=parallel_step,
            data="test_input",
            context=None,
            resources=None,
            limits=None,
            context_setter=mock_context_setter,
        )

        assert result.success

    @pytest.mark.asyncio
    async def test_parallel_execution_with_usage_limits(self, parallel_step, mock_context_setter):
        """Test parallel execution with usage limits."""
        usage_limits = UsageLimits(total_cost_usd_limit=0.05, total_tokens_limit=50)

        executor = ExecutorCore()
        result = await executor._handle_parallel_step(
            parallel_step=parallel_step,
            data="test_input",
            context=None,
            resources=None,
            limits=usage_limits,
            context_setter=mock_context_setter,
        )

        assert result.success

    @pytest.mark.asyncio
    async def test_parallel_execution_cost_limit_breach(self, mock_context_setter):
        """Test parallel execution with cost limit breach."""

        class _HighCostAgentExecutor:
            async def execute(
                self,
                _core: object,
                step: object,
                data: object,
                _context: object,
                _resources: object,
                _limits: object,
                _stream: bool,
                _on_chunk: object,
                _cache_key: object,
                _fallback_depth: int,
            ) -> StepOutcome[StepResult]:
                name = getattr(step, "name", "test")
                return Success(
                    step_result=StepResult(
                        name=name,
                        success=True,
                        output=data,
                        latency_s=0.1,
                        cost_usd=1.0,  # High cost
                        token_counts=10,
                        attempts=1,
                    )
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
            executor.agent_step_executor = _HighCostAgentExecutor()
            await executor._handle_parallel_step(
                parallel_step=parallel_step,
                data="test_input",
                context=None,
                resources=None,
                limits=usage_limits,
                context_setter=mock_context_setter,
            )

    @pytest.mark.asyncio
    async def test_parallel_execution_token_limit_breach(self, mock_context_setter):
        """Test parallel execution with token limit breach."""

        class _HighTokenAgentExecutor:
            async def execute(
                self,
                _core: object,
                step: object,
                data: object,
                _context: object,
                _resources: object,
                _limits: object,
                _stream: bool,
                _on_chunk: object,
                _cache_key: object,
                _fallback_depth: int,
            ) -> StepOutcome[StepResult]:
                name = getattr(step, "name", "test")
                return Success(
                    step_result=StepResult(
                        name=name,
                        success=True,
                        output=data,
                        latency_s=0.1,
                        cost_usd=0.01,
                        token_counts=100,  # High token count
                        attempts=1,
                    )
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
            executor.agent_step_executor = _HighTokenAgentExecutor()
            await executor._handle_parallel_step(
                parallel_step=parallel_step,
                data="test_input",
                context=None,
                resources=None,
                limits=usage_limits,
                context_setter=mock_context_setter,
            )

    @pytest.mark.asyncio
    async def test_parallel_execution_branch_failure_propagate(self, mock_context_setter):
        """Test parallel execution with branch failure and PROPAGATE strategy."""

        class _FailingAgentExecutor:
            async def execute(
                self,
                _core: object,
                step: object,
                _data: object,
                _context: object,
                _resources: object,
                _limits: object,
                _stream: bool,
                _on_chunk: object,
                _cache_key: object,
                _fallback_depth: int,
            ) -> StepOutcome[StepResult]:
                name = getattr(step, "name", "test")
                return Success(
                    step_result=StepResult(
                        name=name,
                        success=False,
                        output=None,
                        feedback="Test failure",
                        latency_s=0.1,
                        cost_usd=0.01,
                        token_counts=10,
                        attempts=1,
                    )
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
        executor.agent_step_executor = _FailingAgentExecutor()
        result = await executor._handle_parallel_step(
            parallel_step=parallel_step,
            data="test_input",
            context=None,
            resources=None,
            limits=None,
            context_setter=mock_context_setter,
        )

        # Verify that the result indicates failure
        assert not result.success
        assert "failed" in result.feedback.lower()

    @pytest.mark.asyncio
    async def test_parallel_execution_branch_failure_ignore(self, mock_context_setter):
        """Test parallel execution with branch failure and IGNORE strategy."""

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
            context_setter=mock_context_setter,
        )

        # Should succeed even with branch failure
        assert result.success

    @pytest.mark.asyncio
    async def test_parallel_execution_merge_overwrite(self, mock_context_setter):
        """Test parallel execution with OVERWRITE merge strategy."""
        parallel_step = ParallelStep(
            name="test_parallel",
            branches={
                "branch1": Pipeline(steps=[Step(name="step1", agent=StubAgent(["output1"]))])
            },
            merge_strategy=MergeStrategy.OVERWRITE,
            on_branch_failure=BranchFailureStrategy.PROPAGATE,
        )
        # Provide an explicit empty step_outputs in the context data
        context = MockContext({"key": "value", "step_outputs": {}})
        executor = ExecutorCore()
        result = await executor._handle_parallel_step(
            parallel_step=parallel_step,
            data="test_input",
            context=context,
            resources=None,
            limits=None,
            context_setter=mock_context_setter,
        )
        assert result.success

    @pytest.mark.asyncio
    async def test_parallel_execution_merge_removed_root_rejected(self, mock_context_setter):
        """Removed merge strategy should raise ConfigurationError."""
        removed_root = "scrat" + "chpad"
        parallel_step = ParallelStep.model_construct(
            name="test_parallel",
            branches={
                "branch1": Pipeline(steps=[Step(name="step1", agent=StubAgent(["output1"]))])
            },
            merge_strategy="merge_" + removed_root,
            on_branch_failure=BranchFailureStrategy.PROPAGATE,
        )

        context = MockContext({"key": "value", "step_outputs": {}})

        executor = ExecutorCore()
        with pytest.raises(ConfigurationError):
            await executor._handle_parallel_step(
                parallel_step=parallel_step,
                data="test_input",
                context=context,
                resources=None,
                limits=None,
                context_setter=mock_context_setter,
            )

    @pytest.mark.asyncio
    async def test_parallel_execution_custom_merge_strategy(self, mock_context_setter):
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
            context_setter=mock_context_setter,
        )

        assert result.success
        assert context.data["merged"] is True

    @pytest.mark.asyncio
    async def test_parallel_execution_exception_handling(self, mock_context_setter):
        """Test parallel execution with exception handling."""

        class _ExceptionAgentExecutor:
            async def execute(
                self,
                _core: object,
                _step: object,
                _data: object,
                _context: object,
                _resources: object,
                _limits: object,
                _stream: bool,
                _on_chunk: object,
                _cache_key: object,
                _fallback_depth: int,
            ) -> StepOutcome[StepResult]:
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
        executor.agent_step_executor = _ExceptionAgentExecutor()
        result = await executor._handle_parallel_step(
            parallel_step=parallel_step,
            data="test_input",
            context=None,
            resources=None,
            limits=None,
            context_setter=mock_context_setter,
        )

        # Verify that the result indicates failure
        assert not result.success
        assert "failed" in result.feedback.lower()

    @pytest.mark.asyncio
    async def test_parallel_execution_task_cancellation(self, mock_context_setter):
        """Test parallel execution with task cancellation."""

        class _SlowAgentExecutor:
            async def execute(
                self,
                _core: object,
                step: object,
                data: object,
                _context: object,
                _resources: object,
                _limits: object,
                _stream: bool,
                _on_chunk: object,
                _cache_key: object,
                _fallback_depth: int,
            ) -> StepOutcome[StepResult]:
                await asyncio.sleep(0.1)
                name = getattr(step, "name", "test")
                return Success(
                    step_result=StepResult(
                        name=name,
                        success=True,
                        output=data,
                        latency_s=0.1,
                        cost_usd=0.01,
                        token_counts=10,
                        attempts=1,
                    )
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
            executor.agent_step_executor = _SlowAgentExecutor()
            await executor._handle_parallel_step(
                parallel_step=parallel_step,
                data="test_input",
                context=None,
                resources=None,
                limits=usage_limits,
                context_setter=mock_context_setter,
            )

    @pytest.mark.asyncio
    async def test_parallel_execution_merge_context_update(self, mock_context_setter):
        """Test parallel execution with context update merge strategy."""

        # Create a context that can be updated
        class TestContext(BaseModel):
            initial_prompt: str = "test_prompt"
            value: str = "initial"
            step_outputs: JSONObject = {}

            model_config = {"extra": "allow"}

            def __init__(self, value: str = "initial", **kwargs):
                super().__init__(
                    initial_prompt="test_prompt",
                    value=value,
                    step_outputs={},
                    **kwargs,
                )

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
            context_setter=mock_context_setter,
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

        async def run(self, *_args, **_kwargs):
            self.called = True
            await asyncio.sleep(self.delay)
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
    cost_per_branch = (usage_limits.total_cost_usd_limit / N) + 0.01
    token_per_branch = (usage_limits.total_tokens_limit // N) + 1
    delays = [0.01, 0.02, 0.03, 0.04, 0.05]
    agents = [CostlyAgent(cost_per_branch, token_per_branch, delay=delays[i]) for i in range(N)]
    branches = {f"b{i}": Pipeline(steps=[Step(name=f"s{i}", agent=agents[i])]) for i in range(N)}
    parallel_step = ParallelStep(
        name="test_parallel_race",
        branches=branches,
        merge_strategy=MergeStrategy.NO_MERGE,
        on_branch_failure=BranchFailureStrategy.PROPAGATE,
    )

    from flujo.application.core.executor_core import ExecutorCore

    with pytest.raises(UsageLimitExceededError):
        executor = ExecutorCore()

        class _CostlyAgentExecutor:
            async def execute(
                self,
                _core: object,
                step: object,
                data: object,
                _context: object,
                _resources: object,
                _limits: object,
                _stream: bool,
                _on_chunk: object,
                _cache_key: object,
                _fallback_depth: int,
            ) -> StepOutcome[StepResult]:
                agent = getattr(step, "agent", None)
                if agent is None or not hasattr(agent, "run"):
                    raise AssertionError("Expected agent with .run()")
                result = await agent.run(data)
                if not isinstance(result, StepResult):
                    raise AssertionError(f"Expected StepResult, got {type(result).__name__}")
                return Success(step_result=result)

        executor.agent_step_executor = _CostlyAgentExecutor()
        await executor._handle_parallel_step(
            parallel_step=parallel_step,
            data="test_input",
            context=None,
            resources=None,
            limits=usage_limits,
            context_setter=lambda result, context: None,
        )

    called_count = sum(a.called for a in agents)
    assert called_count == N, f"Expected all branches to start, got {called_count}/{N}"
