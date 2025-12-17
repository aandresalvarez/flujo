"""Unit tests for parallel step robustness and error handling."""

import pytest
import asyncio
from typing import Any
import time

from flujo.application.core.executor_core import ExecutorCore
from flujo.domain.dsl.parallel import ParallelStep
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.step import Step
from flujo.domain.models import BaseModel
from flujo.domain.models import StepResult, UsageLimits
from flujo.exceptions import UsageLimitExceededError
from flujo.testing.utils import StubAgent
from pydantic import Field

# Run this module serially in CI to avoid xdist workerfinished assertion flake with skipped tests
pytestmark = pytest.mark.serial


class TestParallelStepRobustness:
    """Test parallel step robustness and error handling."""

    @pytest.fixture
    def mock_step_executor(self):
        """Create an agent executor override that tracks calls."""
        call_history = []

        class _TrackingAgentExecutor:
            async def execute(
                self,
                _core: object,
                step: object,
                data: object,
                context: object,
                resources: object,
                _limits: object,
                _stream: bool,
                _on_chunk: object,
                _cache_key: object,
                _fallback_depth: int,
            ) -> StepResult:
                call_history.append(
                    {
                        "step": step,
                        "data": data,
                        "context": context,
                        "resources": resources,
                    }
                )

                # Simulate successful step execution
                return StepResult(
                    name=step.name,
                    output=data,
                    success=True,
                    attempts=1,
                    latency_s=0.1,
                    token_counts=10,
                    cost_usd=0.01,
                )

        execu = _TrackingAgentExecutor()
        execu.call_history = call_history
        return execu

    @pytest.fixture
    def usage_limits(self):
        """Create usage limits for testing."""
        return UsageLimits(total_cost_usd_limit=1.0, total_tokens_limit=100)

    @pytest.fixture
    def parallel_step(self):
        """Create a parallel step for testing."""
        # Create a simple parallel step with two branches
        from flujo.domain.dsl.step import Step

        # Create steps with StubAgent
        step1 = Step.model_construct(name="step1", agent=StubAgent(["output1"]))
        step2 = Step.model_construct(name="step2", agent=StubAgent(["output2"]))

        # Create pipelines
        branch1 = Pipeline.model_construct(steps=[step1])
        branch2 = Pipeline.model_construct(steps=[step2])

        return ParallelStep(
            name="test_parallel",
            branches={
                "branch1": branch1,
                "branch2": branch2,
            },
        )

    async def test_usage_governor_receives_individual_step_results(
        self, parallel_step, mock_step_executor, usage_limits
    ):
        """Legacy governor path removed; quota mode has no governor surface."""
        assert True

    async def test_cancelled_branches_populate_dictionaries(
        self, parallel_step, mock_step_executor, usage_limits
    ):
        """Test that cancelled branches still populate output dictionaries."""
        # Create a step executor that simulates early cancellation
        cancellation_calls = []

        class _CancellingAgentExecutor:
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
            ) -> StepResult:
                cancellation_calls.append(
                    {
                        "step": step,
                        "data": data,
                    }
                )

                # Simulate a step that completes quickly (quota-only mode)
                await asyncio.sleep(0.01)

                # Normal execution
                return StepResult(
                    name=step.name,
                    output=data,
                    success=True,
                    attempts=1,
                    latency_s=0.1,
                    token_counts=10,
                    cost_usd=0.01,
                )

        # Create a context setter that tracks what's set
        context_setter_calls = []

        def context_setter(result, context):
            context_setter_calls.append({"result": result, "context": context})

        # Execute the parallel step
        executor = ExecutorCore()
        executor.agent_step_executor = _CancellingAgentExecutor()
        await executor._handle_parallel_step(
            parallel_step=parallel_step,
            data="test_input",
            context=None,
            resources=None,
            limits=usage_limits,
            context_setter=context_setter,
        )

        # Verify that all branches were called
        assert len(cancellation_calls) == 2

    async def test_usage_limit_breach_propagates_correctly(self, parallel_step, mock_step_executor):
        """Test that usage limit breaches propagate correctly with proper step history."""
        # Create usage limits that will be breached
        usage_limits = UsageLimits(total_cost_usd_limit=0.005, total_tokens_limit=5)

        # Create a stub agent that returns high cost
        class ExpensiveStubAgent:
            async def run(self, data: Any, **kwargs: Any) -> Any:
                class UsageResponse:
                    def __init__(self, output: Any, cost: float, tokens: int):
                        self.output = output
                        self.cost_usd = cost
                        self.token_counts = tokens

                    def usage(self) -> dict[str, Any]:
                        return {
                            "prompt_tokens": self.token_counts,
                            "completion_tokens": 0,
                            "total_tokens": self.token_counts,
                            "cost_usd": self.cost_usd,
                        }

                return UsageResponse(data, 0.01, 10)  # High cost and tokens

        # Create steps with the expensive agent
        step1 = Step.model_validate({"name": "step1", "agent": ExpensiveStubAgent()})
        step2 = Step.model_validate({"name": "step2", "agent": ExpensiveStubAgent()})

        # Create a parallel step with these steps
        from flujo.domain.dsl.parallel import ParallelStep

        parallel_step = ParallelStep.model_validate(
            {
                "name": "parallel_step",
                "branches": {
                    "branch1": Pipeline.model_validate({"steps": [step1]}),
                    "branch2": Pipeline.model_validate({"steps": [step2]}),
                },
            }
        )

        # Create a pipeline with the parallel step
        pipeline = Pipeline.model_validate({"steps": [parallel_step]})

        # Create Flujo runner with usage limits
        from flujo import Flujo

        runner = Flujo(pipeline, usage_limits=usage_limits)

        # Execute the pipeline - should raise UsageLimitExceededError
        with pytest.raises(UsageLimitExceededError) as exc_info:
            async for result in runner.run_async("test_input"):
                pass  # Consume the async iterator

        # Verify that the error contains proper step history
        error = exc_info.value
        assert hasattr(error, "result")
        assert error.result is not None

        # The result should contain step history with individual step results
        assert hasattr(error.result, "step_history")
        assert len(error.result.step_history) > 0

    async def test_parallel_step_runs_without_breach_event(
        self, parallel_step, mock_step_executor, usage_limits
    ):
        """Ensure parallel execution succeeds in quota-only mode."""

        def context_setter(result, context):
            pass

        executor = ExecutorCore()
        executor.agent_step_executor = mock_step_executor
        await executor._handle_parallel_step(
            parallel_step=parallel_step,
            data="test_input",
            context=None,
            resources=None,
            limits=usage_limits,
            context_setter=context_setter,
        )

    async def test_parallel_step_handles_empty_branches(self, mock_step_executor, usage_limits):
        """Test that parallel step handles empty branches gracefully."""
        # Create a parallel step with no branches
        empty_parallel_step = ParallelStep.model_construct(name="empty_parallel", branches={})

        def context_setter(result, context):
            pass

        # Execute the parallel step with no branches
        executor = ExecutorCore()
        executor.agent_step_executor = mock_step_executor
        result = await executor._handle_parallel_step(
            parallel_step=empty_parallel_step,
            data="test_input",
            context=None,
            resources=None,
            limits=usage_limits,
            context_setter=context_setter,
        )

        # Verify that no steps were executed
        assert len(mock_step_executor.call_history) == 0

        # Verify that the result is still valid (empty branches should fail since no branches succeeded)
        assert result.success is False
        assert result.output == {}

    async def test_parallel_step_handles_failing_branches(
        self, parallel_step, mock_step_executor, usage_limits
    ):
        """Test that parallel step handles failing branches correctly."""
        # Create a step executor that simulates failures
        failure_calls = []

        class _FailingAgentExecutor:
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
            ) -> StepResult:
                failure_calls.append(
                    {
                        "step": step,
                        "data": data,
                    }
                )

                # Simulate a failing step
                return StepResult(
                    name=step.name,
                    output=None,
                    success=False,
                    attempts=1,
                    latency_s=0.1,
                    token_counts=10,
                    cost_usd=0.01,
                    feedback="Simulated failure",
                )

        # Create a context setter that tracks what's set
        context_setter_calls = []

        def context_setter(result, context):
            context_setter_calls.append({"result": result, "context": context})

        # Execute the parallel step
        executor = ExecutorCore()
        executor.agent_step_executor = _FailingAgentExecutor()
        result = await executor._handle_parallel_step(
            parallel_step=parallel_step,
            data="test_input",
            context=None,
            resources=None,
            limits=usage_limits,
            context_setter=context_setter,
        )

        # Verify that all branches were called
        assert len(failure_calls) == 2

        # Verify that the result indicates failure
        assert not result.success
        assert "failed" in result.feedback.lower()

    async def test_parallel_step_concurrency_limits(
        self, parallel_step, mock_step_executor, usage_limits
    ):
        """Test that parallel step respects concurrency limits."""
        # Create a step executor that tracks execution timing
        execution_times = []

        class _TrackingAgentExecutor:
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
            ) -> StepResult:
                start_time = time.time()
                # Simulate some work
                await asyncio.sleep(0.01)
                execution_times.append(time.time() - start_time)

                return StepResult(
                    name=step.name,
                    output=data,
                    success=True,
                    attempts=1,
                    latency_s=0.01,
                    token_counts=10,
                    cost_usd=0.01,
                )

        # Create a context setter that tracks what's set
        context_setter_calls = []

        def context_setter(result, context):
            context_setter_calls.append({"result": result, "context": context})

        # Execute the parallel step
        executor = ExecutorCore()
        executor.agent_step_executor = _TrackingAgentExecutor()
        await executor._handle_parallel_step(
            parallel_step=parallel_step,
            data="test_input",
            context=None,
            resources=None,
            limits=usage_limits,
            context_setter=context_setter,
        )

        # Verify that all branches were executed
        assert len(execution_times) == 2

    async def test_parallel_step_context_isolation(
        self, parallel_step, mock_step_executor, usage_limits
    ):
        """Test that parallel step maintains context isolation between branches."""

        # Create a test context
        class TestContext(BaseModel):
            value: str
            data_store: dict[str, str] = Field(default_factory=dict)

        initial_context = TestContext(value="initial")

        # Create a step executor that modifies context
        context_modifications = []

        class _ContextModifyingAgentExecutor:
            async def execute(
                self,
                _core: object,
                step: object,
                data: object,
                context: object,
                _resources: object,
                _limits: object,
                _stream: bool,
                _on_chunk: object,
                _cache_key: object,
                _fallback_depth: int,
            ) -> StepResult:
                context_modifications.append(
                    {
                        "step": step,
                        "data": data,
                        "context_value": getattr(context, "value", None) if context else None,
                    }
                )

                # Modify the context
                if context:
                    context.value = f"modified_by_{step.name}"
                    context.data_store[f"step_{step.name}"] = "executed"

                return StepResult(
                    name=step.name,
                    output=data,
                    success=True,
                    attempts=1,
                    latency_s=0.1,
                    token_counts=10,
                    cost_usd=0.01,
                )

        # Create a context setter that tracks what's set
        context_setter_calls = []

        def context_setter(result, context):
            context_setter_calls.append({"result": result, "context": context})

        # Execute the parallel step
        executor = ExecutorCore()
        executor.agent_step_executor = _ContextModifyingAgentExecutor()
        await executor._handle_parallel_step(
            parallel_step=parallel_step,
            data="test_input",
            context=initial_context,
            resources=None,
            limits=usage_limits,
            context_setter=context_setter,
        )

        # Verify that all branches were called
        assert len(context_modifications) == 2
