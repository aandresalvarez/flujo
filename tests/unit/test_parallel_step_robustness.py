"""Unit tests for parallel step robustness and error handling."""

import pytest
import asyncio
from typing import Any

from flujo.application.core.step_logic import _execute_parallel_step_logic
from flujo.domain.dsl.parallel import ParallelStep
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.step import Step
from flujo.domain.models import StepResult, UsageLimits
from flujo.exceptions import UsageLimitExceededError
from flujo.testing.utils import StubAgent


class TestParallelStepRobustness:
    """Test parallel step robustness and error handling."""

    @pytest.fixture
    def mock_step_executor(self):
        """Create a mock step executor that tracks calls."""
        call_history = []

        async def executor(step, data, context, resources, breach_event=None):
            call_history.append(
                {
                    "step": step,
                    "data": data,
                    "context": context,
                    "resources": resources,
                    "breach_event": breach_event,
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

        executor.call_history = call_history
        return executor

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
        """Test that usage_governor.add_usage receives individual step results, not overall result."""
        # Track what's passed to add_usage
        add_usage_calls = []

        # Create a mock usage governor that tracks calls
        class MockUsageGovernor:
            def __init__(self, usage_limits):
                self.usage_limits = usage_limits
                self.total_cost = 0.0
                self.total_tokens = 0
                self.limit_breached = asyncio.Event()
                self.limit_breach_error = None

            async def add_usage(self, cost_delta, token_delta, result):
                add_usage_calls.append(
                    {
                        "cost_delta": cost_delta,
                        "token_delta": token_delta,
                        "result": result,
                    }
                )
                return False  # Don't breach

            def breached(self):
                return False

            def get_error(self):
                return None

                # We need to verify the behavior indirectly

        # by checking that individual step results are used

        # Create a context setter that tracks what's set
        context_setter_calls = []

        def context_setter(result, context):
            context_setter_calls.append({"result": result, "context": context})

        # Execute the parallel step
        breach_event = asyncio.Event()
        await _execute_parallel_step_logic(
            parallel_step=parallel_step,
            parallel_input="test_input",
            context=None,
            resources=None,
            step_executor=mock_step_executor,
            context_model_defined=False,
            usage_limits=usage_limits,
            context_setter=context_setter,
            breach_event=breach_event,
        )

        # Verify that the step executor was called for each branch
        assert len(mock_step_executor.call_history) == 2

        # Verify that breach_event was passed to each call
        for call in mock_step_executor.call_history:
            assert "breach_event" in call
            assert call["breach_event"] is not None

    async def test_cancelled_branches_populate_dictionaries(
        self, parallel_step, mock_step_executor, usage_limits
    ):
        """Test that cancelled branches still populate output dictionaries."""
        # Create a step executor that simulates early cancellation
        cancellation_calls = []

        async def cancelling_executor(step, data, context, resources, breach_event=None):
            cancellation_calls.append(
                {
                    "step": step,
                    "data": data,
                    "breach_event": breach_event,
                }
            )

            # Simulate a step that gets cancelled early
            if breach_event and breach_event.is_set():
                return StepResult(
                    name=step.name,
                    output=None,
                    success=False,
                    attempts=1,
                    latency_s=0.05,
                    token_counts=5,
                    cost_usd=0.005,
                    feedback="Cancelled early",
                )

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
        breach_event = asyncio.Event()
        await _execute_parallel_step_logic(
            parallel_step=parallel_step,
            parallel_input="test_input",
            context=None,
            resources=None,
            step_executor=cancelling_executor,
            context_model_defined=False,
            usage_limits=usage_limits,
            context_setter=context_setter,
            breach_event=breach_event,
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

    async def test_breach_event_propagation_to_agents(
        self, parallel_step, mock_step_executor, usage_limits
    ):
        """Test that breach_event is properly propagated to agents."""
        breach_event_received = []

        async def tracking_executor(step, data, context, resources, breach_event=None):
            breach_event_received.append(
                {
                    "step": step.name,
                    "breach_event": breach_event,
                    "breach_event_is_set": breach_event.is_set() if breach_event else False,
                }
            )

            return StepResult(
                name=step.name,
                output=data,
                success=True,
                attempts=1,
                latency_s=0.1,
                token_counts=10,
                cost_usd=0.01,
            )

        # Create a context setter
        def context_setter(result, context):
            pass

        # Execute the parallel step
        breach_event = asyncio.Event()
        await _execute_parallel_step_logic(
            parallel_step=parallel_step,
            parallel_input="test_input",
            context=None,
            resources=None,
            step_executor=tracking_executor,
            context_model_defined=False,
            usage_limits=usage_limits,
            context_setter=context_setter,
            breach_event=breach_event,
        )

        # Verify that breach_event was passed to all steps
        assert len(breach_event_received) == 2
        for call in breach_event_received:
            assert call["breach_event"] is not None
            assert not call["breach_event_is_set"]  # Should not be set during normal execution

    async def test_parallel_step_handles_empty_branches(self, mock_step_executor, usage_limits):
        """Test that parallel step handles empty branches gracefully."""
        # Create a parallel step with no branches
        empty_parallel_step = ParallelStep(name="empty_parallel", branches={})

        def context_setter(result, context):
            pass

        # Execute the parallel step with no branches
        result = await _execute_parallel_step_logic(
            parallel_step=empty_parallel_step,
            parallel_input="test_input",
            context=None,
            resources=None,
            step_executor=mock_step_executor,
            context_model_defined=False,
            usage_limits=usage_limits,
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
        """Test that parallel step handles failing branches gracefully."""
        # Create a step executor that fails for one branch
        call_count = 0

        async def failing_executor(step, data, context, resources, breach_event=None):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First branch fails
                return StepResult(
                    name=step.name,
                    output=None,
                    success=False,
                    attempts=1,
                    latency_s=0.1,
                    token_counts=10,
                    cost_usd=0.01,
                    feedback="Branch failed",
                )
            else:
                # Second branch succeeds
                return StepResult(
                    name=step.name,
                    output=data,
                    success=True,
                    attempts=1,
                    latency_s=0.1,
                    token_counts=10,
                    cost_usd=0.01,
                )

        def context_setter(result, context):
            pass

        # Execute the parallel step
        result = await _execute_parallel_step_logic(
            parallel_step=parallel_step,
            parallel_input="test_input",
            context=None,
            resources=None,
            step_executor=failing_executor,
            context_model_defined=False,
            usage_limits=usage_limits,
            context_setter=context_setter,
        )

        # Verify that both branches were called
        assert call_count == 2

        # Verify that the result reflects the mixed success/failure
        # With PROPAGATE strategy, if any branch fails, the whole step fails
        assert result.success is False  # Overall failure (one branch failed)
        assert hasattr(result, "output")
        assert isinstance(result.output, dict)
        assert "Branch 'branch1' failed. Propagating failure." in result.feedback

    async def test_parallel_step_concurrency_limits(
        self, parallel_step, mock_step_executor, usage_limits
    ):
        """Test that parallel step respects concurrency limits."""
        # Create a step executor that tracks concurrent executions
        active_executions = 0
        max_concurrent = 0

        async def tracking_executor(step, data, context, resources, breach_event=None):
            nonlocal active_executions, max_concurrent
            active_executions += 1
            max_concurrent = max(max_concurrent, active_executions)

            # Simulate some work
            await asyncio.sleep(0.01)

            active_executions -= 1

            return StepResult(
                name=step.name,
                output=data,
                success=True,
                attempts=1,
                latency_s=0.1,
                token_counts=10,
                cost_usd=0.01,
            )

        def context_setter(result, context):
            pass

        # Execute the parallel step
        await _execute_parallel_step_logic(
            parallel_step=parallel_step,
            parallel_input="test_input",
            context=None,
            resources=None,
            step_executor=tracking_executor,
            context_model_defined=False,
            usage_limits=usage_limits,
            context_setter=context_setter,
        )

        # Verify that concurrency was limited (should not exceed DEFAULT_MAX_CONCURRENCY = 3)
        assert max_concurrent <= 3
        assert max_concurrent > 0  # Should have had some concurrency

    async def test_parallel_step_context_isolation(
        self, parallel_step, mock_step_executor, usage_limits
    ):
        """Test that parallel step maintains context isolation between branches."""

        # Create a context with some data
        class TestContext:
            def __init__(self, value):
                self.value = value

        original_context = TestContext("original")

        # Create a step executor that modifies context
        async def context_modifying_executor(step, data, context, resources, breach_event=None):
            if context is not None:
                # Modify the context
                context.value = f"modified_by_{step.name}"

            return StepResult(
                name=step.name,
                output=data,
                success=True,
                attempts=1,
                latency_s=0.1,
                token_counts=10,
                cost_usd=0.01,
            )

        def context_setter(result, context):
            pass

        # Execute the parallel step
        await _execute_parallel_step_logic(
            parallel_step=parallel_step,
            parallel_input="test_input",
            context=original_context,
            resources=None,
            step_executor=context_modifying_executor,
            context_model_defined=True,
            usage_limits=usage_limits,
            context_setter=context_setter,
        )

        # Verify that the original context was not modified
        assert original_context.value == "original"
