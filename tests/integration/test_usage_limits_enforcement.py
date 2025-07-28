"""Integration tests for usage limits enforcement in UltraStepExecutor."""

import pytest
from typing import Any

from flujo import Step, Pipeline, Flujo
from flujo.domain.models import UsageLimits
from flujo.exceptions import UsageLimitExceededError
from flujo.testing import StubAgent


class CostTrackingStubAgent(StubAgent):
    """Stub agent that simulates cost and token usage for testing."""

    def __init__(
        self,
        cost_per_call: float = 0.50,
        tokens_per_call: int = 100,
        outputs: list[Any] = None,
    ):
        # Provide default outputs if none specified
        if outputs is None:
            outputs = ["test_output"] * 10  # Default to 10 outputs
        super().__init__(outputs)
        self.cost_per_call = cost_per_call
        self.tokens_per_call = tokens_per_call

    async def run(self, data: Any, **kwargs: Any) -> Any:
        """Return a response with usage metrics."""
        response = await super().run(data, **kwargs)

        # Create a response object with usage metrics
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

        return UsageResponse(response, self.cost_per_call, self.tokens_per_call)


def test_debug_usage_limits_passing():
    """Debug test to check if usage limits are being passed correctly."""

    # Create a stub agent that costs $0.50 per call
    agent = CostTrackingStubAgent(cost_per_call=0.50, tokens_per_call=100)

    # Create a single step
    step1 = Step.model_validate({"name": "step1", "agent": agent})

    # Create a pipeline with just one step
    pipeline = Pipeline.model_validate({"steps": [step1]})

    # Set usage limit to $0.25 (should be breached)
    usage_limits = UsageLimits(total_cost_usd_limit=0.25)

    # Create Flujo runner with usage limits
    runner = Flujo(pipeline, usage_limits=usage_limits)

    # Run the pipeline and expect it to fail with UsageLimitExceededError
    with pytest.raises(UsageLimitExceededError) as excinfo:
        runner.run("start")

    # Verify the exception details
    error = excinfo.value
    assert "Cost limit of $0.25 exceeded" in str(error)

    # Verify the result contains the expected step history
    result = error.result
    assert result is not None
    assert len(result.step_history) == 1  # One step should have completed
    assert result.total_cost_usd == 0.50  # One step * $0.50 = $0.50


def test_usage_limits_enforcement_simple_steps():
    """Test that usage limits are enforced for simple agent steps."""

    # Create a stub agent that costs $0.50 per call
    agent = CostTrackingStubAgent(cost_per_call=0.50, tokens_per_call=100)

    # Create three simple steps that will each cost $0.50
    step1 = Step.model_validate({"name": "step1", "agent": agent})
    step2 = Step.model_validate({"name": "step2", "agent": agent})
    step3 = Step.model_validate({"name": "step3", "agent": agent})

    # Create a pipeline with these three steps
    pipeline = Pipeline.model_validate({"steps": [step1, step2, step3]})

    # Set usage limit to $1.20 (should be breached on the 3rd step)
    usage_limits = UsageLimits(total_cost_usd_limit=1.20)

    # Create Flujo runner with usage limits
    runner = Flujo(pipeline, usage_limits=usage_limits)

    # Run the pipeline and expect it to fail with UsageLimitExceededError
    with pytest.raises(UsageLimitExceededError) as excinfo:
        runner.run("start")

    # Verify the exception details
    error = excinfo.value
    assert "Cost limit of $1.2 exceeded" in str(error)

    # Verify the result contains the expected step history
    result = error.result
    assert result is not None
    assert (
        len(result.step_history) == 3
    )  # All 3 steps should be in the result (including the breaching step)
    assert result.total_cost_usd == 1.5  # 3 steps * $0.50 = $1.50

    # Verify the step results
    assert result.step_history[0].name == "step1"
    assert result.step_history[0].success is True
    assert result.step_history[0].cost_usd == 0.50

    assert result.step_history[1].name == "step2"
    assert result.step_history[1].success is True
    assert result.step_history[1].cost_usd == 0.50


def test_usage_limits_enforcement_token_limits():
    """Test that token limits are enforced for simple agent steps."""

    # Create a stub agent that uses 100 tokens per call
    agent = CostTrackingStubAgent(cost_per_call=0.10, tokens_per_call=100)

    # Create three simple steps
    step1 = Step.model_validate({"name": "step1", "agent": agent})
    step2 = Step.model_validate({"name": "step2", "agent": agent})
    step3 = Step.model_validate({"name": "step3", "agent": agent})

    # Create a pipeline with these three steps
    pipeline = Pipeline.model_validate({"steps": [step1, step2, step3]})

    # Set token limit to 250 (should be breached on the 3rd step)
    usage_limits = UsageLimits(total_tokens_limit=250)

    # Create Flujo runner with usage limits
    runner = Flujo(pipeline, usage_limits=usage_limits)

    # Run the pipeline and expect it to fail with UsageLimitExceededError
    with pytest.raises(UsageLimitExceededError) as excinfo:
        runner.run("start")

    # Verify the exception details
    error = excinfo.value
    assert "Token limit of 250 exceeded" in str(error)

    # Verify the result contains the expected step history
    result = error.result
    assert result is not None
    assert (
        len(result.step_history) == 3
    )  # All 3 steps should be in the result (including the breaching step)
    assert (
        sum(step.token_counts for step in result.step_history) == 300
    )  # 3 steps * 100 tokens = 300


def test_usage_limits_enforcement_loop_steps():
    """Test that usage limits are enforced in loop steps with simple agent steps."""

    # Create a stub agent that costs $0.30 per call
    agent = CostTrackingStubAgent(cost_per_call=0.30, tokens_per_call=50)

    # Create a simple step
    simple_step = Step.model_validate({"name": "simple_step", "agent": agent})

    # Create a loop step that will iterate multiple times
    from flujo.domain.dsl.loop import LoopStep

    loop_step = LoopStep.model_validate(
        {
            "name": "loop_step",
            "loop_body_pipeline": Pipeline.model_validate({"steps": [simple_step]}),
            "exit_condition_callable": lambda out,
            context: False,  # Never exit, will be limited by usage
            "max_loops": 10,
        }
    )

    # Create a pipeline with the loop step
    pipeline = Pipeline.model_validate({"steps": [loop_step]})

    # Set usage limit to $1.00 (should be breached after ~3 iterations)
    usage_limits = UsageLimits(total_cost_usd_limit=1.00)

    # Create Flujo runner with usage limits
    runner = Flujo(pipeline, usage_limits=usage_limits)

    # Run the pipeline and expect it to fail with UsageLimitExceededError
    with pytest.raises(UsageLimitExceededError) as excinfo:
        runner.run("start")

    # Verify the exception details
    error = excinfo.value
    assert "Cost limit of $1.0 exceeded" in str(error)

    # Verify the result contains the expected step history
    result = error.result
    assert result is not None
    # Should have 1 step (the loop step itself)
    assert len(result.step_history) == 1
    assert result.total_cost_usd == 1.2  # 4 iterations * $0.30 = $1.20


def test_usage_limits_enforcement_complex_steps():
    """Test that usage limits are enforced for complex steps (with plugins/validators)."""

    # Create a stub agent that costs $0.40 per call
    agent = CostTrackingStubAgent(cost_per_call=0.40, tokens_per_call=80)

    # Create a step with a validator (complex step)
    from flujo import Step
    from flujo.validation import validator

    @validator
    def always_pass(output: Any, context: Any) -> bool:
        return True

    complex_step = Step.validate_step(agent, validators=[always_pass])

    # Create a pipeline with the complex step
    pipeline = Pipeline.model_validate({"steps": [complex_step]})

    # Set usage limit to $0.30 (should be breached)
    usage_limits = UsageLimits(total_cost_usd_limit=0.30)

    # Create Flujo runner with usage limits
    runner = Flujo(pipeline, usage_limits=usage_limits)

    # Run the pipeline and expect it to fail with UsageLimitExceededError
    with pytest.raises(UsageLimitExceededError) as excinfo:
        runner.run("start")

    # Verify the exception details
    error = excinfo.value
    assert "Cost limit of $0.3 exceeded" in str(error)

    # Verify the result contains the expected step history
    result = error.result
    assert result is not None
    assert len(result.step_history) == 1  # One step should have completed
    assert result.total_cost_usd == 0.40  # One step * $0.40 = $0.40


def test_usage_limits_no_enforcement_when_no_limits():
    """Test that pipelines run normally when no usage limits are set."""

    # Create a stub agent that costs $0.25 per call
    agent = CostTrackingStubAgent(cost_per_call=0.25, tokens_per_call=60)

    # Create three simple steps
    step1 = Step.model_validate({"name": "step1", "agent": agent})
    step2 = Step.model_validate({"name": "step2", "agent": agent})
    step3 = Step.model_validate({"name": "step3", "agent": agent})

    # Create a pipeline with these three steps
    pipeline = Pipeline.model_validate({"steps": [step1, step2, step3]})

    # Create Flujo runner WITHOUT usage limits
    runner = Flujo(pipeline)

    # Run the pipeline - should complete successfully
    result = runner.run("start")

    # Verify the pipeline completed successfully
    assert result is not None
    assert len(result.step_history) == 3  # All 3 steps should have completed
    assert result.total_cost_usd == 0.75  # 3 steps * $0.25 = $0.75

    # Verify all steps succeeded
    for step_result in result.step_history:
        assert step_result.success is True
        assert step_result.cost_usd == 0.25
        assert step_result.token_counts == 60


def test_usage_limits_enforcement_precise_timing():
    """Test that usage limits are enforced at the exact moment they are exceeded."""

    # Create a stub agent that costs $0.33 per call
    agent = CostTrackingStubAgent(cost_per_call=0.33, tokens_per_call=75)

    # Create steps
    step1 = Step.model_validate({"name": "step1", "agent": agent})
    step2 = Step.model_validate({"name": "step2", "agent": agent})
    step3 = Step.model_validate({"name": "step3", "agent": agent})
    step4 = Step.model_validate({"name": "step4", "agent": agent})

    # Create a pipeline with these steps
    pipeline = Pipeline.model_validate({"steps": [step1, step2, step3, step4]})

    # Set usage limit to $1.00 (should be breached exactly on the 4th step)
    usage_limits = UsageLimits(total_cost_usd_limit=1.00)

    # Create Flujo runner with usage limits
    runner = Flujo(pipeline, usage_limits=usage_limits)

    # Run the pipeline and expect it to fail with UsageLimitExceededError
    with pytest.raises(UsageLimitExceededError) as excinfo:
        runner.run("start")

    # Verify the exception details
    error = excinfo.value
    assert "Cost limit of $1.0 exceeded" in str(error)

    # Verify the result contains the expected step history
    result = error.result
    assert result is not None
    assert (
        len(result.step_history) == 4
    )  # All 4 steps should be in the result (including the breaching step)
    assert result.total_cost_usd == 1.32  # 4 steps * $0.33 = $1.32

    # Verify all completed steps succeeded
    for step_result in result.step_history:
        assert step_result.success is True
        assert step_result.cost_usd == 0.33


def test_debug_usage_limits_detailed():
    """Debug test with detailed output to understand the execution flow."""

    # Create a stub agent that costs $0.50 per call
    agent = CostTrackingStubAgent(cost_per_call=0.50, tokens_per_call=100)

    # Create three simple steps that will each cost $0.50
    step1 = Step.model_validate({"name": "step1", "agent": agent})
    step2 = Step.model_validate({"name": "step2", "agent": agent})
    step3 = Step.model_validate({"name": "step3", "agent": agent})

    # Create a pipeline with these three steps
    pipeline = Pipeline.model_validate({"steps": [step1, step2, step3]})

    # Set usage limit to $1.20 (should be breached on the 3rd step)
    usage_limits = UsageLimits(total_cost_usd_limit=1.20)

    # Create Flujo runner with usage limits
    runner = Flujo(pipeline, usage_limits=usage_limits)

    # Run the pipeline and expect it to fail with UsageLimitExceededError
    try:
        result = runner.run("start")
        # If we reach here, the pipeline completed unexpectedly
        assert False, "Pipeline should have been stopped by usage limits"
    except UsageLimitExceededError as e:
        # Verify the exception details
        assert "Cost limit of $1.2 exceeded" in str(e)

        # Verify the result contains the expected step history
        result = e.result
        assert result is not None
        assert (
            len(result.step_history) == 3
        )  # All 3 steps should be in the result (including the breaching step)
        assert result.total_cost_usd == 1.5  # Total cost should be $1.50 (3 steps * $0.50)

        # Verify each step result
        assert result.step_history[0].name == "step1"
        assert result.step_history[0].cost_usd == 0.50
        assert result.step_history[1].name == "step2"
        assert result.step_history[1].cost_usd == 0.50
        assert result.step_history[2].name == "step3"
        assert result.step_history[2].cost_usd == 0.50
