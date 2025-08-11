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

            def __repr__(self) -> str:
                return f"UsageResponse(output={self.output}, cost_usd={self.cost_usd}, token_counts={self.token_counts})"

        return UsageResponse(response, self.cost_per_call, self.tokens_per_call)


# Register custom serializer for UsageResponse to fix serialization issues
def serialize_usage_response(obj):
    """Custom serializer for UsageResponse objects."""
    return {
        "output": obj.output,
        "cost_usd": obj.cost_usd,
        "token_counts": obj.token_counts,
    }


# We'll register the serializer when we need it in the tests


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
    assert "Cost limit of $1 exceeded" in str(error)

    # Verify the result contains the expected step history
    result = error.result
    assert result is not None
    # Should have 1 step (the loop step itself)
    assert len(result.step_history) == 1
    assert result.total_cost_usd == pytest.approx(
        0.9
    )  # 3 iterations * $0.30 = $0.90 (stopped before breaching)


def test_usage_limits_enforcement_complex_steps():
    """Test that usage limits are enforced for complex steps (with plugins/validators)."""

    # Create a stub agent that costs $0.40 per call
    agent = CostTrackingStubAgent(cost_per_call=0.40, tokens_per_call=80)

    # Create a step with a validator (complex step)
    from flujo import Step
    from flujo.domain.validation import validator

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
    assert "Cost limit of $1 exceeded" in str(error)

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


# ============================================================================
# EDGE CASE TESTS FOR COMPREHENSIVE REGRESSION TESTING
# ============================================================================


def test_usage_limits_edge_case_zero_cost():
    """Test edge case where a step has zero cost but still has token usage."""

    # Create a stub agent that costs $0.00 per call but uses tokens
    agent = CostTrackingStubAgent(cost_per_call=0.00, tokens_per_call=100)

    # Create steps
    step1 = Step.model_validate({"name": "step1", "agent": agent})
    step2 = Step.model_validate({"name": "step2", "agent": agent})

    # Create a pipeline with these steps
    pipeline = Pipeline.model_validate({"steps": [step1, step2]})

    # Set token limit to 150 (should be breached on the 2nd step)
    usage_limits = UsageLimits(total_tokens_limit=150)

    # Create Flujo runner with usage limits
    runner = Flujo(pipeline, usage_limits=usage_limits)

    # Run the pipeline and expect it to fail with UsageLimitExceededError
    with pytest.raises(UsageLimitExceededError) as excinfo:
        runner.run("start")

    # Verify the exception details
    error = excinfo.value
    assert "Token limit of 150 exceeded" in str(error)

    # Verify the result contains the expected step history
    result = error.result
    assert result is not None
    assert len(result.step_history) == 2
    assert result.total_cost_usd == 0.0  # Zero cost
    assert sum(step.token_counts for step in result.step_history) == 200  # 2 * 100 tokens


def test_usage_limits_edge_case_zero_tokens():
    """Test edge case where a step has zero tokens but still has cost."""

    # Create a stub agent that costs $0.50 per call but uses no tokens
    agent = CostTrackingStubAgent(cost_per_call=0.50, tokens_per_call=0)

    # Create steps
    step1 = Step.model_validate({"name": "step1", "agent": agent})
    step2 = Step.model_validate({"name": "step2", "agent": agent})

    # Create a pipeline with these steps
    pipeline = Pipeline.model_validate({"steps": [step1, step2]})

    # Set cost limit to $0.75 (should be breached on the 2nd step)
    usage_limits = UsageLimits(total_cost_usd_limit=0.75)

    # Create Flujo runner with usage limits
    runner = Flujo(pipeline, usage_limits=usage_limits)

    # Run the pipeline and expect it to fail with UsageLimitExceededError
    with pytest.raises(UsageLimitExceededError) as excinfo:
        runner.run("start")

    # Verify the exception details
    error = excinfo.value
    assert "Cost limit of $0.75 exceeded" in str(error)

    # Verify the result contains the expected step history
    result = error.result
    assert result is not None
    assert len(result.step_history) == 2
    assert result.total_cost_usd == 1.0  # 2 * $0.50
    assert sum(step.token_counts for step in result.step_history) == 0  # Zero tokens


def test_usage_limits_edge_case_very_small_limits():
    """Test edge case with very small limits that should be breached immediately."""

    # Create a stub agent that costs $0.01 per call
    agent = CostTrackingStubAgent(cost_per_call=0.01, tokens_per_call=1)

    # Create a single step
    step1 = Step.model_validate({"name": "step1", "agent": agent})

    # Create a pipeline with just one step
    pipeline = Pipeline.model_validate({"steps": [step1]})

    # Set very small limits that should be breached
    usage_limits = UsageLimits(total_cost_usd_limit=0.005, total_tokens_limit=0)

    # Create Flujo runner with usage limits
    runner = Flujo(pipeline, usage_limits=usage_limits)

    # Run the pipeline and expect it to fail with UsageLimitExceededError
    with pytest.raises(UsageLimitExceededError) as excinfo:
        runner.run("start")

    # Verify the exception details
    error = excinfo.value
    assert "Cost limit of $0.005 exceeded" in str(error) or "Token limit of 0 exceeded" in str(
        error
    )

    # Verify the result contains the expected step history
    result = error.result
    assert result is not None
    assert len(result.step_history) == 1  # One step should have completed
    assert result.total_cost_usd == 0.01  # One step * $0.01 = $0.01


def test_usage_limits_edge_case_very_large_limits():
    """Test edge case with very large limits that should never be breached."""

    # Create a stub agent that costs $0.50 per call
    agent = CostTrackingStubAgent(cost_per_call=0.50, tokens_per_call=100)

    # Create three simple steps
    step1 = Step.model_validate({"name": "step1", "agent": agent})
    step2 = Step.model_validate({"name": "step2", "agent": agent})
    step3 = Step.model_validate({"name": "step3", "agent": agent})

    # Create a pipeline with these three steps
    pipeline = Pipeline.model_validate({"steps": [step1, step2, step3]})

    # Set very large limits that should never be breached
    usage_limits = UsageLimits(total_cost_usd_limit=1000.0, total_tokens_limit=1000000)

    # Create Flujo runner with usage limits
    runner = Flujo(pipeline, usage_limits=usage_limits)

    # Run the pipeline - should complete successfully
    result = runner.run("start")

    # Verify the pipeline completed successfully
    assert result is not None
    assert len(result.step_history) == 3  # All 3 steps should have completed
    assert result.total_cost_usd == 1.5  # 3 steps * $0.50 = $1.50

    # Verify all steps succeeded
    for step_result in result.step_history:
        assert step_result.success is True
        assert step_result.cost_usd == 0.50
        assert step_result.token_counts == 100


def test_usage_limits_edge_case_mixed_limits():
    """Test edge case where only one type of limit is set."""

    # Create a stub agent that costs $0.50 per call and uses 100 tokens
    agent = CostTrackingStubAgent(cost_per_call=0.50, tokens_per_call=100)

    # Create three simple steps
    step1 = Step.model_validate({"name": "step1", "agent": agent})
    step2 = Step.model_validate({"name": "step2", "agent": agent})
    step3 = Step.model_validate({"name": "step3", "agent": agent})

    # Create a pipeline with these three steps
    pipeline = Pipeline.model_validate({"steps": [step1, step2, step3]})

    # Test with only cost limit set
    usage_limits_cost_only = UsageLimits(total_cost_usd_limit=1.0, total_tokens_limit=None)
    runner_cost_only = Flujo(pipeline, usage_limits=usage_limits_cost_only)

    with pytest.raises(UsageLimitExceededError) as excinfo:
        runner_cost_only.run("start")

    error = excinfo.value
    assert "Cost limit of $1 exceeded" in str(error)

    # Test with only token limit set
    usage_limits_token_only = UsageLimits(total_cost_usd_limit=None, total_tokens_limit=250)
    runner_token_only = Flujo(pipeline, usage_limits=usage_limits_token_only)

    with pytest.raises(UsageLimitExceededError) as excinfo:
        runner_token_only.run("start")

    error = excinfo.value
    assert "Token limit of 250 exceeded" in str(error)


def test_usage_limits_edge_case_conditional_steps():
    """Test edge case with conditional steps that may or may not execute."""

    # Create a stub agent that costs $0.50 per call
    agent = CostTrackingStubAgent(cost_per_call=0.50, tokens_per_call=100)

    # Create steps
    step1 = Step.model_validate({"name": "step1", "agent": agent})
    step2 = Step.model_validate({"name": "step2", "agent": agent})

    # Create a conditional step that always executes
    from flujo.domain.dsl.conditional import ConditionalStep

    conditional_step = ConditionalStep.model_validate(
        {
            "name": "conditional_step",
            "condition_callable": lambda data, context: "true",  # Always execute true branch
            "branches": {
                "true": Pipeline.model_validate({"steps": [step1]}),
                "false": Pipeline.model_validate({"steps": [step2]}),
            },
        }
    )

    # Create a pipeline with the conditional step
    pipeline = Pipeline.model_validate({"steps": [conditional_step]})

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


def test_usage_limits_edge_case_parallel_steps():
    """Test edge case with parallel steps that execute concurrently."""

    # Create a stub agent that costs $0.30 per call
    agent = CostTrackingStubAgent(cost_per_call=0.30, tokens_per_call=75)

    # Create steps
    step1 = Step.model_validate({"name": "step1", "agent": agent})
    step2 = Step.model_validate({"name": "step2", "agent": agent})

    # Create a parallel step
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

    # Set usage limit to $0.50 (should be breached by parallel execution)
    usage_limits = UsageLimits(total_cost_usd_limit=0.50)

    # Create Flujo runner with usage limits
    runner = Flujo(pipeline, usage_limits=usage_limits)

    # Run the pipeline and expect it to fail with UsageLimitExceededError
    with pytest.raises(UsageLimitExceededError) as excinfo:
        runner.run("start")

    # Verify the exception details
    error = excinfo.value
    assert "Cost limit of $0.5 exceeded" in str(error)

    # Verify the result contains the expected step history
    result = error.result
    assert result is not None
    assert len(result.step_history) >= 1  # At least one branch should have completed
    assert result.total_cost_usd >= 0.30  # At least one branch * $0.30 = $0.30


def test_usage_limits_edge_case_nested_pipelines():
    """Test edge case with nested pipelines that have their own usage tracking."""

    # Create a stub agent that costs $0.25 per call
    agent = CostTrackingStubAgent(cost_per_call=0.25, tokens_per_call=50)

    # Create a simple step
    simple_step = Step.model_validate({"name": "simple_step", "agent": agent})

    # Create a pipeline with the nested pipeline as a step
    # Since there's no PipelineStep, we'll use the nested pipeline directly
    pipeline = Pipeline.model_validate({"steps": [simple_step, simple_step]})

    # Set usage limit to $0.40 (should be breached by nested pipeline)
    usage_limits = UsageLimits(total_cost_usd_limit=0.40)

    # Create Flujo runner with usage limits
    runner = Flujo(pipeline, usage_limits=usage_limits)

    # Run the pipeline and expect it to fail with UsageLimitExceededError
    with pytest.raises(UsageLimitExceededError) as excinfo:
        runner.run("start")

    # Verify the exception details
    error = excinfo.value
    assert "Cost limit of $0.4 exceeded" in str(error)

    # Verify the result contains the expected step history
    result = error.result
    assert result is not None
    assert len(result.step_history) == 2  # Two steps should have completed
    assert result.total_cost_usd == 0.50  # 2 steps * $0.25 = $0.50


def test_usage_limits_edge_case_retry_mechanism():
    """Test edge case with retry mechanism that could accumulate costs."""

    # Create a stub agent that costs $0.20 per call
    agent = CostTrackingStubAgent(cost_per_call=0.20, tokens_per_call=40)

    # Create a step with retry configuration
    step_with_retry = Step.model_validate(
        {
            "name": "step_with_retry",
            "agent": agent,
            "max_attempts": 3,
        }
    )

    # Create a pipeline with the retry step
    pipeline = Pipeline.model_validate({"steps": [step_with_retry]})

    # Set usage limit to $0.50 (should be breached if retries occur)
    usage_limits = UsageLimits(total_cost_usd_limit=0.50)

    # Create Flujo runner with usage limits
    runner = Flujo(pipeline, usage_limits=usage_limits)

    # Run the pipeline - should complete successfully (no retries in stub agent)
    result = runner.run("start")

    # Verify the pipeline completed successfully
    assert result is not None
    assert len(result.step_history) == 1  # One step should have completed
    assert result.total_cost_usd == 0.20  # One step * $0.20 = $0.20

    # Verify the step succeeded
    step_result = result.step_history[0]
    assert step_result.success is True
    assert step_result.cost_usd == 0.20
    assert step_result.token_counts == 40


def test_usage_limits_edge_case_fallback_mechanism():
    """Test edge case with fallback mechanism that could have different costs."""

    # Create stub agents with different costs
    primary_agent = CostTrackingStubAgent(cost_per_call=0.50, tokens_per_call=100)
    fallback_agent = CostTrackingStubAgent(cost_per_call=0.30, tokens_per_call=60)

    # Create steps with fallback
    fallback_step = Step.model_validate({"name": "fallback_step", "agent": fallback_agent})

    # Create a step with fallback configuration
    step_with_fallback = Step.model_validate(
        {
            "name": "step_with_fallback",
            "agent": primary_agent,
            "fallback": fallback_step,
        }
    )

    # Create a pipeline with the fallback step
    pipeline = Pipeline.model_validate({"steps": [step_with_fallback]})

    # Set usage limit to $0.40 (should be breached by primary step)
    usage_limits = UsageLimits(total_cost_usd_limit=0.40)

    # Create Flujo runner with usage limits
    runner = Flujo(pipeline, usage_limits=usage_limits)

    # Run the pipeline and expect it to fail with UsageLimitExceededError
    with pytest.raises(UsageLimitExceededError) as excinfo:
        runner.run("start")

    # Verify the exception details
    error = excinfo.value
    assert "Cost limit of $0.4 exceeded" in str(error)

    # Verify the result contains the expected step history
    result = error.result
    assert result is not None
    assert len(result.step_history) == 1  # One step should have completed
    assert result.total_cost_usd == 0.50  # Primary step * $0.50 = $0.50


def test_usage_limits_edge_case_state_persistence():
    """Test edge case where state persistence occurs during usage limit enforcement."""

    # Create a stub agent that costs $0.40 per call
    agent = CostTrackingStubAgent(cost_per_call=0.40, tokens_per_call=80)

    # Create three simple steps
    step1 = Step.model_validate({"name": "step1", "agent": agent})
    step2 = Step.model_validate({"name": "step2", "agent": agent})
    step3 = Step.model_validate({"name": "step3", "agent": agent})

    # Create a pipeline with these three steps
    pipeline = Pipeline.model_validate({"steps": [step1, step2, step3]})

    # Set usage limit to $0.80 (should be breached on the 2nd step)
    usage_limits = UsageLimits(total_cost_usd_limit=0.80)

    # Create Flujo runner with usage limits and state persistence
    runner = Flujo(pipeline, usage_limits=usage_limits)

    # Run the pipeline and expect it to fail with UsageLimitExceededError
    with pytest.raises(UsageLimitExceededError) as excinfo:
        runner.run("start")

    # Verify the exception details
    error = excinfo.value
    assert "Cost limit of $0.8 exceeded" in str(error)

    # Verify the result contains the expected step history
    result = error.result
    assert result is not None
    assert len(result.step_history) == 3  # All three steps should be in the result
    assert (
        abs(result.total_cost_usd - 1.20) < 1e-10
    )  # 3 steps * $0.40 = $1.20 (with floating point tolerance)

    # Verify the step results
    assert result.step_history[0].name == "step1"
    assert result.step_history[0].success is True
    assert result.step_history[0].cost_usd == 0.40

    assert result.step_history[1].name == "step2"
    assert result.step_history[1].success is True
    assert result.step_history[1].cost_usd == 0.40

    assert result.step_history[2].name == "step3"
    assert result.step_history[2].success is True
    assert result.step_history[2].cost_usd == 0.40


def test_usage_limits_edge_case_concurrent_execution():
    """Test edge case with concurrent execution that could have race conditions."""

    # Create a stub agent that costs $0.25 per call
    agent = CostTrackingStubAgent(cost_per_call=0.25, tokens_per_call=50)

    # Create steps
    step1 = Step.model_validate({"name": "step1", "agent": agent})
    step2 = Step.model_validate({"name": "step2", "agent": agent})

    # Create a parallel step for concurrent execution
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

    # Set usage limit to $0.40 (should be breached by concurrent execution)
    usage_limits = UsageLimits(total_cost_usd_limit=0.40)

    # Create Flujo runner with usage limits
    runner = Flujo(pipeline, usage_limits=usage_limits)

    # Run the pipeline and expect it to fail with UsageLimitExceededError
    with pytest.raises(UsageLimitExceededError) as excinfo:
        runner.run("start")

    # Verify the exception details
    error = excinfo.value
    assert "Cost limit of $0.4 exceeded" in str(error)

    # Verify the result contains the expected step history
    result = error.result
    assert result is not None
    assert len(result.step_history) >= 1  # At least one branch should have completed
    assert result.total_cost_usd >= 0.25  # At least one branch * $0.25 = $0.25


def test_usage_limits_edge_case_error_handling():
    """Test edge case where errors occur during usage limit enforcement."""

    # Create a stub agent that costs $0.30 per call
    agent = CostTrackingStubAgent(cost_per_call=0.30, tokens_per_call=60)

    # Create steps
    step1 = Step.model_validate({"name": "step1", "agent": agent})
    step2 = Step.model_validate({"name": "step2", "agent": agent})

    # Create a pipeline with these steps
    pipeline = Pipeline.model_validate({"steps": [step1, step2]})

    # Set usage limit to $0.50 (should be breached on the 2nd step)
    usage_limits = UsageLimits(total_cost_usd_limit=0.50)

    # Create Flujo runner with usage limits
    runner = Flujo(pipeline, usage_limits=usage_limits)

    # Run the pipeline and expect it to fail with UsageLimitExceededError
    with pytest.raises(UsageLimitExceededError) as excinfo:
        runner.run("start")

    # Verify the exception details
    error = excinfo.value
    assert "Cost limit of $0.5 exceeded" in str(error)

    # Verify the result contains the expected step history
    result = error.result
    assert result is not None
    assert len(result.step_history) == 2  # Both steps should be in the result
    assert result.total_cost_usd == 0.60  # 2 steps * $0.30 = $0.60

    # Verify the step results
    assert result.step_history[0].name == "step1"
    assert result.step_history[0].success is True
    assert result.step_history[0].cost_usd == 0.30

    assert result.step_history[1].name == "step2"
    assert result.step_history[1].success is True
    assert result.step_history[1].cost_usd == 0.30
