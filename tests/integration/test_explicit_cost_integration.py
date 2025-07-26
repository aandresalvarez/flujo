"""Integration tests for ExplicitCostReporter protocol functionality."""

import pytest
from flujo import Flujo
from flujo.domain.dsl.step import Step
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.models import UsageLimits
from flujo.exceptions import UsageLimitExceededError


class MockImageResult:
    """Mock image generation result that implements ExplicitCostReporter."""

    def __init__(self, cost_usd: float, token_counts: int = 0):
        self.cost_usd = cost_usd
        self.token_counts = token_counts


class MockAgentWithExplicitCost:
    """Mock agent that returns objects implementing ExplicitCostReporter."""

    def __init__(self, cost_usd: float, token_counts: int = 0):
        self.cost_usd = cost_usd
        self.token_counts = token_counts
        self.model_id = "openai:gpt-4o"  # Not used for explicit cost

    async def run(self, *args, **kwargs):
        """Return a result that implements ExplicitCostReporter."""
        return MockImageResult(self.cost_usd, self.token_counts)


class TestExplicitCostIntegration:
    """Test ExplicitCostReporter protocol in integration scenarios."""

    def test_end_to_end_success_case(self):
        """Test end-to-end pipeline execution with ExplicitCostReporter."""

        # Create a step with an agent that returns ExplicitCostReporter
        step = Step(
            name="image_generation", agent=MockAgentWithExplicitCost(cost_usd=0.04, token_counts=0)
        )

        # Create a simple pipeline
        pipeline = Pipeline(name="test_explicit_cost_pipeline", steps=[step])

        # Run the pipeline
        runner = Flujo(pipeline)
        result = runner.run("test input")

        # Verify the step result has the correct cost
        assert len(result.step_history) == 1
        step_result = result.step_history[0]
        assert step_result.name == "image_generation"
        assert step_result.cost_usd == 0.04
        assert step_result.token_counts == 0

        # Verify the pipeline result includes the cost
        assert result.total_cost_usd == 0.04

    def test_explicit_cost_with_usage_limits(self):
        """Test that explicit costs are correctly integrated into usage limits."""

        # Create a step with an agent that returns ExplicitCostReporter
        step = Step(
            name="image_generation", agent=MockAgentWithExplicitCost(cost_usd=0.04, token_counts=0)
        )

        # Create a pipeline with usage limits
        pipeline = Pipeline(name="test_explicit_cost_pipeline", steps=[step])

        # Set usage limits below the expected cost
        usage_limits = UsageLimits(total_cost_usd_limit=0.01)

        # Run the pipeline - should fail due to usage limits
        runner = Flujo(pipeline, usage_limits=usage_limits)

        with pytest.raises(UsageLimitExceededError) as exc_info:
            runner.run("test input")

        # Verify the error message indicates cost limit exceeded
        assert "cost" in str(exc_info.value).lower()

    def test_explicit_cost_with_token_counts(self):
        """Test ExplicitCostReporter with token counts."""

        # Create a step with an agent that returns ExplicitCostReporter with tokens
        step = Step(
            name="custom_agent", agent=MockAgentWithExplicitCost(cost_usd=0.25, token_counts=1000)
        )

        # Create a pipeline
        pipeline = Pipeline(name="test_explicit_cost_with_tokens", steps=[step])

        # Run the pipeline
        runner = Flujo(pipeline)
        result = runner.run("test input")

        # Verify the step result has the correct cost and tokens
        assert len(result.step_history) == 1
        step_result = result.step_history[0]
        assert step_result.name == "custom_agent"
        assert step_result.cost_usd == 0.25
        assert step_result.token_counts == 1000

        # Verify the pipeline result includes the cost
        assert result.total_cost_usd == 0.25

    def test_explicit_cost_with_zero_cost(self):
        """Test ExplicitCostReporter with zero cost."""

        # Create a step with an agent that returns ExplicitCostReporter with zero cost
        step = Step(
            name="free_operation", agent=MockAgentWithExplicitCost(cost_usd=0.0, token_counts=500)
        )

        # Create a pipeline
        pipeline = Pipeline(name="test_explicit_cost_zero", steps=[step])

        # Run the pipeline
        runner = Flujo(pipeline)
        result = runner.run("test input")

        # Verify the step result has zero cost but preserves tokens
        assert len(result.step_history) == 1
        step_result = result.step_history[0]
        assert step_result.name == "free_operation"
        assert step_result.cost_usd == 0.0
        assert step_result.token_counts == 500

        # Verify the pipeline result has zero cost
        assert result.total_cost_usd == 0.0

    def test_explicit_cost_with_negative_cost(self):
        """Test ExplicitCostReporter with negative cost (edge case)."""

        # Create a step with an agent that returns ExplicitCostReporter with negative cost
        step = Step(
            name="credit_operation",
            agent=MockAgentWithExplicitCost(cost_usd=-0.10, token_counts=100),
        )

        # Create a pipeline
        pipeline = Pipeline(name="test_explicit_cost_negative", steps=[step])

        # Run the pipeline
        runner = Flujo(pipeline)
        result = runner.run("test input")

        # Verify the step result preserves the negative cost
        assert len(result.step_history) == 1
        step_result = result.step_history[0]
        assert step_result.name == "credit_operation"
        assert step_result.cost_usd == -0.10
        assert step_result.token_counts == 100

        # Verify the pipeline result includes the negative cost
        assert result.total_cost_usd == -0.10

    def test_explicit_cost_with_multiple_steps(self):
        """Test ExplicitCostReporter with multiple steps in a pipeline."""

        # Create multiple steps with different explicit costs
        step1 = Step(
            name="image_generation", agent=MockAgentWithExplicitCost(cost_usd=0.04, token_counts=0)
        )

        step2 = Step(
            name="text_processing", agent=MockAgentWithExplicitCost(cost_usd=0.15, token_counts=800)
        )

        step3 = Step(
            name="data_analysis", agent=MockAgentWithExplicitCost(cost_usd=0.08, token_counts=200)
        )

        # Create a pipeline with multiple steps
        pipeline = Pipeline(name="test_explicit_cost_multiple", steps=[step1, step2, step3])

        # Run the pipeline
        runner = Flujo(pipeline)
        result = runner.run("test input")

        # Verify all step results have correct costs
        assert len(result.step_history) == 3

        # Check first step
        assert result.step_history[0].name == "image_generation"
        assert result.step_history[0].cost_usd == 0.04
        assert result.step_history[0].token_counts == 0

        # Check second step
        assert result.step_history[1].name == "text_processing"
        assert result.step_history[1].cost_usd == 0.15
        assert result.step_history[1].token_counts == 800

        # Check third step
        assert result.step_history[2].name == "data_analysis"
        assert result.step_history[2].cost_usd == 0.08
        assert result.step_history[2].token_counts == 200

        # Verify the total cost is the sum of all steps
        expected_total = 0.04 + 0.15 + 0.08
        assert result.total_cost_usd == expected_total

    def test_explicit_cost_protocol_priority_over_usage_method(self):
        """Test that ExplicitCostReporter protocol takes priority over .usage() method."""

        # Create a mock agent that returns an object with BOTH ExplicitCostReporter and .usage()
        class MockAgentWithBoth:
            def __init__(self):
                self.model_id = "openai:gpt-4o"

            async def run(self, *args, **kwargs):
                class MockResponseWithBoth:
                    def __init__(self):
                        self.cost_usd = 1.23  # Protocol cost
                        self.token_counts = 200  # Protocol tokens

                    def usage(self):
                        class MockUsage:
                            def __init__(self):
                                self.request_tokens = 100
                                self.response_tokens = 50

                        return MockUsage()

                return MockResponseWithBoth()

        # Create a step with this agent
        step = Step(name="mixed_cost_agent", agent=MockAgentWithBoth())

        # Create a pipeline
        pipeline = Pipeline(name="test_protocol_priority", steps=[step])

        # Run the pipeline
        runner = Flujo(pipeline)
        result = runner.run("test input")

        # Verify that the protocol cost is used, not the .usage() method cost
        assert len(result.step_history) == 1
        step_result = result.step_history[0]
        assert step_result.name == "mixed_cost_agent"
        assert step_result.cost_usd == 1.23  # Should use protocol cost
        assert step_result.token_counts == 200  # Should use protocol tokens

        # Verify the pipeline result includes the protocol cost
        assert result.total_cost_usd == 1.23

    def test_explicit_cost_with_none_values(self):
        """Test ExplicitCostReporter with None values (edge case)."""

        # Create a mock agent that returns an object with None values
        class MockAgentWithNoneValues:
            def __init__(self):
                self.model_id = "openai:gpt-4o"

            async def run(self, *args, **kwargs):
                class MockResponseWithNone:
                    def __init__(self):
                        self.cost_usd = None
                        self.token_counts = None

                return MockResponseWithNone()

        # Create a step with this agent
        step = Step(name="none_values_agent", agent=MockAgentWithNoneValues())

        # Create a pipeline
        pipeline = Pipeline(name="test_none_values", steps=[step])

        # Run the pipeline
        runner = Flujo(pipeline)
        result = runner.run("test input")

        # Verify that None values are handled gracefully
        assert len(result.step_history) == 1
        step_result = result.step_history[0]
        assert step_result.name == "none_values_agent"
        assert step_result.cost_usd == 0.0  # Should default to 0.0
        assert step_result.token_counts == 0  # Should default to 0

        # Verify the pipeline result has zero cost
        assert result.total_cost_usd == 0.0

    def test_explicit_cost_regression_with_usage_method(self):
        """Test that existing .usage() method functionality is not broken."""

        # Create a mock agent that returns an object with only .usage() method
        class MockAgentWithUsageOnly:
            def __init__(self):
                self.model_id = "openai:gpt-4o"

            async def run(self, *args, **kwargs):
                class MockResponseWithUsageOnly:
                    def __init__(self):
                        self.output = "test response"

                    def usage(self):
                        class MockUsage:
                            def __init__(self):
                                self.request_tokens = 100
                                self.response_tokens = 50

                        return MockUsage()

                return MockResponseWithUsageOnly()

        # Create a step with this agent
        step = Step(name="usage_only_agent", agent=MockAgentWithUsageOnly())

        # Create a pipeline
        pipeline = Pipeline(name="test_usage_method_regression", steps=[step])

        # Run the pipeline
        runner = Flujo(pipeline)
        result = runner.run("test input")

        # Verify that the .usage() method still works correctly
        assert len(result.step_history) == 1
        step_result = result.step_history[0]
        assert step_result.name == "usage_only_agent"
        assert step_result.cost_usd > 0.0  # Should calculate cost from tokens
        assert step_result.token_counts > 0  # Should have token counts

        # Verify the pipeline result includes the calculated cost
        assert result.total_cost_usd > 0.0
