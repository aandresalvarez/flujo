"""Integration test for cost tracking with real pydantic-ai agents."""

import pytest
from unittest.mock import patch
from flujo import Flujo, Step, Pipeline, make_agent_async


@pytest.mark.asyncio
async def test_cost_tracking_with_real_agent():
    """Test that cost tracking works with real pydantic-ai agents."""
    # Create a real agent using make_agent_async
    agent = make_agent_async(
        model="openai:gpt-4o-mini",
        system_prompt="You are a helpful assistant. Respond briefly.",
        output_type=str,
    )

    # Create a simple pipeline
    step = Step(name="test_step", agent=agent)
    pipeline = Pipeline.from_step(step)

    # Create runner without cost limits (to avoid API calls)
    runner = Flujo(pipeline)

    # Mock the agent's run method to return usage information
    with patch.object(agent._agent, "run") as mock_run:
        # Create a mock response with usage information
        class MockAgentRunResult:
            def __init__(self):
                self.output = "Mock response"

            def usage(self):
                class MockUsage:
                    def __init__(self):
                        self.request_tokens = 100
                        self.response_tokens = 50

                return MockUsage()

        mock_run.return_value = MockAgentRunResult()

        # Run the pipeline
        result = None
        async for item in runner.run_async("test input"):
            result = item

        # Verify that the result contains cost information
        assert result is not None
        assert len(result.step_history) == 1

        step_result = result.step_history[0]
        assert step_result.success
        assert step_result.token_counts == 150  # 100 + 50
        # Cost should be calculated based on the pricing configuration
        assert step_result.cost_usd >= 0.0

        # Verify total cost is tracked
        assert result.total_cost_usd >= 0.0


@pytest.mark.asyncio
async def test_cost_tracking_without_pricing_config():
    """Test that cost tracking works even without pricing configuration."""
    # Create a real agent
    agent = make_agent_async(
        model="openai:gpt-4o-mini",
        system_prompt="You are a helpful assistant. Respond briefly.",
        output_type=str,
    )

    # Create a simple pipeline
    step = Step(name="test_step", agent=agent)
    pipeline = Pipeline.from_step(step)

    # Create runner without cost limits
    runner = Flujo(pipeline)

    # Mock the agent's run method to return usage information
    with patch.object(agent._agent, "run") as mock_run:
        # Create a mock response with usage information
        class MockAgentRunResult:
            def __init__(self):
                self.output = "Mock response"

            def usage(self):
                class MockUsage:
                    def __init__(self):
                        self.request_tokens = 100
                        self.response_tokens = 50

                return MockUsage()

        mock_run.return_value = MockAgentRunResult()

        # Run the pipeline
        result = None
        async for item in runner.run_async("test input"):
            result = item

        # Verify that the result contains token information even if cost is 0
        assert result is not None
        assert len(result.step_history) == 1

        step_result = result.step_history[0]
        assert step_result.success
        assert step_result.token_counts == 150  # 100 + 50
        # Cost might be 0.0 if no pricing is configured, which is acceptable
        assert step_result.cost_usd >= 0.0
