"""Integration test for cost tracking with mock agents."""

import pytest
from flujo import Flujo, Step, Pipeline
from flujo.domain.models import UsageLimits
from flujo.exceptions import UsageLimitExceededError


class MockAgentWithUsage:
    """A mock agent that simulates pydantic-ai usage tracking."""

    def __init__(
        self,
        prompt_tokens: int = 100,
        completion_tokens: int = 50,
        model_name: str = "gpt-4o",
    ):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self._model_name = f"openai:{model_name}"

    async def run(self, data: str):
        """Simulate a pydantic-ai agent response with usage information."""

        # Create a response that mimics pydantic-ai's AgentRunResult
        class MockResponse:
            def __init__(self, output, usage_info):
                self.output = output
                self._usage_info = usage_info

            def usage(self):
                return self._usage_info

        # Create usage information
        class MockUsage:
            def __init__(self, request_tokens, response_tokens):
                self.request_tokens = request_tokens
                self.response_tokens = response_tokens

        usage_info = MockUsage(self.prompt_tokens, self.completion_tokens)
        return MockResponse(f"Response to: {data}", usage_info)


@pytest.mark.asyncio
async def test_cost_tracking_basic():
    """Test basic cost tracking functionality."""
    # Create a mock agent with usage information
    mock_agent = MockAgentWithUsage(prompt_tokens=100, completion_tokens=50)

    # Create a simple pipeline
    step = Step(name="test_step", agent=mock_agent)
    pipeline = Pipeline.from_step(step)

    # Create and run the Flujo runner
    runner = Flujo(pipeline)
    async for result in runner.run_async("Hello"):
        pass

    # Verify that the step result contains cost information
    assert len(result.step_history) == 1
    assert result.step_history[0].cost_usd > 0
    assert result.step_history[0].token_counts > 0


@pytest.mark.asyncio
async def test_cost_tracking_with_limits():
    """Test cost tracking with usage limits."""
    # Create a mock agent that would exceed the cost limit
    mock_agent = MockAgentWithUsage(prompt_tokens=10000, completion_tokens=5000)

    # Create a simple pipeline
    step = Step(name="expensive_step", agent=mock_agent)
    pipeline = Pipeline.from_step(step)

    # Create and run the Flujo runner with usage limits
    limits = UsageLimits(total_cost_usd_limit=0.01)
    runner = Flujo(pipeline, usage_limits=limits)

    # Run the pipeline and expect it to fail due to cost limit
    with pytest.raises(UsageLimitExceededError):
        async for result in runner.run_async("Hello"):
            pass


@pytest.mark.asyncio
async def test_cost_tracking_with_token_limits():
    """Test cost tracking with token limits."""
    # Create a mock agent that would exceed the token limit
    mock_agent = MockAgentWithUsage(prompt_tokens=1000, completion_tokens=500)

    # Create a simple pipeline
    step = Step(name="token_heavy_step", agent=mock_agent)
    pipeline = Pipeline.from_step(step)

    # Create and run the Flujo runner with token limits
    limits = UsageLimits(total_tokens_limit=100)
    runner = Flujo(pipeline, usage_limits=limits)

    # Run the pipeline and expect it to fail due to token limit
    with pytest.raises(UsageLimitExceededError):
        async for result in runner.run_async("Hello"):
            pass


@pytest.mark.asyncio
async def test_cost_tracking_no_usage_info():
    """Test cost tracking when no usage information is available."""

    # Create a mock agent without usage information
    class SimpleMockAgent:
        async def run(self, data: str):
            return "Simple response"

    mock_agent = SimpleMockAgent()

    # Create a simple pipeline
    step = Step(name="test_step", agent=mock_agent)
    pipeline = Pipeline.from_step(step)

    # Create and run the Flujo runner
    runner = Flujo(pipeline)

    # Run the pipeline - should not fail even without usage info
    async for result in runner.run_async("Hello"):
        pass

    # Verify that the step result exists
    assert len(result.step_history) == 1
    assert result.step_history[0].output == "Simple response"


@pytest.mark.asyncio
async def test_cost_tracking_multiple_steps():
    """Test cost tracking across multiple steps."""
    # Create mock agents with different usage patterns
    agent1 = MockAgentWithUsage(prompt_tokens=100, completion_tokens=50)
    agent2 = MockAgentWithUsage(prompt_tokens=200, completion_tokens=100)

    # Create a pipeline with multiple steps
    step1 = Step(name="step1", agent=agent1)
    step2 = Step(name="step2", agent=agent2)
    pipeline = Pipeline.from_step(step1) >> step2

    # Create and run the Flujo runner
    runner = Flujo(pipeline)

    # Run the pipeline
    async for result in runner.run_async("Hello"):
        pass

    # Verify that both steps have cost information
    assert len(result.step_history) == 2
    assert result.step_history[0].cost_usd > 0
    assert result.step_history[1].cost_usd > 0
    assert result.step_history[0].token_counts > 0
    assert result.step_history[1].token_counts > 0
