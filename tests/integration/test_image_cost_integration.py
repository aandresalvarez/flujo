"""Integration tests for image generation cost tracking."""

import pytest
from unittest.mock import patch, Mock
from flujo import Flujo, Step, Pipeline
from flujo.infra.agents import make_agent_async
from flujo.domain.models import UsageLimits
from flujo.exceptions import UsageLimitExceededError


class MockImageGenerationAgent:
    """A mock agent that simulates DALL-E 3 image generation."""

    def __init__(self, image_count: int = 1, quality: str = "standard", size: str = "1024x1024"):
        self.image_count = image_count
        self.quality = quality
        self.size = size
        self.model_id = "openai:dall-e-3"

    async def run(self, data: str):
        """Simulate a DALL-E 3 image generation response with usage information."""
        
        # Create a response object that mimics pydantic-ai's AgentRunResult
        class AgentResponse:
            def __init__(self, image_count, quality, size):
                self.output = f"Generated {image_count} image(s) with quality {quality} and size {size}"
                self._image_count = image_count
                self._quality = quality
                self._size = size

            def usage(self):
                class UsageInfo:
                    def __init__(self, image_count):
                        self.details = {"images": image_count}
                        self.cost_usd = None  # Will be set by post-processor

                return UsageInfo(self._image_count)

        return AgentResponse(self.image_count, self.quality, self.size)


class TestImageCostIntegration:
    """Test image generation cost tracking in integration scenarios."""

    @pytest.mark.asyncio
    async def test_image_cost_tracking_end_to_end_success(self):
        """Test end-to-end image cost tracking with valid pricing configuration."""
        
        # Create a mock agent that simulates DALL-E 3
        agent = MockImageGenerationAgent(image_count=2, quality="standard", size="1024x1024")
        
        # Create a simple pipeline
        step = Step(name="image_generation", agent=agent)
        pipeline = Pipeline.from_step(step)
        
        # Create runner without cost limits
        runner = Flujo(pipeline)
        
        # Run the pipeline
        result = None
        async for item in runner.run_async("Generate a beautiful landscape"):
            result = item
        
        # Verify that the result contains cost information
        assert result is not None
        assert len(result.step_history) == 1
        
        step_result = result.step_history[0]
        assert step_result.success
        # Cost should be calculated based on the pricing configuration
        # 2 images * $0.040 = $0.080
        assert step_result.cost_usd == 0.080
        
        # Verify total cost is tracked
        assert result.total_cost_usd == 0.080

    @pytest.mark.asyncio
    async def test_image_cost_tracking_with_hd_quality(self):
        """Test image cost tracking with HD quality."""
        
        # Create a mock agent with HD quality
        agent = MockImageGenerationAgent(image_count=1, quality="hd", size="1024x1024")
        
        # Create a simple pipeline
        step = Step(name="image_generation", agent=agent)
        pipeline = Pipeline.from_step(step)
        
        # Create runner without cost limits
        runner = Flujo(pipeline)
        
        # Run the pipeline
        result = None
        async for item in runner.run_async("Generate a high-quality portrait"):
            result = item
        
        # Verify that the result contains cost information
        assert result is not None
        assert len(result.step_history) == 1
        
        step_result = result.step_history[0]
        assert step_result.success
        # Cost should be calculated based on HD pricing
        # 1 image * $0.080 = $0.080
        assert step_result.cost_usd == 0.080

    @pytest.mark.asyncio
    async def test_image_cost_tracking_with_large_size(self):
        """Test image cost tracking with large image size."""
        
        # Create a mock agent with large size
        agent = MockImageGenerationAgent(image_count=1, quality="standard", size="1792x1024")
        
        # Create a simple pipeline
        step = Step(name="image_generation", agent=agent)
        pipeline = Pipeline.from_step(step)
        
        # Create runner without cost limits
        runner = Flujo(pipeline)
        
        # Run the pipeline
        result = None
        async for item in runner.run_async("Generate a wide landscape"):
            result = item
        
        # Verify that the result contains cost information
        assert result is not None
        assert len(result.step_history) == 1
        
        step_result = result.step_history[0]
        assert step_result.success
        # Cost should be calculated based on large size pricing
        # 1 image * $0.080 = $0.080
        assert step_result.cost_usd == 0.080

    @pytest.mark.asyncio
    async def test_image_cost_tracking_with_usage_limits(self):
        """Test that image costs are correctly integrated into usage limits."""
        
        # Create a mock agent
        agent = MockImageGenerationAgent(image_count=1, quality="standard", size="1024x1024")
        
        # Create a simple pipeline
        step = Step(name="image_generation", agent=agent)
        pipeline = Pipeline.from_step(step)
        
        # Set usage limits below the expected cost
        usage_limits = UsageLimits(total_cost_usd_limit=0.01)  # $0.01 limit
        
        # Run the pipeline - should fail due to usage limits
        runner = Flujo(pipeline, usage_limits=usage_limits)
        
        with pytest.raises(UsageLimitExceededError) as exc_info:
            async for item in runner.run_async("Generate an image"):
                pass
        
        # Verify the error message indicates cost limit exceeded
        assert "cost" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_image_cost_tracking_with_multiple_steps(self):
        """Test image cost tracking with multiple steps in a pipeline."""
        
        # Create multiple steps with different image configurations
        agent1 = MockImageGenerationAgent(image_count=1, quality="standard", size="1024x1024")
        agent2 = MockImageGenerationAgent(image_count=2, quality="hd", size="1024x1024")
        
        # Create a pipeline with multiple steps
        step1 = Step(name="image_generation_1", agent=agent1)
        step2 = Step(name="image_generation_2", agent=agent2)
        pipeline = Pipeline.from_step(step1) >> step2
        
        # Create runner without cost limits
        runner = Flujo(pipeline)
        
        # Run the pipeline
        result = None
        async for item in runner.run_async("Generate multiple images"):
            result = item
        
        # Verify all step results have correct costs
        assert len(result.step_history) == 2
        
        # Check first step: 1 image * $0.040 = $0.040
        assert result.step_history[0].name == "image_generation_1"
        assert result.step_history[0].cost_usd == 0.040
        
        # Check second step: 2 images * $0.080 = $0.160
        assert result.step_history[1].name == "image_generation_2"
        assert result.step_history[1].cost_usd == 0.160
        
        # Verify the total cost is the sum of all steps
        expected_total = 0.040 + 0.160
        assert result.total_cost_usd == expected_total

    @pytest.mark.asyncio
    async def test_image_cost_tracking_with_missing_pricing(self):
        """Test image cost tracking when pricing is not configured."""
        
        # Create a mock agent with a model that has no pricing
        class MockAgentWithoutPricing:
            async def run(self, data: str):
                class AgentResponse:
                    def __init__(self):
                        self.output = "Generated image without pricing"
                        
                    def usage(self):
                        class UsageInfo:
                            def __init__(self):
                                self.details = {"images": 1}
                                self.cost_usd = None
                        
                        return UsageInfo()
                
                return AgentResponse()
        
        agent = MockAgentWithoutPricing()
        
        # Create a simple pipeline
        step = Step(name="image_generation", agent=agent)
        pipeline = Pipeline.from_step(step)
        
        # Create runner without cost limits
        runner = Flujo(pipeline)
        
        # Run the pipeline
        result = None
        async for item in runner.run_async("Generate an image"):
            result = item
        
        # Verify that the result contains cost information
        assert result is not None
        assert len(result.step_history) == 1
        
        step_result = result.step_history[0]
        assert step_result.success
        # Cost should be 0.0 when pricing is not configured
        assert step_result.cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_image_cost_tracking_regression_with_chat_models(self):
        """Test that adding image cost tracking doesn't interfere with chat models."""
        
        # Create a mock chat agent (non-image model)
        class MockChatAgent:
            async def run(self, data: str):
                class AgentResponse:
                    def __init__(self):
                        self.output = "Chat response"
                        
                    def usage(self):
                        class UsageInfo:
                            def __init__(self):
                                self.request_tokens = 100
                                self.response_tokens = 50
                        
                        return UsageInfo()
                
                return AgentResponse()
        
        agent = MockChatAgent()
        
        # Create a simple pipeline
        step = Step(name="chat_step", agent=agent)
        pipeline = Pipeline.from_step(step)
        
        # Create runner without cost limits
        runner = Flujo(pipeline)
        
        # Run the pipeline
        result = None
        async for item in runner.run_async("Hello"):
            result = item
        
        # Verify that the result contains cost information
        assert result is not None
        assert len(result.step_history) == 1
        
        step_result = result.step_history[0]
        assert step_result.success
        # Chat models should still work normally (cost calculation based on tokens)
        assert step_result.cost_usd >= 0.0 