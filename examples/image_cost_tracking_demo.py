#!/usr/bin/env python3
"""
Image Cost Tracking Demo

This example demonstrates how to use Flujo's image generation cost tracking feature.
Make sure you have a flujo.toml file with DALL-E 3 cost configuration before running this example.
"""

import asyncio
from flujo import Step, Flujo, UsageLimits
from flujo.exceptions import UsageLimitExceededError


class MockDALLE3Agent:
    """A mock DALL-E 3 agent that simulates image generation with cost tracking."""

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
                self.output = (
                    f"Generated {image_count} image(s) with quality {quality} and size {size}"
                )
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


async def basic_image_cost_tracking():
    """Demonstrate basic image cost tracking functionality."""
    print("=== Basic Image Cost Tracking ===")

    # Create a mock DALL-E 3 agent
    dalle_agent = MockDALLE3Agent(image_count=2, quality="standard", size="1024x1024")

    # Create pipeline
    pipeline = Step.solution(dalle_agent)
    runner = Flujo(pipeline)

    # Run pipeline
    result = await runner.run_async("Generate a beautiful landscape")

    # Display cost information
    print("\nPipeline completed successfully!")
    print(f"Total steps: {len(result.step_history)}")

    total_cost = 0

    for step_result in result.step_history:
        cost = step_result.cost_usd
        total_cost += cost

        print(f"\n{step_result.name}:")
        print(f"  Cost: ${cost:.4f}")
        print(f"  Success: {step_result.success}")
        print(f"  Latency: {step_result.latency_s:.3f}s")

    print(f"\nTotal cost: ${total_cost:.4f}")


async def image_cost_tracking_with_different_qualities():
    """Demonstrate image cost tracking with different qualities and sizes."""
    print("\n=== Image Cost Tracking with Different Qualities ===")

    # Create agents with different configurations
    standard_agent = MockDALLE3Agent(image_count=1, quality="standard", size="1024x1024")
    hd_agent = MockDALLE3Agent(image_count=1, quality="hd", size="1024x1024")
    large_agent = MockDALLE3Agent(image_count=1, quality="standard", size="1792x1024")

    # Create pipeline with multiple steps
    pipeline = Step.solution(standard_agent) >> Step.validate(hd_agent) >> Step.reflect(large_agent)
    runner = Flujo(pipeline)

    # Run pipeline
    result = await runner.run_async("Generate multiple images with different qualities")

    # Display cost information
    print("\nPipeline completed successfully!")
    print(f"Total steps: {len(result.step_history)}")

    total_cost = 0

    for i, step_result in enumerate(result.step_history):
        cost = step_result.cost_usd
        total_cost += cost

        print(f"\nStep {i + 1} ({step_result.name}):")
        print(f"  Cost: ${cost:.4f}")
        print(f"  Success: {step_result.success}")

    print(f"\nTotal cost: ${total_cost:.4f}")


async def image_cost_tracking_with_usage_limits():
    """Demonstrate image cost tracking with usage limits."""
    print("\n=== Image Cost Tracking with Usage Limits ===")

    # Create a mock DALL-E 3 agent
    dalle_agent = MockDALLE3Agent(image_count=1, quality="hd", size="1024x1024")

    # Create pipeline
    pipeline = Step.solution(dalle_agent)

    # Set usage limits below the expected cost
    usage_limits = UsageLimits(total_cost_usd_limit=0.01)  # $0.01 limit
    runner = Flujo(pipeline, usage_limits=usage_limits)

    try:
        # Run pipeline - should fail due to usage limits
        await runner.run_async("Generate a high-quality image")
        print("Pipeline completed (unexpected)")
    except UsageLimitExceededError as e:
        print(f"Pipeline failed as expected: {e}")
        print("This demonstrates that image costs are properly integrated with usage limits.")


async def image_cost_tracking_regression_test():
    """Test that image cost tracking doesn't interfere with chat models."""
    print("\n=== Image Cost Tracking Regression Test ===")

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

    chat_agent = MockChatAgent()

    # Create pipeline
    pipeline = Step.solution(chat_agent)
    runner = Flujo(pipeline)

    # Run pipeline
    result = await runner.run_async("Hello")

    # Display cost information
    print("\nChat pipeline completed successfully!")
    print(f"Total steps: {len(result.step_history)}")

    for step_result in result.step_history:
        cost = step_result.cost_usd
        tokens = step_result.token_counts

        print(f"\n{step_result.name}:")
        print(f"  Cost: ${cost:.4f}")
        print(f"  Tokens: {tokens}")
        print(f"  Success: {step_result.success}")

    print(
        "\nThis demonstrates that chat models still work normally with token-based cost calculation."
    )


async def main():
    """Run all image cost tracking demonstrations."""
    print("🚀 Image Cost Tracking Demo")
    print("=" * 50)

    try:
        await basic_image_cost_tracking()
        await image_cost_tracking_with_different_qualities()
        await image_cost_tracking_with_usage_limits()
        await image_cost_tracking_regression_test()

        print("\n✅ All demonstrations completed successfully!")
        print("\nKey Features Demonstrated:")
        print("- Automatic image cost calculation based on quality and size")
        print("- Integration with usage limits")
        print("- Support for multiple image configurations")
        print("- Backward compatibility with chat models")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("Make sure you have DALL-E 3 pricing configured in your flujo.toml file.")


if __name__ == "__main__":
    asyncio.run(main())
