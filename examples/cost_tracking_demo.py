#!/usr/bin/env python3
"""
Cost Tracking Demo

This example demonstrates how to use Flujo's cost tracking and usage limits features.
Make sure you have a flujo.toml file with cost configuration before running this example.
"""

import asyncio
from flujo import Step, Flujo, UsageLimits
from flujo.agents import make_agent_async
from flujo.exceptions import UsageLimitExceededError


async def basic_cost_tracking():
    """Demonstrate basic cost tracking functionality."""
    print("=== Basic Cost Tracking ===")

    # Create agents
    solution_agent = make_agent_async(
        "openai:gpt-4o", "You are a helpful assistant. Provide concise responses.", str
    )
    validator_agent = make_agent_async(
        "openai:gpt-4o", "You are a validator. Rate the quality of the response from 1-10.", str
    )

    # Create pipeline
    pipeline = Step.solution(solution_agent) >> Step.validate(validator_agent)
    runner = Flujo(pipeline)

    # Run pipeline
    result = await runner.run_async("Write a short poem about coding.")

    # Display cost information
    print("\nPipeline completed successfully!")
    print(f"Total steps: {len(result.step_history)}")

    total_cost = 0
    total_tokens = 0

    for step_result in result.step_history:
        cost = step_result.cost_usd
        tokens = step_result.token_counts
        total_cost += cost
        total_tokens += tokens

        print(f"\n{step_result.name}:")
        print(f"  Cost: ${cost:.4f}")
        print(f"  Tokens: {tokens}")
        print(f"  Success: {step_result.success}")
        print(f"  Latency: {step_result.latency_s:.3f}s")

    print(f"\nTotal cost: ${total_cost:.4f}")
    print(f"Total tokens: {total_tokens}")


async def usage_limits_demo():
    """Demonstrate usage limits functionality."""
    print("\n=== Usage Limits Demo ===")

    # Create an agent that might use many tokens
    verbose_agent = make_agent_async(
        "openai:gpt-4o",
        "You are a verbose assistant. Provide very detailed, comprehensive responses with lots of examples and explanations.",
        str,
    )

    # Set strict limits
    limits = UsageLimits(
        total_cost_usd_limit=0.10,  # Maximum $0.10
        total_tokens_limit=500,  # Maximum 500 tokens
    )

    pipeline = Step.solution(verbose_agent)
    runner = Flujo(pipeline, usage_limits=limits)

    try:
        await runner.run_async("Explain quantum computing in detail with many examples.")
        print("Pipeline completed successfully!")
    except UsageLimitExceededError as e:
        print(f"Pipeline stopped due to usage limits: {e}")
        print("Partial results available:")

        # Show partial results
        for step_result in e.partial_result.step_history:
            print(
                f"  {step_result.name}: ${step_result.cost_usd:.4f} ({step_result.token_counts} tokens)"
            )


async def step_level_limits():
    """Demonstrate step-level usage limits."""
    print("\n=== Step-Level Limits Demo ===")

    # Create agents
    solution_agent = make_agent_async(
        "openai:gpt-4o", "You are a solution agent. Provide detailed solutions.", str
    )
    validator_agent = make_agent_async(
        "openai:gpt-4o", "You are a validator. Provide brief validation feedback.", str
    )

    # Set different limits for different steps
    solution_limits = UsageLimits(
        total_cost_usd_limit=0.15,  # More budget for solution
        total_tokens_limit=800,
    )
    validation_limits = UsageLimits(
        total_cost_usd_limit=0.05,  # Less budget for validation
        total_tokens_limit=200,
    )

    pipeline = Step.solution(solution_agent, usage_limits=solution_limits) >> Step.validate(
        validator_agent, usage_limits=validation_limits
    )

    runner = Flujo(pipeline)

    try:
        result = await runner.run_async("Design a web application architecture.")
        print("Pipeline completed successfully!")

        for step_result in result.step_history:
            print(
                f"{step_result.name}: ${step_result.cost_usd:.4f} ({step_result.token_counts} tokens)"
            )

    except UsageLimitExceededError as e:
        print(f"Pipeline stopped due to step limits: {e}")


async def cost_efficient_pipeline():
    """Demonstrate cost-efficient pipeline design."""
    print("\n=== Cost-Efficient Pipeline Demo ===")

    # Use cheaper model for simple tasks
    simple_agent = make_agent_async(
        "openai:gpt-3.5-turbo", "You are a simple assistant. Provide brief responses.", str
    )

    # Use expensive model only for complex tasks
    complex_agent = make_agent_async(
        "openai:gpt-4o", "You are a complex assistant. Provide detailed analysis.", str
    )

    # Design pipeline with cost considerations
    pipeline = Step.solution(
        simple_agent, usage_limits=UsageLimits(total_cost_usd_limit=0.02)
    ) >> Step.validate(complex_agent, usage_limits=UsageLimits(total_cost_usd_limit=0.08))

    # Set overall pipeline limits
    runner = Flujo(pipeline, usage_limits=UsageLimits(total_cost_usd_limit=0.12))

    try:
        result = await runner.run_async("Analyze the benefits of microservices architecture.")
        print("Cost-efficient pipeline completed!")

        total_cost = sum(step.cost_usd for step in result.step_history)
        print(f"Total cost: ${total_cost:.4f}")

    except UsageLimitExceededError as e:
        print(f"Pipeline stopped due to limits: {e}")


async def monitoring_demo():
    """Demonstrate cost monitoring in production."""
    print("\n=== Cost Monitoring Demo ===")

    import logging

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def log_costs(result):
        """Log cost information for monitoring."""
        total_cost = sum(step.cost_usd for step in result.step_history)
        total_tokens = sum(step.token_counts for step in result.step_history)

        logger.info(f"Pipeline completed - Cost: ${total_cost:.4f}, Tokens: {total_tokens}")

        # Log per-step details
        for step in result.step_history:
            logger.info(f"  {step.name}: ${step.cost_usd:.4f} ({step.token_counts} tokens)")

    # Create pipeline
    agent = make_agent_async("openai:gpt-4o", "You are a helpful assistant.", str)
    pipeline = Step.solution(agent)
    runner = Flujo(pipeline)

    # Run with monitoring
    result = await runner.run_async("Explain machine learning concepts.")
    log_costs(result)


async def main():
    """Run all cost tracking demos."""
    print("Flujo Cost Tracking Demo")
    print("=" * 50)
    print("Make sure you have configured pricing in your flujo.toml file!")
    print()

    try:
        await basic_cost_tracking()
        await usage_limits_demo()
        await step_level_limits()
        await cost_efficient_pipeline()
        await monitoring_demo()

    except Exception as e:
        print(f"Error running demo: {e}")
        print("Make sure you have:")
        print("1. Valid API keys configured")
        print("2. Cost configuration in flujo.toml")
        print("3. Proper model access")


if __name__ == "__main__":
    asyncio.run(main())
