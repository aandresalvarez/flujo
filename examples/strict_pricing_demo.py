#!/usr/bin/env python3
"""
Demonstration of Flujo's Strict Pricing Enforcement Mode.

This example shows how to enable strict pricing mode to ensure that
all cost calculations are based on explicit configuration in flujo.toml,
preventing silent fallbacks to potentially outdated hardcoded defaults.

Usage:
    python examples/strict_pricing_demo.py

This will demonstrate:
1. How strict mode prevents pipelines from running with unconfigured models
2. How strict mode allows pipelines to run when models are properly configured
3. How the default behavior (strict=False) allows fallbacks to hardcoded defaults
"""

import asyncio
import tempfile
import os
from pathlib import Path

from flujo import Flujo, Step, Pipeline
from flujo.exceptions import PricingNotConfiguredError


class MockAgent:
    """A mock agent that simulates LLM usage."""

    def __init__(self, model_id: str, prompt_tokens: int = 100, completion_tokens: int = 50):
        self.model_id = model_id
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    async def run(self, data: str):
        """Simulate an LLM response with usage information."""

        class AgentResponse:
            def __init__(self, output, prompt_tokens, completion_tokens):
                self.output = output
                self._prompt_tokens = prompt_tokens
                self._completion_tokens = completion_tokens

            def usage(self):
                class UsageInfo:
                    def __init__(self, prompt_tokens, completion_tokens):
                        self.request_tokens = prompt_tokens
                        self.response_tokens = completion_tokens

                return UsageInfo(self._prompt_tokens, self._completion_tokens)

        return AgentResponse(f"Response to: {data}", self.prompt_tokens, self.completion_tokens)


def create_flujo_toml(content: str, temp_dir: Path) -> Path:
    """Create a temporary flujo.toml file with the given content."""
    toml_path = temp_dir / "flujo.toml"
    with open(toml_path, "w") as f:
        f.write(content)
    return toml_path


async def demo_strict_mode_success():
    """Demonstrate strict mode working correctly with proper configuration."""
    print("\n=== Demo 1: Strict Mode Success ===")
    print("This demonstrates strict mode working when models are properly configured.")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create flujo.toml with strict mode enabled and proper pricing
        toml_content = """
[cost]
strict = true

[cost.providers.openai.gpt-4o]
prompt_tokens_per_1k = 0.005
completion_tokens_per_1k = 0.015
"""
        create_flujo_toml(toml_content, temp_path)

        # Change to the temp directory so flujo can find the toml file
        original_cwd = os.getcwd()
        os.chdir(temp_path)

        try:
            # Create a pipeline with a properly configured model
            agent = MockAgent("openai:gpt-4o")
            step = Step(name="test_step", agent=agent)
            pipeline = Pipeline.from_step(step)
            runner = Flujo(pipeline)

            # Run the pipeline
            result = None
            async for item in runner.run_async("test input"):
                result = item

            print("✅ SUCCESS: Pipeline completed successfully!")
            print(f"   Step cost: ${result.step_history[0].cost_usd:.4f}")
            print(f"   Total cost: ${result.total_cost_usd:.4f}")

        except Exception as e:
            print(f"❌ FAILED: {e}")
        finally:
            os.chdir(original_cwd)


async def demo_strict_mode_failure():
    """Demonstrate strict mode preventing execution with unconfigured models."""
    print("\n=== Demo 2: Strict Mode Failure ===")
    print("This demonstrates strict mode preventing execution when models are not configured.")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create flujo.toml with strict mode enabled but no pricing for gpt-4o
        toml_content = """
[cost]
strict = true

[cost.providers.openai.gpt-3.5-turbo]
prompt_tokens_per_1k = 0.0015
completion_tokens_per_1k = 0.002
"""
        create_flujo_toml(toml_content, temp_path)

        # Change to the temp directory so flujo can find the toml file
        original_cwd = os.getcwd()
        os.chdir(temp_path)

        try:
            # Create a pipeline with an unconfigured model
            agent = MockAgent(
                "openai:unknown-model"
            )  # Not configured in toml and no hardcoded defaults
            step = Step(name="test_step", agent=agent)
            pipeline = Pipeline.from_step(step)
            runner = Flujo(pipeline)

            # Run the pipeline - should raise PricingNotConfiguredError
            async for item in runner.run_async("test input"):
                pass

            print("❌ FAILED: Pipeline should have raised PricingNotConfiguredError")

        except PricingNotConfiguredError as e:
            print("✅ SUCCESS: Strict mode correctly prevented execution!")
            print(f"   Error: {e}")
        except Exception as e:
            print(f"❌ UNEXPECTED ERROR: {e}")
        finally:
            os.chdir(original_cwd)


async def demo_default_behavior():
    """Demonstrate default behavior (strict=False) allowing fallbacks."""
    print("\n=== Demo 3: Default Behavior (Strict=False) ===")
    print("This demonstrates the default behavior allowing fallbacks to hardcoded defaults.")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create flujo.toml with strict mode disabled (default)
        toml_content = """
# No strict mode specified (defaults to false)
# No pricing configuration
"""
        create_flujo_toml(toml_content, temp_path)

        # Change to the temp directory so flujo can find the toml file
        original_cwd = os.getcwd()
        os.chdir(temp_path)

        try:
            # Create a pipeline with a model that has hardcoded defaults
            agent = MockAgent("openai:gpt-4o")
            step = Step(name="test_step", agent=agent)
            pipeline = Pipeline.from_step(step)
            runner = Flujo(pipeline)

            # Run the pipeline - should succeed with hardcoded defaults
            result = None
            async for item in runner.run_async("test input"):
                result = item

            print("✅ SUCCESS: Pipeline completed with hardcoded defaults!")
            print(f"   Step cost: ${result.step_history[0].cost_usd:.4f}")
            print(f"   Total cost: ${result.total_cost_usd:.4f}")
            print("   Note: This used hardcoded default pricing (may be outdated)")

        except Exception as e:
            print(f"❌ FAILED: {e}")
        finally:
            os.chdir(original_cwd)


async def demo_strict_mode_prevents_hardcoded_fallbacks():
    """Demonstrate that strict mode prevents fallbacks to hardcoded defaults."""
    print("\n=== Demo 4: Strict Mode Prevents Hardcoded Fallbacks ===")
    print(
        "This demonstrates that strict mode prevents fallbacks even for models with hardcoded defaults."
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create flujo.toml with strict mode enabled but no pricing for gpt-4o
        toml_content = """
[cost]
strict = true

# No pricing configuration for gpt-4o (even though it has hardcoded defaults)
"""
        toml_path = create_flujo_toml(toml_content, temp_path)

        # Change to the temp directory so flujo can find the toml file
        original_cwd = os.getcwd()
        os.chdir(temp_path)

        try:
            # Force reload configuration to ensure we get the correct config
            from flujo.infra.config_manager import get_config_manager

            config_manager = get_config_manager(force_reload=True)
            config_manager.config_path = toml_path
            config_manager._config = None  # Clear cache

            # Debug: Check what configuration is being loaded
            from flujo.infra.config import get_cost_config

            cost_config = get_cost_config()
            print(
                f"   Debug: strict={cost_config.strict}, providers={list(cost_config.providers.keys())}"
            )

            # Create a pipeline with gpt-4o (has hardcoded defaults but not configured)
            agent = MockAgent("openai:gpt-4o")
            step = Step(name="test_step", agent=agent)
            pipeline = Pipeline.from_step(step)
            runner = Flujo(pipeline)

            # Run the pipeline - should raise PricingNotConfiguredError
            async for item in runner.run_async("test input"):
                pass

            print("❌ FAILED: Pipeline should have raised PricingNotConfiguredError")

        except PricingNotConfiguredError as e:
            print("✅ SUCCESS: Strict mode correctly prevented hardcoded fallback!")
            print(f"   Error: {e}")
            print(
                "   Note: Even though gpt-4o has hardcoded defaults, strict mode prevents their use"
            )
        except Exception as e:
            print(f"❌ UNEXPECTED ERROR: {e}")
        finally:
            os.chdir(original_cwd)


async def demo_no_toml_file():
    """Demonstrate behavior when no flujo.toml file exists."""
    print("\n=== Demo 5: No flujo.toml File ===")
    print("This demonstrates behavior when no configuration file exists.")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Don't create any flujo.toml file

        # Change to the temp directory
        original_cwd = os.getcwd()
        os.chdir(temp_path)

        try:
            # Create a pipeline with a model that has hardcoded defaults
            agent = MockAgent("openai:gpt-4o")
            step = Step(name="test_step", agent=agent)
            pipeline = Pipeline.from_step(step)
            runner = Flujo(pipeline)

            # Run the pipeline - should succeed with hardcoded defaults
            result = None
            async for item in runner.run_async("test input"):
                result = item

            print("✅ SUCCESS: Pipeline completed with hardcoded defaults!")
            print(f"   Step cost: ${result.step_history[0].cost_usd:.4f}")
            print(f"   Total cost: ${result.total_cost_usd:.4f}")
            print("   Note: This used hardcoded default pricing (may be outdated)")

        except Exception as e:
            print(f"❌ FAILED: {e}")
        finally:
            os.chdir(original_cwd)


async def main():
    """Run all demonstrations."""
    print("🚀 Flujo Strict Pricing Mode Demonstration")
    print("=" * 50)

    await demo_strict_mode_success()
    await demo_strict_mode_failure()
    await demo_default_behavior()
    await demo_strict_mode_prevents_hardcoded_fallbacks()
    await demo_no_toml_file()

    print("\n" + "=" * 50)
    print("📋 Summary:")
    print("• Strict mode ensures all pricing comes from flujo.toml")
    print("• Strict mode prevents silent fallbacks to hardcoded defaults")
    print("• Default behavior (strict=false) allows fallbacks for backward compatibility")
    print("• No flujo.toml file defaults to strict=false behavior")
    print("\n💡 Recommendation: Enable strict mode in production for accurate cost tracking!")


if __name__ == "__main__":
    asyncio.run(main())
