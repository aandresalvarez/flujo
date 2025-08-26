"""
Demonstrates the extensibility benefits of Flujo's object-oriented step complexity detection.

This example shows how new complex step types can be added without modifying core Flujo code,
following the Open-Closed Principle and algebraic closure.
"""

import asyncio
from typing import Any, List

from flujo import Flujo, Step, step
from flujo.models import PipelineResult


# =============================================================================
# Custom Complex Step Types (No Core Changes Required!)
# =============================================================================


class BatchProcessingStep(Step):
    """A custom complex step that processes data in batches.

    This step demonstrates how to create new complex step types without
    modifying Flujo's core code. Simply set is_complex = True and the
    system will automatically handle it correctly.
    """

    is_complex = True  # Declare complexity - no core changes needed!

    def __init__(self, name: str, batch_size: int = 10, **kwargs):
        super().__init__(name=name, **kwargs)
        self.batch_size = batch_size
        self.processed_items = 0

    async def run(self, data: List[Any], **kwargs) -> List[Any]:
        """Process data in batches."""
        print(f"🔄 Processing {len(data)} items in batches of {self.batch_size}")

        results = []
        for i in range(0, len(data), self.batch_size):
            batch = data[i : i + self.batch_size]
            print(f"   Processing batch {i // self.batch_size + 1}: {len(batch)} items")

            # Simulate batch processing
            batch_results = [f"processed_{item}" for item in batch]
            results.extend(batch_results)
            self.processed_items += len(batch)

        print(f"✅ Completed batch processing: {self.processed_items} items total")
        return results


class AdaptiveStep(Step):
    """A step with dynamic complexity based on input characteristics.

    This step demonstrates how to implement dynamic complexity detection
    using properties, allowing the step to adapt its behavior based on
    runtime conditions.
    """

    def __init__(self, name: str, complexity_threshold: int = 100, **kwargs):
        super().__init__(name=name, **kwargs)
        self.complexity_threshold = complexity_threshold
        self._current_input_size = 0

    @property
    def is_complex(self) -> bool:
        """Dynamic complexity based on input size."""
        return self._current_input_size > self.complexity_threshold

    async def run(self, data: Any, **kwargs) -> Any:
        """Process data with adaptive complexity."""
        self._current_input_size = len(data) if isinstance(data, (list, tuple)) else 1

        if self.is_complex:
            print(f"🔧 Using complex processing for {self._current_input_size} items")
            # Simulate complex processing
            await asyncio.sleep(0.1)
            return f"complex_processed_{data}"
        else:
            print(f"⚡ Using simple processing for {self._current_input_size} items")
            # Simulate simple processing
            return f"simple_processed_{data}"


class RetryStep(Step):
    """A custom step that implements retry logic with exponential backoff.

    This step demonstrates how complex steps can implement sophisticated
    behavior while being automatically detected by Flujo's system.
    """

    is_complex = True  # Declare as complex for retry logic

    def __init__(self, name: str, max_retries: int = 3, base_delay: float = 0.1, **kwargs):
        super().__init__(name=name, **kwargs)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.retry_count = 0

    async def run(self, data: Any, **kwargs) -> Any:
        """Execute with retry logic."""
        for attempt in range(self.max_retries + 1):
            try:
                print(f"🔄 Attempt {attempt + 1}/{self.max_retries + 1}")

                # Simulate processing that might fail
                if attempt < self.max_retries and "fail" in str(data).lower():
                    raise ValueError(f"Simulated failure on attempt {attempt + 1}")

                result = f"successful_{data}"
                print(f"✅ Succeeded on attempt {attempt + 1}")
                return result

            except Exception as e:
                self.retry_count += 1
                if attempt < self.max_retries:
                    delay = self.base_delay * (2**attempt)  # Exponential backoff
                    print(f"❌ Failed on attempt {attempt + 1}: {e}")
                    print(f"   Retrying in {delay:.2f}s...")
                    await asyncio.sleep(delay)
                else:
                    print(f"💥 All {self.max_retries + 1} attempts failed")
                    raise


# =============================================================================
# Simple Agent Steps (No Complexity Declaration Needed)
# =============================================================================


@step(name="DataGenerator")
async def generate_data_agent(count: int) -> List[str]:
    """Generate sample data for processing."""
    print(f"📊 Generating {count} data items...")
    return [f"item_{i}" for i in range(count)]


@step(name="DataValidator")
async def validate_data_agent(data: List[str]) -> List[str]:
    """Validate data items."""
    print(f"🔍 Validating {len(data)} items...")
    return [item for item in data if "item" in item]


@step(name="DataTransformer")
async def transform_data_agent(data: List[str]) -> List[str]:
    """Transform data items."""
    print(f"🔄 Transforming {len(data)} items...")
    return [f"transformed_{item}" for item in data]


# =============================================================================
# Pipeline Demonstrations
# =============================================================================


async def demonstrate_extensibility():
    """Demonstrate the extensibility benefits of the new architecture."""

    print("🚀 Demonstrating Flujo's Extensibility Benefits")
    print("=" * 60)

    # Create custom complex steps (no core changes required!)
    batch_processor = BatchProcessingStep("BatchProcessor", batch_size=3)
    adaptive_processor = AdaptiveStep("AdaptiveProcessor", complexity_threshold=5)
    retry_processor = RetryStep("RetryProcessor", max_retries=2)

    # Create a pipeline that uses all our custom steps
    pipeline = (
        generate_data_agent
        >> validate_data_agent
        >> batch_processor
        >> adaptive_processor
        >> retry_processor
    )

    runner = Flujo(pipeline)

    print("\n📋 Pipeline Configuration:")
    print("   - Data Generator: Simple step (no complexity declaration)")
    print("   - Data Validator: Simple step (no complexity declaration)")
    print("   - Batch Processor: Complex step (is_complex = True)")
    print("   - Adaptive Processor: Dynamic complexity (property-based)")
    print("   - Retry Processor: Complex step (is_complex = True)")

    print("\n🔄 Executing pipeline...")
    result: PipelineResult = await runner.run(10)

    print("\n✅ Pipeline completed successfully!")
    print(f"   Final output: {result.step_history[-1].output}")
    print(f"   Total steps executed: {len(result.step_history)}")

    # Demonstrate that all steps were handled correctly
    for step_result in result.step_history:
        step_name = step_result.step.name
        step_type = type(step_result.step).__name__
        print(
            f"   - {step_name} ({step_type}): {'Complex' if getattr(step_result.step, 'is_complex', False) else 'Simple'}"
        )


async def demonstrate_dynamic_complexity():
    """Demonstrate dynamic complexity detection."""

    print("\n🔄 Demonstrating Dynamic Complexity Detection")
    print("=" * 50)

    # Create adaptive steps with different thresholds
    simple_adaptive = AdaptiveStep("SimpleAdaptive", complexity_threshold=100)
    complex_adaptive = AdaptiveStep("ComplexAdaptive", complexity_threshold=3)

    # Test with small data (should be simple)
    small_data = ["a", "b", "c"]
    print(f"\n📊 Testing with small data ({len(small_data)} items):")

    pipeline_small = simple_adaptive
    runner_small = Flujo(pipeline_small)
    result_small = await runner_small.run(small_data)
    print(f"   Output: {result_small.step_history[-1].output}")

    # Test with large data (should be complex)
    large_data = ["item_" + str(i) for i in range(10)]
    print(f"\n📊 Testing with large data ({len(large_data)} items):")

    pipeline_large = complex_adaptive
    runner_large = Flujo(pipeline_large)
    result_large = await runner_large.run(large_data)
    print(f"   Output: {result_large.step_history[-1].output}")


async def demonstrate_retry_logic():
    """Demonstrate retry logic with exponential backoff."""

    print("\n🔄 Demonstrating Retry Logic")
    print("=" * 40)

    retry_step = RetryStep("RetryDemo", max_retries=2)
    runner = Flujo(retry_step)

    # Test with data that will fail initially
    print("\n📊 Testing retry logic with failing data:")
    try:
        result = await runner.run("fail_on_first_attempt")
        print(f"   Final result: {result.step_history[-1].output}")
    except Exception as e:
        print(f"   Final failure: {e}")

    # Test with data that will succeed
    print("\n📊 Testing retry logic with successful data:")
    result = await runner.run("successful_data")
    print(f"   Final result: {result.step_history[-1].output}")


async def main():
    """Run all demonstrations."""
    await demonstrate_extensibility()
    await demonstrate_dynamic_complexity()
    await demonstrate_retry_logic()

    print("\n🎉 Extensibility Demonstration Complete!")
    print("\nKey Benefits Demonstrated:")
    print("✅ New complex step types can be added without core changes")
    print("✅ Dynamic complexity detection using properties")
    print("✅ Sophisticated behavior (retry logic, batch processing)")
    print("✅ Automatic detection by Flujo's system")
    print("✅ Full backward compatibility maintained")


if __name__ == "__main__":
    asyncio.run(main())
