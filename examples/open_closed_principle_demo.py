"""
Demonstrates the Open-Closed Principle in action with Flujo's step complexity detection.

This example shows how new complex step types can be added without modifying
core Flujo code, following the Open-Closed Principle: "open for extension,
closed for modification."

Key Demonstrations:
1. Adding new complex step types without core changes
2. Automatic detection by Flujo's system
3. Sophisticated behavior in custom steps
4. Full backward compatibility
"""

import asyncio
import time
from typing import List, Dict, ClassVar

from flujo import Flujo, Step, step


# =============================================================================
# Custom Complex Step Types (No Core Changes Required!)
# =============================================================================


class CircuitBreakerStep(Step):
    """A custom complex step that implements circuit breaker pattern.

    This step demonstrates how to create sophisticated complex steps without
    modifying Flujo's core code. The circuit breaker pattern automatically
    handles failures and prevents cascading failures.
    """

    is_complex: ClassVar[bool] = True  # Declare complexity - automatically detected by Flujo!

    # Define fields for Pydantic
    failure_threshold: int = 3
    recovery_timeout: float = 60.0
    failure_count: int = 0
    last_failure_time: float = 0
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def __init__(
        self, name: str, failure_threshold: int = 3, recovery_timeout: float = 60.0, **kwargs
    ):
        super().__init__(name=name, agent=self, **kwargs)  # Use self as agent
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

    async def run(self, data: str, **kwargs) -> str:
        """Execute with circuit breaker pattern."""
        current_time = time.time()

        # Check circuit breaker state
        if self.state == "OPEN":
            if current_time - self.last_failure_time > self.recovery_timeout:
                print("🔄 Circuit breaker transitioning to HALF_OPEN")
                self.state = "HALF_OPEN"
            else:
                print("🚫 Circuit breaker is OPEN - request rejected")
                raise Exception("Circuit breaker is OPEN - service unavailable")

        try:
            # Simulate processing that might fail
            if "fail" in data.lower():
                raise ValueError(f"Simulated failure for: {data}")

            result = f"circuit_breaker_processed_{data}"
            print(f"✅ Circuit breaker processing successful: {result}")

            # Reset on success
            if self.state == "HALF_OPEN":
                print("🔄 Circuit breaker reset to CLOSED")
                self.state = "CLOSED"
                self.failure_count = 0

            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = current_time

            print(f"❌ Circuit breaker failure #{self.failure_count}: {e}")

            if self.failure_count >= self.failure_threshold:
                print(f"🚫 Circuit breaker opening after {self.failure_count} failures")
                self.state = "OPEN"

            raise


class RateLimitingStep(Step):
    """A custom complex step that implements rate limiting.

    This step demonstrates how complex steps can implement sophisticated
    behavior while being automatically detected by Flujo's system.
    """

    is_complex: ClassVar[bool] = True  # Declare complexity - no core changes needed!

    # Define fields for Pydantic
    requests_per_second: int = 10
    request_times: List[float] = []
    min_interval: float = 0.1

    def __init__(self, name: str, requests_per_second: int = 10, **kwargs):
        super().__init__(name=name, agent=self, **kwargs)  # Use self as agent
        self.requests_per_second = requests_per_second
        self.request_times = []
        self.min_interval = 1.0 / requests_per_second

    async def run(self, data: str, **kwargs) -> str:
        """Execute with rate limiting."""
        current_time = time.time()

        # Clean old request times
        self.request_times = [t for t in self.request_times if current_time - t < 1.0]

        # Check rate limit
        if len(self.request_times) >= self.requests_per_second:
            oldest_request = min(self.request_times)
            wait_time = 1.0 - (current_time - oldest_request)
            if wait_time > 0:
                print(f"⏳ Rate limit reached, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                current_time = time.time()

        # Record this request
        self.request_times.append(current_time)

        # Process the request
        result = f"rate_limited_{data}"
        print(f"✅ Rate-limited processing: {result} (requests/sec: {len(self.request_times)})")

        return result


class CachingStep(Step):
    """A custom complex step that implements intelligent caching.

    This step demonstrates how complex steps can implement advanced caching
    behavior with automatic cache invalidation and TTL management.
    """

    is_complex: ClassVar[bool] = True  # Declare complexity - automatically handled!

    # Define fields for Pydantic
    cache_ttl: float = 300.0
    max_cache_size: int = 100
    cache: Dict[str, tuple[str, float]] = {}
    cache_hits: int = 0
    cache_misses: int = 0

    def __init__(self, name: str, cache_ttl: float = 300.0, max_cache_size: int = 100, **kwargs):
        super().__init__(name=name, agent=self, **kwargs)  # Use self as agent
        self.cache_ttl = cache_ttl
        self.max_cache_size = max_cache_size
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def _generate_cache_key(self, data: str) -> str:
        """Generate a cache key for the data."""
        return f"cache_{hash(data)}"

    def _cleanup_cache(self) -> None:
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key
            for key, (_, timestamp) in self.cache.items()
            if current_time - timestamp > self.cache_ttl
        ]
        for key in expired_keys:
            del self.cache[key]

        # If still over limit, remove oldest entries
        if len(self.cache) > self.max_cache_size:
            sorted_items = sorted(self.cache.items(), key=lambda x: x[1][1])
            for key, _ in sorted_items[: len(self.cache) - self.max_cache_size]:
                del self.cache[key]

    async def run(self, data: str, **kwargs) -> str:
        """Execute with intelligent caching."""
        cache_key = self._generate_cache_key(data)
        current_time = time.time()

        # Clean up expired cache entries
        self._cleanup_cache()

        # Check cache
        if cache_key in self.cache:
            cached_result, _ = self.cache[cache_key]
            self.cache_hits += 1
            print(f"🎯 Cache HIT: {cached_result}")
            return cached_result

        # Cache miss - process the data
        self.cache_misses += 1
        print("💾 Cache MISS - processing data")

        # Simulate processing
        await asyncio.sleep(0.1)  # Simulate processing time
        result = f"cached_{data}"

        # Store in cache
        self.cache[cache_key] = (result, current_time)

        print(f"✅ Cached result: {result} (hits: {self.cache_hits}, misses: {self.cache_misses})")
        return result


class AdaptiveProcessingStep(Step):
    """A step with dynamic complexity based on input characteristics.

    This step demonstrates how steps can adapt their complexity at runtime
    based on input characteristics, showing the flexibility of the
    object-oriented approach.
    """

    # Define fields for Pydantic
    complexity_threshold: int = 100
    _current_input_size: int = 0
    _current_input_complexity: float = 0.0

    def __init__(self, name: str, complexity_threshold: int = 100, **kwargs):
        super().__init__(name=name, agent=self, **kwargs)  # Use self as agent
        self.complexity_threshold = complexity_threshold
        self._current_input_size = 0
        self._current_input_complexity = 0.0

    @property
    def is_complex(self) -> bool:
        """Dynamic complexity based on input characteristics."""
        return (
            self._current_input_size > self.complexity_threshold
            or self._current_input_complexity > 0.7
        )

    def _calculate_complexity(self, data: str) -> float:
        """Calculate complexity score for the input data."""
        return min(1.0, len(data) / 1000.0)  # Normalize to 0-1

    async def run(self, data: str, **kwargs) -> str:
        """Process data with adaptive complexity."""
        self._current_input_size = len(data)
        self._current_input_complexity = self._calculate_complexity(data)

        if self.is_complex:
            print(
                f"🔧 Using COMPLEX processing for {self._current_input_size} chars (complexity: {self._current_input_complexity:.2f})"
            )
            # Simulate complex processing
            await asyncio.sleep(0.2)
            return f"complex_processed_{data}"
        else:
            print(
                f"⚡ Using SIMPLE processing for {self._current_input_size} chars (complexity: {self._current_input_complexity:.2f})"
            )
            # Simulate simple processing
            return f"simple_processed_{data}"


# =============================================================================
# Simple Agent Steps (No Complexity Declaration Needed)
# =============================================================================


@step(name="DataGenerator")
async def generate_data_agent(count: int) -> str:
    """Generate sample data for processing."""
    print(f"📊 Generating data for count: {count}")
    return f"sample_data_{count}"


@step(name="DataValidator")
async def validate_data_agent(data: str) -> str:
    """Validate data."""
    print(f"🔍 Validating data: {data}")
    if "sample_data" in data:
        return f"validated_{data}"
    else:
        raise ValueError(f"Invalid data: {data}")


@step(name="DataTransformer")
async def transform_data_agent(data: str) -> str:
    """Transform data."""
    print(f"🔄 Transforming data: {data}")
    return f"transformed_{data}"


# =============================================================================
# Open-Closed Principle Demonstrations
# =============================================================================


async def demonstrate_open_closed_principle():
    """Demonstrate the Open-Closed Principle in action."""

    print("🚀 Demonstrating the Open-Closed Principle in Action")
    print("=" * 65)
    print("✅ Adding new complex step types WITHOUT modifying core Flujo code")
    print("✅ Automatic detection by Flujo's object-oriented system")
    print("✅ Sophisticated behavior in custom steps")
    print("✅ Full backward compatibility maintained")
    print()

    # Create custom complex steps (no core changes required!)
    circuit_breaker = CircuitBreakerStep("CircuitBreaker", failure_threshold=2)
    rate_limiter = RateLimitingStep("RateLimiter", requests_per_second=5)
    cache_step = CachingStep("CacheStep", cache_ttl=60.0, max_cache_size=10)
    adaptive_processor = AdaptiveProcessingStep("AdaptiveProcessor", complexity_threshold=5)

    # Create a pipeline that uses all our custom steps
    pipeline = (
        generate_data_agent
        >> validate_data_agent
        >> circuit_breaker
        >> rate_limiter
        >> cache_step
        >> adaptive_processor
    )

    runner = Flujo(pipeline)

    print("📋 Pipeline Configuration (All Custom Steps Automatically Detected):")
    print("   - Data Generator: Simple step (no complexity declaration)")
    print("   - Data Validator: Simple step (no complexity declaration)")
    print("   - Circuit Breaker: Complex step (is_complex = True)")
    print("   - Rate Limiter: Complex step (is_complex = True)")
    print("   - Cache Step: Complex step (is_complex = True)")
    print("   - Adaptive Processor: Dynamic complexity (property-based)")
    print()

    print("🔄 Executing pipeline with custom complex steps...")
    result = None
    async for item in runner.run_async(8):
        result = item

    print("\n✅ Pipeline completed successfully!")
    print(f"   Final output: {result.step_history[-1].output}")
    print(f"   Total steps executed: {len(result.step_history)}")

    # Demonstrate that all steps were handled correctly
    print("\n📊 Step Complexity Analysis:")
    for step_result in result.step_history:
        step_name = (
            step_result.name if hasattr(step_result, "name") else f"Step_{step_result.step_id}"
        )
        is_complex = (
            getattr(step_result.step, "is_complex", False)
            if hasattr(step_result, "step")
            else False
        )
        complexity_type = "Complex" if is_complex else "Simple"
        print(f"   - {step_name}: {complexity_type}")


async def demonstrate_circuit_breaker():
    """Demonstrate circuit breaker pattern in action."""

    print("\n🔄 Demonstrating Circuit Breaker Pattern")
    print("=" * 45)

    circuit_breaker = CircuitBreakerStep("CircuitBreakerDemo", failure_threshold=2)
    runner = Flujo(circuit_breaker)

    # Test with data that will fail
    print("\n📊 Testing circuit breaker with failing data:")
    try:
        result = None
        async for item in runner.run_async("fail_on_first_attempt"):
            result = item
        print(f"   Result: {result.step_history[-1].output}")
    except Exception as e:
        print(f"   Expected failure: {e}")

    # Test with data that will succeed
    print("\n📊 Testing circuit breaker with successful data:")
    result = None
    async for item in runner.run_async("successful_data"):
        result = item
    print(f"   Result: {result.step_history[-1].output}")


async def demonstrate_rate_limiting():
    """Demonstrate rate limiting in action."""

    print("\n🔄 Demonstrating Rate Limiting")
    print("=" * 35)

    rate_limiter = RateLimitingStep("RateLimiterDemo", requests_per_second=3)
    runner = Flujo(rate_limiter)

    print("\n📊 Testing rate limiting with multiple rapid requests:")
    for i in range(5):
        result = None
        async for item in runner.run_async(f"request_{i}"):
            result = item
        print(f"   Request {i}: {result.step_history[-1].output}")


async def demonstrate_caching():
    """Demonstrate intelligent caching in action."""

    print("\n🔄 Demonstrating Intelligent Caching")
    print("=" * 40)

    cache_step = CachingStep("CacheDemo", cache_ttl=30.0, max_cache_size=5)
    runner = Flujo(cache_step)

    print("\n📊 Testing caching with repeated requests:")
    for i in range(3):
        result = None
        async for item in runner.run_async("same_data"):
            result = item
        print(f"   Request {i}: {result.step_history[-1].output}")


async def demonstrate_adaptive_processing():
    """Demonstrate adaptive processing in action."""

    print("\n🔄 Demonstrating Adaptive Processing")
    print("=" * 40)

    adaptive_processor = AdaptiveProcessingStep("AdaptiveDemo", complexity_threshold=5)
    runner = Flujo(adaptive_processor)

    # Test with small data (should be simple)
    small_data = "abc"
    print(f"\n📊 Testing with small data ({len(small_data)} chars):")
    result = None
    async for item in runner.run_async(small_data):
        result = item
    print(f"   Output: {result.step_history[-1].output}")

    # Test with large data (should be complex)
    large_data = "very_long_data_string_that_exceeds_threshold"
    print(f"\n📊 Testing with large data ({len(large_data)} chars):")
    result = None
    async for item in runner.run_async(large_data):
        result = item
    print(f"   Output: {result.step_history[-1].output}")


async def demonstrate_extensibility_without_core_changes():
    """Demonstrate that new step types work without core changes."""

    print("\n🔄 Demonstrating Extensibility Without Core Changes")
    print("=" * 55)

    # Create a completely new step type that wasn't in the original Flujo
    class TimeoutStep(Step):
        """A step that implements timeout functionality."""

        is_complex: ClassVar[bool] = True  # Declare as complex - automatically detected!

        # Define fields for Pydantic
        timeout_seconds: float = 5.0

        def __init__(self, name: str, timeout_seconds: float = 5.0, **kwargs):
            super().__init__(name=name, agent=self, **kwargs)  # Use self as agent
            self.timeout_seconds = timeout_seconds

        async def run(self, data: str, **kwargs) -> str:
            """Execute with timeout protection."""
            print(f"⏱️ Executing with {self.timeout_seconds}s timeout...")

            # Simulate processing that might take time
            if "slow" in data.lower():
                await asyncio.sleep(self.timeout_seconds + 1)  # Exceed timeout
                raise TimeoutError(f"Processing exceeded {self.timeout_seconds}s timeout")

            result = f"timeout_processed_{data}"
            print(f"✅ Timeout processing completed: {result}")
            return result

    # Use the new step type without any core changes
    timeout_step = TimeoutStep("TimeoutDemo", timeout_seconds=2.0)
    runner = Flujo(timeout_step)

    print("\n📊 Testing new TimeoutStep (automatically detected as complex):")

    # Test with normal data
    result = None
    async for item in runner.run_async("normal_data"):
        result = item
    print(f"   Normal data result: {result.step_history[-1].output}")

    # Test with slow data (should timeout)
    try:
        result = None
        async for item in runner.run_async("slow_data"):
            result = item
        print(f"   Slow data result: {result.step_history[-1].output}")
    except Exception as e:
        print(f"   Expected timeout: {e}")


async def main():
    """Run all Open-Closed Principle demonstrations."""
    await demonstrate_open_closed_principle()
    await demonstrate_circuit_breaker()
    await demonstrate_rate_limiting()
    await demonstrate_caching()
    await demonstrate_adaptive_processing()
    await demonstrate_extensibility_without_core_changes()

    print("\n🎉 Open-Closed Principle Demonstration Complete!")
    print("\nKey Benefits Demonstrated:")
    print("✅ New complex step types can be added WITHOUT modifying core Flujo code")
    print("✅ Automatic detection by Flujo's object-oriented system")
    print("✅ Sophisticated behavior (circuit breaker, rate limiting, caching)")
    print("✅ Dynamic complexity detection using properties")
    print("✅ Full backward compatibility with existing code")
    print("✅ Open-Closed Principle in action: 'open for extension, closed for modification'")


if __name__ == "__main__":
    asyncio.run(main())
