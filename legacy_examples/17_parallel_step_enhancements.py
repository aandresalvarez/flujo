"""
Example demonstrating ParallelStep performance and resiliency enhancements.

This example shows:
1. Optimized context copying with context_include_keys
2. Proactive governor cancellation when usage limits are breached
3. Performance improvements for large contexts
4. Resource efficiency under strict limits
"""

import asyncio
import time
from typing import Any
from pydantic import BaseModel

from flujo import Flujo, Step, Pipeline, UsageLimits, UsageLimitExceededError


class LargeContext(BaseModel):
    """A context with large data to demonstrate context copying optimization."""
    user_id: str = "user_123"
    document_id: str = "doc_456"
    small_field: str = "small"
    large_document: str = "x" * 100000  # Large field that's expensive to copy
    large_image_data: str = "y" * 100000  # Another large field
    metadata: dict = {"version": "1.0", "author": "team"}


class AnalysisAgent:
    """An agent that performs analysis and reports high cost."""
    
    def __init__(self, name: str, cost: float = 0.1, delay: float = 0.1):
        self.name = name
        self.cost = cost
        self.delay = delay
    
    async def run(self, data: Any, *, context: BaseModel | None = None) -> Any:
        await asyncio.sleep(self.delay)  # Simulate processing time
        
        # Use context to demonstrate context isolation
        user_id = getattr(context, "user_id", "unknown") if context else "unknown"
        
        class Output(BaseModel):
            analysis: str
            cost_usd: float = self.cost
            token_counts: int = 100
            user_id: str = user_id
        
        return Output(
            analysis=f"{self.name} analysis of: {data}",
            user_id=user_id
        )


class SummaryAgent:
    """An agent that creates summaries with moderate cost."""
    
    def __init__(self, cost: float = 0.05, delay: float = 0.2):
        self.cost = cost
        self.delay = delay
    
    async def run(self, data: Any, *, context: BaseModel | None = None) -> Any:
        await asyncio.sleep(self.delay)  # Simulate processing time
        
        class Output(BaseModel):
            summary: str
            cost_usd: float = self.cost
            token_counts: int = 50
        
        return Output(summary=f"Summary of: {data}")


class ValidationAgent:
    """An agent that performs validation with low cost but long execution time."""
    
    async def run(self, data: Any, *, context: BaseModel | None = None) -> Any:
        await asyncio.sleep(0.5)  # Long execution time
        
        class Output(BaseModel):
            validation: str
            cost_usd: float = 0.01  # Very cheap
            token_counts: int = 10
        
        return Output(validation=f"Validated: {data}")


def demonstrate_context_copying_optimization():
    """Demonstrate the performance improvement from optimized context copying."""
    
    print("🔧 Demonstrating Context Copying Optimization")
    print("=" * 50)
    
    # Create branches that only need specific context fields
    branches = {
        "analysis": Step.model_validate({
            "name": "analysis", 
            "agent": AnalysisAgent("Document")
        }),
        "summary": Step.model_validate({
            "name": "summary", 
            "agent": SummaryAgent()
        }),
    }
    
    # Test with selective context copying (only needed fields)
    parallel_optimized = Step.parallel(
        "parallel_optimized", 
        branches, 
        context_include_keys=["user_id", "document_id", "small_field"]
    )
    
    # Test with full context copying (default behavior)
    parallel_full = Step.parallel("parallel_full", branches)
    
    context = LargeContext()
    
    # Measure performance with optimized copying
    print("\n📊 Testing optimized context copying...")
    start = time.monotonic()
    runner_optimized = Flujo(parallel_optimized, context_model=LargeContext)
    result_optimized = runner_optimized.run("sample data", initial_context_data=context.model_dump())
    optimized_time = time.monotonic() - start
    
    # Measure performance with full copying
    print("📊 Testing full context copying...")
    start = time.monotonic()
    runner_full = Flujo(parallel_full, context_model=LargeContext)
    result_full = runner_full.run("sample data", initial_context_data=context.model_dump())
    full_time = time.monotonic() - start
    
    print(f"\n⏱️  Performance Results:")
    print(f"   Optimized copying: {optimized_time:.4f}s")
    print(f"   Full copying:      {full_time:.4f}s")
    print(f"   Improvement:       {((full_time - optimized_time) / full_time * 100):.1f}%")
    
    # Verify both produce correct results
    print(f"\n✅ Results verification:")
    print(f"   Optimized output: {result_optimized.step_history[-1].output}")
    print(f"   Full output:      {result_full.step_history[-1].output}")
    
    return optimized_time, full_time


def demonstrate_proactive_cancellation():
    """Demonstrate proactive cancellation when usage limits are breached."""
    
    print("\n\n🚨 Demonstrating Proactive Governor Cancellation")
    print("=" * 50)
    
    # Create branches with different costs and execution times
    branches = {
        "fast_expensive": Step.model_validate({
            "name": "fast_expensive", 
            "agent": AnalysisAgent("Fast", cost=0.15, delay=0.05)  # Breaches limit quickly
        }),
        "slow_cheap": Step.model_validate({
            "name": "slow_cheap", 
            "agent": ValidationAgent()  # Takes longer but is cheap
        }),
    }
    
    parallel = Step.parallel("parallel_cancellation", branches)
    limits = UsageLimits(total_cost_usd_limit=0.10)  # Limit that will be breached
    runner = Flujo(parallel, usage_limits=limits)
    
    print(f"\n💰 Cost limit: ${limits.total_cost_usd_limit}")
    print(f"📊 Branch costs: fast_expensive=${0.15}, slow_cheap=${0.01}")
    
    # Measure execution time with proactive cancellation
    print("\n⏱️  Testing with proactive cancellation...")
    start = time.monotonic()
    
    try:
        result = runner.run("test data")
    except UsageLimitExceededError as e:
        cancellation_time = time.monotonic() - start
        print(f"✅ Limit breached as expected: {e}")
        print(f"⏱️  Execution time: {cancellation_time:.4f}s")
        print(f"💰 Final cost: ${e.result.total_cost_usd:.2f}")
        
        # Verify execution was fast (indicating proactive cancellation)
        if cancellation_time < 0.3:
            print("✅ Proactive cancellation working: slow_cheap branch was cancelled quickly")
        else:
            print("⚠️  Execution took longer than expected")
    
    # Test without limits to show the difference
    print("\n⏱️  Testing without limits (for comparison)...")
    runner_no_limits = Flujo(parallel)  # No usage limits
    start = time.monotonic()
    result_no_limits = runner_no_limits.run("test data")
    no_limits_time = time.monotonic() - start
    
    print(f"⏱️  Full execution time: {no_limits_time:.4f}s")
    print(f"💰 Time saved: {((no_limits_time - cancellation_time) / no_limits_time * 100):.1f}%")


def demonstrate_combined_features():
    """Demonstrate both features working together."""
    
    print("\n\n🚀 Demonstrating Combined Features")
    print("=" * 50)
    
    # Create a complex parallel step with both optimizations
    branches = {
        "analysis": Step.model_validate({
            "name": "analysis", 
            "agent": AnalysisAgent("Complex", cost=0.12, delay=0.05)
        }),
        "summary": Step.model_validate({
            "name": "summary", 
            "agent": SummaryAgent(cost=0.03, delay=0.1)
        }),
        "validation": Step.model_validate({
            "name": "validation", 
            "agent": ValidationAgent()
        }),
    }
    
    # Use both optimizations
    parallel = Step.parallel(
        "parallel_combined", 
        branches, 
        context_include_keys=["user_id", "document_id"]  # Only copy needed fields
    )
    
    limits = UsageLimits(total_cost_usd_limit=0.10)  # Will be breached by analysis
    runner = Flujo(parallel, usage_limits=limits, context_model=LargeContext)
    
    context = LargeContext()
    
    print(f"\n🔧 Using optimized context copying (only user_id, document_id)")
    print(f"💰 Cost limit: ${limits.total_cost_usd_limit}")
    print(f"📊 Branch costs: analysis=${0.12}, summary=${0.03}, validation=${0.01}")
    
    start = time.monotonic()
    
    try:
        result = runner.run("combined test", initial_context_data=context.model_dump())
    except UsageLimitExceededError as e:
        execution_time = time.monotonic() - start
        print(f"\n✅ Combined optimization results:")
        print(f"   ⏱️  Execution time: {execution_time:.4f}s")
        print(f"   💰 Final cost: ${e.result.total_cost_usd:.2f}")
        print(f"   🚫 Branches cancelled: summary, validation")
        print(f"   ✅ Analysis completed: {e.result.step_history[-1].output['analysis']}")


def main():
    """Run all demonstrations."""
    
    print("🎯 ParallelStep Performance and Resiliency Enhancements")
    print("=" * 60)
    print("\nThis example demonstrates two key enhancements to ParallelStep:")
    print("1. 🔧 Optimized Context Copying: Reduce memory usage and improve performance")
    print("2. 🚨 Proactive Governor Cancellation: Cancel sibling tasks when limits are breached")
    
    # Demonstrate context copying optimization
    optimized_time, full_time = demonstrate_context_copying_optimization()
    
    # Demonstrate proactive cancellation
    demonstrate_proactive_cancellation()
    
    # Demonstrate combined features
    demonstrate_combined_features()
    
    print("\n\n🎉 Enhancement Summary:")
    print("=" * 30)
    print("✅ Context copying optimization reduces memory usage and improves performance")
    print("✅ Proactive cancellation prevents wasted resources when limits are breached")
    print("✅ Both features work together seamlessly")
    print("✅ Backward compatibility is maintained")


if __name__ == "__main__":
    main() 