#!/usr/bin/env python3
"""
Comprehensive ExecutorCore Optimization Demo

This example demonstrates all optimization features including:
- Memory optimizations (object pooling, context management)
- Performance optimizations (step execution, algorithm optimizations)
- Telemetry optimizations (monitoring, metrics)
- Error handling optimizations (circuit breaker, error caching)
- Configuration management and monitoring

Usage:
    python optimization_comprehensive_demo.py
"""

import asyncio
import time
import json
import logging
from typing import Any, List
from dataclasses import dataclass
from datetime import datetime, timezone

from flujo.application.core.executor_core import ExecutorCore, OptimizationConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TestData:
    """Test data for demonstration."""

    id: int
    content: str
    metadata: dict[str, Any]


class OptimizationDemo:
    """Comprehensive demonstration of ExecutorCore optimizations."""

    def __init__(self):
        self.test_data = [
            TestData(1, "Sample content 1", {"category": "test", "priority": "high"}),
            TestData(2, "Sample content 2", {"category": "demo", "priority": "medium"}),
            TestData(3, "Sample content 3", {"category": "test", "priority": "low"}),
        ]

        # Performance tracking
        self.performance_history = []
        self.baseline_metrics = None

    async def run_comprehensive_demo(self):
        """Run the comprehensive optimization demonstration."""
        logger.info("Starting Comprehensive ExecutorCore Optimization Demo")
        logger.info("=" * 60)

        # 1. Baseline Performance Test
        await self.baseline_performance_test()

        # 2. Memory Optimization Demo
        await self.memory_optimization_demo()

        # 3. Performance Optimization Demo
        await self.performance_optimization_demo()

        # 4. Error Handling Demo
        await self.error_handling_demo()

        # 5. Monitoring and Metrics Demo
        await self.monitoring_demo()

        # 6. Configuration Management Demo
        await self.configuration_management_demo()

        # 7. Performance Comparison
        await self.performance_comparison()

        logger.info("Comprehensive Demo Completed!")
        logger.info("=" * 60)

    async def baseline_performance_test(self):
        """Test baseline performance without optimizations."""
        logger.info("1. Baseline Performance Test")
        logger.info("-" * 30)

        # Create baseline executor
        baseline_executor = ExecutorCore()

        # Define test steps
        async def simple_step(data: TestData) -> str:
            await asyncio.sleep(0.1)  # Simulate work
            return f"Processed: {data.content}"

        async def complex_step(data: TestData) -> dict[str, Any]:
            await asyncio.sleep(0.2)  # Simulate complex work
            return {
                "id": data.id,
                "processed_content": data.content.upper(),
                "metadata": data.metadata,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Test baseline performance
        start_time = time.perf_counter()

        results = []
        for data in self.test_data:
            # Simple step
            simple_result = await baseline_executor.execute(simple_step, data)
            results.append(simple_result)

            # Complex step
            complex_result = await baseline_executor.execute(complex_step, data)
            results.append(complex_result)

        end_time = time.perf_counter()
        baseline_time = (end_time - start_time) * 1000  # Convert to milliseconds

        logger.info(f"Baseline execution time: {baseline_time:.2f}ms")
        logger.info(f"Total operations: {len(results)}")
        logger.info(f"Average time per operation: {baseline_time / len(results):.2f}ms")

        # Store baseline metrics
        self.baseline_metrics = {
            "total_time_ms": baseline_time,
            "operations": len(results),
            "avg_time_per_operation_ms": baseline_time / len(results),
        }

        logger.info("Baseline test completed\n")

    async def memory_optimization_demo(self):
        """Demonstrate memory optimizations."""
        logger.info("2. Memory Optimization Demo")
        logger.info("-" * 30)

        # Create optimized executor with memory optimizations
        memory_config = OptimizationConfig(
            # Enable memory optimizations
            enable_object_pool=True,
            enable_context_optimization=True,
            enable_memory_optimization=True,
            # Configure object pooling
            object_pool_max_size=50,
            object_pool_cleanup_threshold=0.8,
            # Disable other optimizations for this demo
            enable_step_optimization=False,
            enable_algorithm_optimization=False,
            enable_concurrency_optimization=False,
            enable_optimized_telemetry=False,
            enable_performance_monitoring=False,
            enable_optimized_error_handling=False,
            enable_circuit_breaker=False,
            enable_cache_optimization=False,
            enable_automatic_optimization=False,
        )

        memory_executor = ExecutorCore(optimization_config=memory_config)

        # Define memory-intensive steps
        async def memory_intensive_step(data: TestData) -> List[str]:
            # Simulate memory allocation
            large_list = [f"item_{i}" for i in range(1000)]
            processed = [f"processed_{item}" for item in large_list]
            return processed[:10]  # Return subset

        async def context_intensive_step(data: TestData, context: dict[str, Any]) -> dict[str, Any]:
            # Simulate context manipulation
            context_copy = context.copy()
            context_copy[f"processed_{data.id}"] = {
                "content": data.content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            return context_copy

        # Test memory optimizations
        logger.info("Testing memory optimizations...")

        start_time = time.perf_counter()

        # Test object pooling
        for i in range(10):
            await memory_executor.execute(memory_intensive_step, self.test_data[0])

        # Test context optimization
        context = {"session_id": "demo_session", "user_id": "demo_user"}
        for data in self.test_data:
            await memory_executor.execute(context_intensive_step, data, context=context)

        end_time = time.perf_counter()
        memory_time = (end_time - start_time) * 1000

        # Get memory statistics
        stats = memory_executor.get_optimization_stats()
        memory_stats = stats.get("memory", {})

        logger.info(f"Memory optimization execution time: {memory_time:.2f}ms")
        logger.info(
            f"Object pool utilization: {memory_stats.get('object_pool_utilization', 'N/A')}%"
        )
        logger.info(f"Memory usage: {memory_stats.get('usage_mb', 'N/A')}MB")

        # Calculate improvement
        if self.baseline_metrics:
            improvement = (
                (self.baseline_metrics["avg_time_per_operation_ms"] - (memory_time / 13))
                / self.baseline_metrics["avg_time_per_operation_ms"]
            ) * 100
            logger.info(f"Memory optimization improvement: {improvement:.1f}%")

        logger.info("Memory optimization demo completed\n")

    async def performance_optimization_demo(self):
        """Demonstrate performance optimizations."""
        logger.info("3. Performance Optimization Demo")
        logger.info("-" * 30)

        # Create optimized executor with performance optimizations
        performance_config = OptimizationConfig(
            # Enable performance optimizations
            enable_step_optimization=True,
            enable_algorithm_optimization=True,
            enable_concurrency_optimization=True,
            # Configure concurrency
            max_concurrent_executions=4,
            # Disable other optimizations for this demo
            enable_object_pool=False,
            enable_context_optimization=False,
            enable_memory_optimization=False,
            enable_optimized_telemetry=False,
            enable_performance_monitoring=False,
            enable_optimized_error_handling=False,
            enable_circuit_breaker=False,
            enable_cache_optimization=False,
            enable_automatic_optimization=False,
        )

        performance_executor = ExecutorCore(optimization_config=performance_config)

        # Define performance-intensive steps
        async def cpu_intensive_step(data: TestData) -> str:
            # Simulate CPU-intensive work
            result = ""
            for i in range(1000):
                result += str(i % 10)
            return f"CPU processed: {data.content} - {len(result)} chars"

        async def parallel_step(data: TestData) -> List[str]:
            # Simulate parallel processing
            tasks = []
            for i in range(5):

                async def worker(worker_id: int) -> str:
                    await asyncio.sleep(0.05)
                    return f"Worker {worker_id} processed {data.content}"

                tasks.append(worker(i))

            results = await asyncio.gather(*tasks)
            return results

        # Test performance optimizations
        logger.info("Testing performance optimizations...")

        start_time = time.perf_counter()

        # Test CPU-intensive operations
        cpu_tasks = []
        for data in self.test_data:
            cpu_tasks.append(performance_executor.execute(cpu_intensive_step, data))

        await asyncio.gather(*cpu_tasks)

        # Test parallel processing
        parallel_tasks = []
        for data in self.test_data:
            parallel_tasks.append(performance_executor.execute(parallel_step, data))

        await asyncio.gather(*parallel_tasks)

        end_time = time.perf_counter()
        performance_time = (end_time - start_time) * 1000

        # Get performance statistics
        stats = performance_executor.get_optimization_stats()
        execution_stats = stats.get("execution", {})
        concurrency_stats = stats.get("concurrency", {})

        logger.info(f"Performance optimization execution time: {performance_time:.2f}ms")
        logger.info(f"Average execution time: {execution_stats.get('avg_time_ms', 'N/A')}ms")
        logger.info(f"Concurrency level: {concurrency_stats.get('current_level', 'N/A')}")

        # Calculate improvement
        if self.baseline_metrics:
            improvement = (
                (self.baseline_metrics["avg_time_per_operation_ms"] - (performance_time / 6))
                / self.baseline_metrics["avg_time_per_operation_ms"]
            ) * 100
            logger.info(f"Performance optimization improvement: {improvement:.1f}%")

        logger.info("Performance optimization demo completed\n")

    async def error_handling_demo(self):
        """Demonstrate error handling optimizations."""
        logger.info("4. Error Handling Demo")
        logger.info("-" * 30)

        # Create optimized executor with error handling
        error_config = OptimizationConfig(
            # Enable error handling optimizations
            enable_optimized_error_handling=True,
            enable_circuit_breaker=True,
            # Configure error handling
            error_cache_size=50,
            circuit_breaker_failure_threshold=3,
            circuit_breaker_recovery_timeout_seconds=30,
            # Disable other optimizations for this demo
            enable_object_pool=False,
            enable_context_optimization=False,
            enable_memory_optimization=False,
            enable_step_optimization=False,
            enable_algorithm_optimization=False,
            enable_concurrency_optimization=False,
            enable_optimized_telemetry=False,
            enable_performance_monitoring=False,
            enable_cache_optimization=False,
            enable_automatic_optimization=False,
        )

        error_executor = ExecutorCore(optimization_config=error_config)

        # Define steps that may fail
        async def unreliable_step(data: TestData) -> str:
            # Simulate unreliable operation
            if data.id % 3 == 0:  # Fail every third request
                raise ValueError(f"Simulated error for data {data.id}")
            await asyncio.sleep(0.1)
            return f"Successfully processed: {data.content}"

        async def slow_step(data: TestData) -> str:
            # Simulate slow operation
            await asyncio.sleep(0.5)
            return f"Slow processed: {data.content}"

        # Test error handling
        logger.info("Testing error handling optimizations...")

        # Test with circuit breaker
        results = []
        errors = []

        for i in range(10):
            try:
                result = await error_executor.execute(
                    unreliable_step, self.test_data[i % len(self.test_data)]
                )
                results.append(result)
            except Exception as e:
                errors.append(str(e))
                logger.warning(f"Error caught: {e}")

        # Test circuit breaker with slow operations
        try:
            for i in range(5):
                result = await error_executor.execute_with_circuit_breaker(
                    slow_step, self.test_data[0], circuit_breaker_name="slow_operations"
                )
                results.append(result)
        except Exception as e:
            logger.warning(f"Circuit breaker triggered: {e}")

        # Get error statistics
        stats = error_executor.get_optimization_stats()
        error_stats = stats.get("error_handling", {})

        logger.info(f"Successful operations: {len(results)}")
        logger.info(f"Failed operations: {len(errors)}")
        logger.info(f"Error rate: {error_stats.get('error_rate', 'N/A')}%")
        logger.info(f"Recovery success rate: {error_stats.get('recovery_success_rate', 'N/A')}%")
        logger.info(f"Cached error patterns: {error_stats.get('cached_patterns', 'N/A')}")

        logger.info("Error handling demo completed\n")

    async def monitoring_demo(self):
        """Demonstrate monitoring and metrics."""
        logger.info("5. Monitoring and Metrics Demo")
        logger.info("-" * 30)

        # Create optimized executor with monitoring
        monitoring_config = OptimizationConfig(
            # Enable monitoring
            enable_optimized_telemetry=True,
            enable_performance_monitoring=True,
            # Configure telemetry
            telemetry_batch_size=5,
            telemetry_flush_interval_seconds=10.0,
            # Enable some optimizations for monitoring
            enable_object_pool=True,
            enable_step_optimization=True,
            # Disable others for this demo
            enable_context_optimization=False,
            enable_memory_optimization=False,
            enable_algorithm_optimization=False,
            enable_concurrency_optimization=False,
            enable_optimized_error_handling=False,
            enable_circuit_breaker=False,
            enable_cache_optimization=False,
            enable_automatic_optimization=False,
        )

        monitoring_executor = ExecutorCore(optimization_config=monitoring_config)

        # Define monitored steps
        async def monitored_step(data: TestData) -> dict[str, Any]:
            await asyncio.sleep(0.1)
            return {
                "processed": True,
                "data_id": data.id,
                "content_length": len(data.content),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Test monitoring
        logger.info("Testing monitoring and metrics...")

        # Execute with monitoring
        for i, data in enumerate(self.test_data):
            result, metrics = await monitoring_executor.execute_with_monitoring(
                monitored_step, data
            )

            logger.info(f"Step {i + 1} metrics:")
            logger.info(f"  Execution time: {metrics.get('execution_time_ms', 'N/A')}ms")
            logger.info(f"  Memory usage: {metrics.get('memory_usage_mb', 'N/A')}MB")
            logger.info(f"  Cache hit rate: {metrics.get('cache_hit_rate', 'N/A')}%")

        # Get comprehensive statistics
        stats = monitoring_executor.get_optimization_stats()

        logger.info("\nComprehensive Statistics:")
        logger.info(f"Memory stats: {json.dumps(stats.get('memory', {}), indent=2)}")
        logger.info(f"Execution stats: {json.dumps(stats.get('execution', {}), indent=2)}")
        logger.info(f"Cache stats: {json.dumps(stats.get('cache', {}), indent=2)}")
        logger.info(f"Telemetry stats: {json.dumps(stats.get('telemetry', {}), indent=2)}")

        # Get performance recommendations
        recommendations = monitoring_executor.get_performance_recommendations()
        logger.info("\nPerformance Recommendations:")
        for rec in recommendations:
            logger.info(f"  - {rec.get('description', 'N/A')}")
            logger.info(f"    Impact: {rec.get('impact', 'N/A')}")
            logger.info(f"    Priority: {rec.get('priority', 'N/A')}")

        logger.info("Monitoring demo completed\n")

    async def configuration_management_demo(self):
        """Demonstrate configuration management."""
        logger.info("6. Configuration Management Demo")
        logger.info("-" * 30)

        # Create executor with configuration management
        config_manager = ExecutorCore()

        # Test configuration export/import
        logger.info("Testing configuration management...")

        # Export current configuration
        config_json = config_manager.export_config(format="json")
        logger.info(f"Exported configuration: {config_json[:200]}...")

        # Import configuration
        config_dict = json.loads(config_json)
        await config_manager.import_config(config_dict, format="dict")
        logger.info("Configuration imported successfully")

        # Test partial configuration updates
        await config_manager.update_config_partial(
            object_pool_max_size=100, telemetry_batch_size=20, cache_max_size=1000
        )
        logger.info("Partial configuration updated")

        # Get configuration recommendations
        recommendations = config_manager.get_config_recommendations()
        logger.info(f"Configuration recommendations: {len(recommendations)} found")

        for rec in recommendations:
            logger.info(f"  - {rec.get('description', 'N/A')}")
            logger.info(f"    Current: {rec.get('current', 'N/A')}")
            logger.info(f"    Recommended: {rec.get('recommended', 'N/A')}")

        # Apply recommended configuration
        if recommendations:
            await config_manager.apply_recommended_config(recommendations)
            logger.info("Applied recommended configuration")

        logger.info("Configuration management demo completed\n")

    async def performance_comparison(self):
        """Compare performance across different configurations."""
        logger.info("7. Performance Comparison")
        logger.info("-" * 30)

        # Define test step
        async def benchmark_step(data: TestData) -> str:
            await asyncio.sleep(0.1)
            return f"Benchmark processed: {data.content}"

        # Test different configurations
        configurations = {
            "Baseline": ExecutorCore(),
            "Memory Optimized": ExecutorCore(
                OptimizationConfig(
                    enable_object_pool=True,
                    enable_context_optimization=True,
                    enable_memory_optimization=True,
                    enable_step_optimization=False,
                    enable_algorithm_optimization=False,
                    enable_concurrency_optimization=False,
                    enable_optimized_telemetry=False,
                    enable_performance_monitoring=False,
                    enable_optimized_error_handling=False,
                    enable_circuit_breaker=False,
                    enable_cache_optimization=False,
                    enable_automatic_optimization=False,
                )
            ),
            "Performance Optimized": ExecutorCore(
                OptimizationConfig(
                    enable_object_pool=False,
                    enable_context_optimization=False,
                    enable_memory_optimization=False,
                    enable_step_optimization=True,
                    enable_algorithm_optimization=True,
                    enable_concurrency_optimization=True,
                    enable_optimized_telemetry=False,
                    enable_performance_monitoring=False,
                    enable_optimized_error_handling=False,
                    enable_circuit_breaker=False,
                    enable_cache_optimization=False,
                    enable_automatic_optimization=False,
                )
            ),
            "Fully Optimized": ExecutorCore(
                OptimizationConfig(
                    enable_object_pool=True,
                    enable_context_optimization=True,
                    enable_memory_optimization=True,
                    enable_step_optimization=True,
                    enable_algorithm_optimization=True,
                    enable_concurrency_optimization=True,
                    enable_optimized_telemetry=True,
                    enable_performance_monitoring=True,
                    enable_optimized_error_handling=True,
                    enable_circuit_breaker=True,
                    enable_cache_optimization=True,
                    enable_automatic_optimization=True,
                )
            ),
        }

        results = {}

        for config_name, executor in configurations.items():
            logger.info(f"Testing {config_name}...")

            start_time = time.perf_counter()

            # Run benchmark
            tasks = []
            for data in self.test_data:
                for _ in range(5):  # 5 iterations per data item
                    tasks.append(executor.execute(benchmark_step, data))

            await asyncio.gather(*tasks)

            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000

            results[config_name] = {
                "execution_time_ms": execution_time,
                "operations": len(tasks),
                "avg_time_per_operation_ms": execution_time / len(tasks),
            }

            # Get optimization stats if available
            if hasattr(executor, "get_optimization_stats"):
                stats = executor.get_optimization_stats()
                results[config_name]["stats"] = stats

        # Print comparison
        logger.info("\nPerformance Comparison Results:")
        logger.info("-" * 50)

        baseline_time = results["Baseline"]["avg_time_per_operation_ms"]

        for config_name, result in results.items():
            improvement = (
                (baseline_time - result["avg_time_per_operation_ms"]) / baseline_time
            ) * 100
            logger.info(f"{config_name}:")
            logger.info(f"  Avg time per operation: {result['avg_time_per_operation_ms']:.2f}ms")
            logger.info(f"  Total time: {result['execution_time_ms']:.2f}ms")
            logger.info(f"  Improvement vs baseline: {improvement:.1f}%")

            if "stats" in result:
                memory_usage = result["stats"].get("memory", {}).get("usage_mb", "N/A")
                logger.info(f"  Memory usage: {memory_usage}MB")

        logger.info("Performance comparison completed\n")

    def print_summary(self):
        """Print demo summary."""
        logger.info("Demo Summary")
        logger.info("=" * 60)
        logger.info("This demo demonstrated:")
        logger.info("✅ Memory optimizations (object pooling, context management)")
        logger.info("✅ Performance optimizations (step execution, concurrency)")
        logger.info("✅ Error handling optimizations (circuit breaker, error caching)")
        logger.info("✅ Monitoring and metrics collection")
        logger.info("✅ Configuration management and recommendations")
        logger.info("✅ Performance comparison across different configurations")
        logger.info("")
        logger.info("Key takeaways:")
        logger.info("- Optimizations provide significant performance improvements")
        logger.info("- Different workloads benefit from different optimization combinations")
        logger.info("- Monitoring helps identify optimization opportunities")
        logger.info("- Configuration management enables dynamic optimization")
        logger.info("")
        logger.info("For production use:")
        logger.info("- Start with default configuration")
        logger.info("- Monitor performance metrics")
        logger.info("- Apply optimizations based on workload characteristics")
        logger.info("- Use automatic optimization for dynamic environments")


async def main():
    """Main demo function."""
    demo = OptimizationDemo()

    try:
        await demo.run_comprehensive_demo()
        demo.print_summary()
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
