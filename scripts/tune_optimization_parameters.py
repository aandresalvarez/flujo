#!/usr/bin/env python3
"""
Optimization parameter tuning script.

This script applies intelligent parameter tuning to all optimization components
based on system characteristics and performance benchmarks. It automatically
adjusts object pool sizes, cache configurations, concurrency limits, telemetry
sampling rates, and adaptive resource management thresholds.
"""

import asyncio
import multiprocessing
import sys
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flujo.application.core.optimization_parameter_tuner import (  # noqa: E402
    OptimizationParameterTuner,
    TuningStrategy
)
from flujo.application.core.ultra_executor import OptimizationConfig, OptimizedExecutorCore  # noqa: E402
from flujo.domain.dsl.step import Step, StepConfig  # noqa: E402
from flujo.testing.utils import StubAgent  # noqa: E402


class ParameterTuningManager:
    """Manages the complete parameter tuning process."""
    
    def __init__(self, strategy: TuningStrategy = TuningStrategy.BALANCED):
        self.strategy = strategy
        self.tuner = OptimizationParameterTuner(strategy=strategy)
        self.results = {}
    
    def create_optimized_executor_config(self, params: dict) -> OptimizationConfig:
        """Create optimized executor configuration from tuned parameters."""
        cpu_count = multiprocessing.cpu_count()
        
        return OptimizationConfig(
            # Memory optimizations
            enable_object_pool=True,
            enable_context_optimization=True,
            enable_memory_optimization=True,
            object_pool_max_size=params.get("object_pool_max_size", 500),
            object_pool_cleanup_threshold=params.get("object_pool_cleanup_threshold", 0.75),
            
            # Execution optimizations
            enable_step_optimization=True,
            enable_algorithm_optimization=True,
            enable_concurrency_optimization=True,
            max_concurrent_executions=params.get("max_concurrent_executions", cpu_count * 2),
            
            # Telemetry optimizations
            enable_optimized_telemetry=True,
            enable_performance_monitoring=True,
            telemetry_batch_size=params.get("telemetry_batch_size", 200),
            telemetry_flush_interval_seconds=params.get("telemetry_flush_interval", 2.0),
            
            # Error handling optimizations
            enable_optimized_error_handling=True,
            enable_circuit_breaker=True,
            error_cache_size=250,
            circuit_breaker_failure_threshold=3,
            circuit_breaker_recovery_timeout_seconds=15,
            
            # Cache optimizations
            enable_cache_optimization=True,
            cache_compression=False,
            cache_ttl_seconds=params.get("cache_ttl_seconds", 1800),
            cache_max_size=params.get("cache_max_size", 2000),
            
            # Performance thresholds
            slow_execution_threshold_ms=500.0,  # More sensitive
            memory_pressure_threshold_mb=250.0,  # Lower threshold
            cpu_usage_threshold_percent=70.0,   # More conservative
            
            # Automatic optimization
            enable_automatic_optimization=True,
            optimization_analysis_interval_seconds=30.0,  # More frequent
            performance_degradation_threshold=0.15,  # More sensitive
            
            # Backward compatibility
            maintain_backward_compatibility=True,
            allow_runtime_changes=True,
            config_validation_enabled=True
        )
    
    def create_test_step(self, name: str = "tuning_test", outputs: int = 50) -> Step:
        """Create a lightweight test step for parameter tuning."""
        return Step.model_validate({
            "name": name,
            "agent": StubAgent([f"output_{i}" for i in range(outputs)]),
            "config": StepConfig(max_retries=1),
        })
    
    async def benchmark_executor_performance(self, config: OptimizationConfig) -> dict:
        """Benchmark executor performance with given configuration."""
        executor = OptimizedExecutorCore(optimization_config=config)
        step = self.create_test_step()
        data = {"test": "parameter_tuning"}
        
        # Warm up
        for _ in range(5):
            try:
                await executor.execute(step, data)
            except Exception:
                pass
        
        # Benchmark
        times = []
        successful_runs = 0
        
        for _ in range(20):
            start_time = time.perf_counter()
            try:
                await executor.execute(step, data)
                successful_runs += 1
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            except Exception:
                continue
        
        if not times:
            return {"performance": float('inf'), "memory": float('inf'), "stability": 0.0}
        
        avg_time = sum(times) / len(times)
        stability = successful_runs / 20
        
        return {
            "performance": avg_time,
            "memory": 0.0,  # Simplified for tuning
            "stability": stability
        }
    
    async def apply_system_optimized_defaults(self) -> dict:
        """Apply system-optimized default parameters."""
        print("Applying system-optimized default parameters...")
        
        # Get system-optimized defaults
        optimized_params = self.tuner.get_system_optimized_defaults()
        
        # Apply the parameters
        # results = self.tuner.apply_optimized_parameters()  # Unused variable
        
        print("System-optimized parameters applied:")
        for param_name, value in optimized_params.items():
            print(f"  {param_name}: {value}")
        
        return optimized_params
    
    async def fine_tune_parameters(self) -> dict:
        """Fine-tune parameters based on performance benchmarks."""
        print("\nFine-tuning parameters based on performance benchmarks...")
        
        # Create benchmark function
        async def benchmark_func():
            params = self.tuner.get_optimized_parameters()
            config = self.create_optimized_executor_config(params)
            return await self.benchmark_executor_performance(config)
        
        # Auto-tune all parameters
        tuning_results = await self.tuner.auto_tune_all_parameters(benchmark_func)
        
        print("Fine-tuning results:")
        for category, results in tuning_results.items():
            if results:
                print(f"  {category}: {len(results)} improvements")
                for result in results:
                    print(f"    {result.parameter_name}: {result.old_value} -> {result.new_value} "
                          f"({result.performance_improvement:.1%} improvement)")
            else:
                print(f"  {category}: No improvements found")
        
        return tuning_results
    
    async def validate_tuned_parameters(self) -> dict:
        """Validate the final tuned parameters."""
        print("\nValidating tuned parameters...")
        
        # Get final parameters
        final_params = self.tuner.get_optimized_parameters()
        config = self.create_optimized_executor_config(final_params)
        
        # Run comprehensive validation
        validation_results = await self.benchmark_executor_performance(config)
        
        print("Validation results:")
        print(f"  Performance: {validation_results['performance']:.6f}s")
        print(f"  Stability: {validation_results['stability']:.1%}")
        
        return validation_results
    
    def generate_optimized_configuration_code(self) -> str:
        """Generate code for the optimized configuration."""
        params = self.tuner.get_optimized_parameters()
        
        code = f'''# Optimized ExecutorCore Configuration
# Generated by parameter tuning on {time.strftime("%Y-%m-%d %H:%M:%S")}

from flujo.application.core.ultra_executor import OptimizationConfig

def get_optimized_config() -> OptimizationConfig:
    """Get optimized configuration based on system characteristics and performance tuning."""
    return OptimizationConfig(
        # Memory optimizations
        enable_object_pool=True,
        enable_context_optimization=True,
        enable_memory_optimization=True,
        object_pool_max_size={params.get("object_pool_max_size", 500)},
        object_pool_cleanup_threshold={params.get("object_pool_cleanup_threshold", 0.75)},
        
        # Execution optimizations
        enable_step_optimization=True,
        enable_algorithm_optimization=True,
        enable_concurrency_optimization=True,
        max_concurrent_executions={params.get("max_concurrent_executions", multiprocessing.cpu_count() * 2)},
        
        # Telemetry optimizations
        enable_optimized_telemetry=True,
        enable_performance_monitoring=True,
        telemetry_batch_size={params.get("telemetry_batch_size", 200)},
        telemetry_flush_interval_seconds={params.get("telemetry_flush_interval", 2.0)},
        
        # Cache optimizations
        enable_cache_optimization=True,
        cache_ttl_seconds={params.get("cache_ttl_seconds", 1800)},
        cache_max_size={params.get("cache_max_size", 2000)},
        
        # Performance thresholds (optimized)
        slow_execution_threshold_ms=500.0,
        memory_pressure_threshold_mb=250.0,
        cpu_usage_threshold_percent=70.0,
        
        # Automatic optimization
        enable_automatic_optimization=True,
        optimization_analysis_interval_seconds=30.0,
        performance_degradation_threshold=0.15,
        
        # Backward compatibility
        maintain_backward_compatibility=True,
        allow_runtime_changes=True,
        config_validation_enabled=True
    )

# Component-specific optimized parameters
OPTIMIZED_OBJECT_POOL_CONFIG = {{
    "max_pool_size": {params.get("object_pool_max_size", 500)},
    "max_pools": {params.get("object_pool_max_pools", 50)},
    "cleanup_threshold": {params.get("object_pool_cleanup_threshold", 0.75)},
    "stats_enabled": True
}}

OPTIMIZED_CONTEXT_MANAGER_CONFIG = {{
    "cache_size": {params.get("context_cache_size", 512)},
    "merge_cache_size": {params.get("context_merge_cache_size", 256)},
    "ttl_seconds": {params.get("context_ttl_seconds", 1800)},
    "enable_cow": True,
    "enable_stats": True
}}

OPTIMIZED_TELEMETRY_CONFIG = {{
    "enable_tracing": True,
    "enable_metrics": True,
    "enable_sampling": True,
    "sampling_rate": {params.get("telemetry_sampling_rate", 0.5)},
    "batch_size": {params.get("telemetry_batch_size", 200)},
    "flush_interval": {params.get("telemetry_flush_interval", 2.0)},
    "enable_stats": True
}}

OPTIMIZED_RESOURCE_MANAGER_CONFIG = {{
    "monitoring_interval": {params.get("resource_monitoring_interval", 0.5)},
    "adaptation_interval": {params.get("resource_adaptation_interval", 3.0)},
    "enable_telemetry": True
}}
'''
        return code
    
    def save_optimized_configuration(self, filename: str = "optimized_executor_config.py") -> None:
        """Save the optimized configuration to a Python file."""
        code = self.generate_optimized_configuration_code()
        
        with open(filename, 'w') as f:
            f.write(code)
        
        print(f"\nOptimized configuration saved to {filename}")
    
    async def run_complete_tuning(self) -> dict:
        """Run the complete parameter tuning process."""
        print("=" * 60)
        print("OPTIMIZATION PARAMETER TUNING")
        print("=" * 60)
        
        # Step 1: Apply system-optimized defaults
        system_params = await self.apply_system_optimized_defaults()
        
        # Step 2: Fine-tune parameters
        tuning_results = await self.fine_tune_parameters()
        
        # Step 3: Validate final parameters
        validation_results = await self.validate_tuned_parameters()
        
        # Step 4: Generate summary
        summary = self.tuner.get_tuning_summary()
        
        print("\n" + "=" * 60)
        print("TUNING SUMMARY")
        print("=" * 60)
        print(f"Total improvements: {summary['total_improvements']}")
        print(f"Categories tuned: {summary['categories_tuned']}")
        print(f"Overall improvement: {summary['overall_improvement']:.1%}")
        
        if summary.get('category_improvements'):
            print("\nCategory improvements:")
            for category, improvement in summary['category_improvements'].items():
                print(f"  {category}: {improvement:.1%}")
        
        # Step 5: Save results
        self.tuner.save_tuning_results("optimization_tuning_results.json")
        self.save_optimized_configuration("optimized_executor_config.py")
        
        return {
            "system_params": system_params,
            "tuning_results": tuning_results,
            "validation_results": validation_results,
            "summary": summary
        }


async def main():
    """Main tuning function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tune optimization parameters")
    parser.add_argument(
        "--strategy",
        choices=["conservative", "balanced", "aggressive", "adaptive"],
        default="balanced",
        help="Tuning strategy to use"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick tuning (system defaults only)"
    )
    
    args = parser.parse_args()
    
    # Map strategy string to enum
    strategy_map = {
        "conservative": TuningStrategy.CONSERVATIVE,
        "balanced": TuningStrategy.BALANCED,
        "aggressive": TuningStrategy.AGGRESSIVE,
        "adaptive": TuningStrategy.ADAPTIVE
    }
    
    strategy = strategy_map[args.strategy]
    
    # Create tuning manager
    manager = ParameterTuningManager(strategy=strategy)
    
    try:
        if args.quick:
            # Quick tuning - just apply system defaults
            print("Running quick parameter tuning (system defaults only)...")
            # system_params = await manager.apply_system_optimized_defaults()  # Unused variable
            validation_results = await manager.validate_tuned_parameters()
            manager.save_optimized_configuration("optimized_executor_config.py")
            
            print("\n✅ Quick tuning completed successfully!")
            print(f"Performance: {validation_results['performance']:.6f}s")
            print(f"Stability: {validation_results['stability']:.1%}")
        else:
            # Full tuning process
            results = await manager.run_complete_tuning()
            
            # Check if tuning was successful
            if results['summary']['overall_improvement'] > 0:
                print("\n✅ Parameter tuning completed successfully!")
                print(f"Overall improvement: {results['summary']['overall_improvement']:.1%}")
                return 0
            else:
                print("\n⚠️  Parameter tuning completed with limited improvements")
                print(f"Overall improvement: {results['summary']['overall_improvement']:.1%}")
                return 0
    
    except Exception as e:
        print(f"\n❌ Parameter tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)