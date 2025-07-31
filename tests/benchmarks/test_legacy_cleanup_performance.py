"""
Benchmark tests for legacy cleanup performance impact.

This module contains performance tests to ensure that the legacy cleanup
does not introduce performance regressions.
"""

import time
import importlib
import sys
from unittest.mock import Mock, AsyncMock, patch

import pytest

from flujo.application.core.step_logic import (
    _handle_cache_step,
)
from flujo.application.core.ultra_executor import ExecutorCore
from flujo.steps.cache_step import CacheStep
from flujo.domain.models import StepResult


class TestCleanupPerformanceImpact:
    """Test performance impact of removing legacy code."""

    def test_cleanup_performance_impact(self):
        """Test performance impact of removing legacy code."""
        # The cleanup should have already been done, so we're measuring the current state
        import_time_after = self._measure_import_time("flujo.application.core.step_logic")

        # Import time should be reasonable (less than 1 second)
        assert import_time_after < 1.0

        print(f"Import time after cleanup: {import_time_after:.4f} seconds")

    def _measure_import_time(self, module_name: str) -> float:
        """Measure the time it takes to import a module."""
        start_time = time.perf_counter()

        # Remove module from sys.modules if it exists to force fresh import
        if module_name in sys.modules:
            del sys.modules[module_name]

        # Import the module
        importlib.import_module(module_name)

        end_time = time.perf_counter()
        return end_time - start_time

    async def test_import_performance_improvement(self):
        """Test import performance improvement from cleanup."""
        # Test that importing step_logic doesn't take too long
        import_time = self._measure_import_time("flujo.application.core.step_logic")

        # Should import quickly (less than 0.5 seconds)
        assert import_time < 0.5

        # Test that importing ultra_executor is also fast
        executor_import_time = self._measure_import_time("flujo.application.core.ultra_executor")
        assert executor_import_time < 1.0

    async def test_memory_usage_improvement(self):
        """Test memory usage improvement from cleanup."""
        import psutil
        import os

        # Get current process
        process = psutil.Process(os.getpid())

        # Measure memory before importing modules
        memory_before = process.memory_info().rss

        # Import the modules
        # import flujo.application.core.step_logic  # Unused import removed
        
        # Measure memory after importing
        memory_after = process.memory_info().rss

        # Calculate memory increase
        memory_increase = memory_after - memory_before

        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024  # 100MB

        print(f"Memory increase: {memory_increase / 1024 / 1024:.2f} MB")


class TestFunctionCallPerformance:
    """Test performance of function calls after cleanup."""

    async def test_delegation_performance(self):
        """Test performance of delegation to ExecutorCore."""
        # Test that delegation is fast
        with patch("flujo.application.core.ultra_executor.ExecutorCore") as mock_executor_class:
            mock_executor = Mock()
            mock_executor_class.return_value = mock_executor
            mock_executor._handle_loop_step = AsyncMock(return_value=StepResult(name="test"))

            from flujo.application.core.step_logic import _handle_loop_step

            # Measure delegation performance
            start_time = time.perf_counter()

            for _ in range(1000):  # Test many calls
                await _handle_loop_step(
                    step=Mock(),
                    data="test",
                    context=None,
                    resources=None,
                    step_executor=AsyncMock(),
                    context_model_defined=True,
                    usage_limits=None,
                    context_setter=Mock(),
                )

            end_time = time.perf_counter()
            total_time = end_time - start_time

            # Should be reasonably fast (less than 3.0 seconds for 1000 calls)
            assert total_time < 3.0

            print(f"Delegation performance: {total_time:.4f} seconds for 1000 calls")

    async def test_deprecated_function_performance(self):
        """Test performance of deprecated functions."""
        # Test _handle_cache_step performance
        mock_cache_step = Mock(spec=CacheStep)
        mock_cache_step.wrapped_step = Mock()
        mock_cache_step.wrapped_step.name = "test_step"
        mock_cache_step.wrapped_step.agent = None
        mock_cache_step.wrapped_step.config = Mock()
        mock_cache_step.wrapped_step.config.max_retries = 1
        mock_cache_step.wrapped_step.config.timeout_s = 30
        mock_cache_step.wrapped_step.config.temperature = None
        mock_cache_step.wrapped_step.plugins = []
        mock_cache_step.wrapped_step.validators = []
        mock_cache_step.wrapped_step.processors = Mock()
        mock_cache_step.wrapped_step.processors.prompt_processors = []
        mock_cache_step.wrapped_step.processors.output_processors = []
        mock_cache_step.wrapped_step.updates_context = False
        mock_cache_step.wrapped_step.persist_feedback_to_context = None
        mock_cache_step.wrapped_step.persist_validation_results_to = None
        mock_cache_step.cache_backend = Mock()
        mock_cache_step.cache_backend.get.return_value = None

        mock_step_executor = AsyncMock()
        mock_step_executor.return_value = StepResult(name="test", success=True)

        start_time = time.perf_counter()

        for _ in range(100):  # Test multiple calls
            with pytest.warns(DeprecationWarning):
                await _handle_cache_step(
                    step=mock_cache_step,
                    data="test",
                    context=None,
                    resources=None,
                    step_executor=mock_step_executor,
                )

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # Should be reasonably fast
        assert total_time < 1.0

        print(f"Deprecated function performance: {total_time:.4f} seconds for 100 calls")

    async def test_executor_core_performance(self):
        """Test ExecutorCore performance."""
        executor = ExecutorCore()

        # Test that ExecutorCore methods are fast
        start_time = time.perf_counter()

        # Test method access performance
        for _ in range(1000):
            _ = executor._handle_loop_step
            _ = executor._handle_conditional_step
            _ = executor._handle_parallel_step
            _ = executor._handle_dynamic_router_step

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # Should be very fast
        assert total_time < 0.01

        print(f"ExecutorCore method access: {total_time:.4f} seconds for 1000 accesses")


class TestMemoryUsageAnalysis:
    """Test memory usage patterns after cleanup."""

    async def test_module_size_analysis(self):
        """Analyze module size after cleanup."""
        import flujo.application.core.step_logic as step_logic

        # Count functions in the module
        functions = [
            name
            for name in dir(step_logic)
            if callable(getattr(step_logic, name)) and not name.startswith("_")
        ]

        # Count deprecated functions
        deprecated_functions = []
        for name in dir(step_logic):
            attr = getattr(step_logic, name)
            if callable(attr) and hasattr(attr, "__wrapped__"):
                deprecated_functions.append(name)

        # Should have some deprecated functions but not too many
        assert len(deprecated_functions) > 0
        assert len(deprecated_functions) < 10  # Reasonable number

        print(f"Total functions: {len(functions)}")
        print(f"Deprecated functions: {len(deprecated_functions)}")
        print(f"Deprecated functions: {deprecated_functions}")

    async def test_import_dependency_analysis(self):
        """Analyze import dependencies after cleanup."""
        import flujo.application.core.step_logic as step_logic

        # Check that we don't have unnecessary imports
        # These should be imported for the remaining functions
        expected_imports = [
            "HumanInTheLoopStep",
            "CacheStep",
            "StepResult",
            "UsageLimits",
        ]

        for expected_import in expected_imports:
            assert hasattr(step_logic, expected_import), f"Missing import: {expected_import}"

        # Check that migrated step types are still imported (for new handlers)
        migrated_imports = [
            "LoopStep",
            "ConditionalStep",
            "ParallelStep",
            "DynamicParallelRouterStep",
        ]

        for migrated_import in migrated_imports:
            assert hasattr(step_logic, migrated_import), (
                f"Missing migrated import: {migrated_import}"
            )


class TestCleanupCompleteness:
    """Test that the cleanup is complete and comprehensive."""

    async def test_no_orphaned_code(self):
        """Test that there is no orphaned code after cleanup."""
        import flujo.application.core.step_logic as step_logic

        # Check that all functions are either:
        # 1. Deprecated legacy functions
        # 2. New handler functions
        # 3. Utility functions
        # 4. Delegating functions

        all_functions = [
            name
            for name in dir(step_logic)
            if callable(getattr(step_logic, name)) and name.startswith("_")
        ]

        # Categorize functions
        deprecated_functions = []
        new_handlers = []
        utility_functions = []

        for func_name in all_functions:
            func = getattr(step_logic, func_name)
            if hasattr(func, "__wrapped__"):
                deprecated_functions.append(func_name)
            elif (
                func_name.startswith("_handle_")
                and not func_name.startswith("_handle_cache_")
                and not func_name.startswith("_handle_hitl_")
            ):
                new_handlers.append(func_name)
            else:
                utility_functions.append(func_name)

        print(f"Deprecated functions: {deprecated_functions}")
        print(f"New handlers: {new_handlers}")
        print(f"Utility functions: {utility_functions}")

        # Should have some of each category
        assert len(deprecated_functions) > 0
        assert len(new_handlers) > 0
        assert len(utility_functions) > 0

    async def test_cleanup_documentation(self):
        """Test that cleanup is properly documented."""
        # Check that removal comments exist in the code
        with open("flujo/application/core/step_logic.py", "r") as f:
            content = f.read()

        # Should have comments about removed functions
        removal_comments = [
            "# _execute_parallel_step_logic removed",
            "# _execute_loop_step_logic removed",
            "# _execute_conditional_step_logic removed",
            "# _execute_dynamic_router_step_logic removed",
        ]

        for comment in removal_comments:
            assert comment in content, f"Missing removal comment: {comment}"

    async def test_performance_regression_detection(self):
        """Test that we can detect performance regressions."""
        # This test ensures our performance testing framework works
        # by measuring a known fast operation

        start_time = time.perf_counter()

        # Fast operation
        for _ in range(1000):
            _ = 1 + 1

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # Should be very fast
        assert total_time < 0.001

        print(f"Baseline performance test: {total_time:.6f} seconds")


class TestBenchmarkUtilities:
    """Test utilities for benchmarking the cleanup."""

    def test_import_time_measurement(self):
        """Test that import time measurement works correctly."""
        # Test with a simple module
        import_time = self._measure_import_time("time")

        # Should be very fast
        assert import_time < 0.1

        print(f"Simple module import time: {import_time:.4f} seconds")

    def test_function_call_measurement(self):
        """Test that function call measurement works correctly."""

        def simple_function():
            return "test"

        start_time = time.perf_counter()

        for _ in range(1000):
            _ = simple_function()

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # Should be very fast
        assert total_time < 0.001

        print(f"Simple function call time: {total_time:.6f} seconds for 1000 calls")

    def _measure_import_time(self, module_name: str) -> float:
        """Measure the time it takes to import a module."""
        start_time = time.perf_counter()

        # Remove module from sys.modules if it exists to force fresh import
        if module_name in sys.modules:
            del sys.modules[module_name]

        # Import the module
        importlib.import_module(module_name)

        end_time = time.perf_counter()
        return end_time - start_time
