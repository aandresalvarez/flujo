"""Performance benchmarks for ConditionalStep implementation."""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from flujo.domain.dsl.conditional import ConditionalStep
from flujo.domain.dsl import Pipeline, Step
from flujo.domain.models import StepResult
from flujo.application.core.executor_core import ExecutorCore
from flujo.domain.dsl.step import StepConfig


class TestConditionalStepPerformance:
    """Performance test suite for ConditionalStep implementation."""

    @pytest.fixture
    def executor_core(self):
        """Create ExecutorCore instance for testing."""
        return ExecutorCore()

    async def test_conditional_step_execution_performance(self, executor_core):
        """Test ConditionalStep execution performance."""
        conditional_step = ConditionalStep(
            name="test_conditional",
            condition_callable=lambda data, context: "branch_a",
            branches={
                "branch_a": Pipeline(
                    name="branch_a",
                    steps=[
                        Step(
                            name="test_agent",
                            agent=Mock(return_value="test_output"),
                            config=StepConfig(max_retries=1),
                        )
                    ],
                )
            },
        )

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_agent", success=True, output="test_output"
            )

            # Measure execution time
            start_time = time.perf_counter()
            result = await executor_core._handle_conditional_step(
                conditional_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )
            end_time = time.perf_counter()

            execution_time = end_time - start_time
            print(f"ConditionalStep execution time: {execution_time:.6f} seconds")

            assert result.success is True
            # Log performance (no tight assertion - micro-timing variance in CI)
            print(f"  Execution time: {execution_time * 1000:.2f}ms")
            assert execution_time < 1.0  # Sanity check: major regression only

    async def test_conditional_step_with_multiple_branches_performance(self, executor_core):
        """Test ConditionalStep with multiple branches performance."""
        conditional_step = ConditionalStep(
            name="test_conditional",
            condition_callable=lambda data, context: data.get("branch", "branch_a"),
            branches={
                "branch_a": Pipeline(name="branch_a", steps=[]),
                "branch_b": Pipeline(name="branch_b", steps=[]),
                "branch_c": Pipeline(name="branch_c", steps=[]),
            },
        )

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_agent", success=True, output="test_output"
            )

            start_time = time.perf_counter()
            result = await executor_core._handle_conditional_step(
                conditional_step,
                data={"branch": "branch_b"},
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )
            end_time = time.perf_counter()

            execution_time = end_time - start_time
            print(
                f"ConditionalStep with multiple branches execution time: {execution_time:.6f} seconds"
            )

            assert result.success is True
            assert execution_time < 0.1

    async def test_conditional_step_with_input_mapping_performance(self, executor_core):
        """Test ConditionalStep with input mapping performance."""
        conditional_step = ConditionalStep(
            name="test_conditional",
            condition_callable=lambda data, context: "branch_a",
            branch_input_mapper=lambda data, context: {"mapped": data},
            branches={
                "branch_a": Pipeline(
                    name="branch_a",
                    steps=[
                        Step(
                            name="test_agent",
                            agent=Mock(return_value="test_output"),
                            config=StepConfig(max_retries=1),
                        )
                    ],
                )
            },
        )

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_agent", success=True, output="test_output"
            )

            start_time = time.perf_counter()
            result = await executor_core._handle_conditional_step(
                conditional_step,
                data={"original": "data"},
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )
            end_time = time.perf_counter()

            execution_time = end_time - start_time
            print(
                f"ConditionalStep with input mapping execution time: {execution_time:.6f} seconds"
            )

            assert result.success is True
            assert execution_time < 0.1

    async def test_conditional_step_with_output_mapping_performance(self, executor_core):
        """Test ConditionalStep with output mapping performance."""
        conditional_step = ConditionalStep(
            name="test_conditional",
            condition_callable=lambda data, context: "branch_a",
            branch_output_mapper=lambda output, branch_key, context: {"mapped_output": output},
            branches={
                "branch_a": Pipeline(
                    name="branch_a",
                    steps=[
                        Step(
                            name="test_agent",
                            agent=Mock(return_value="test_output"),
                            config=StepConfig(max_retries=1),
                        )
                    ],
                )
            },
        )

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_agent", success=True, output="test_output"
            )

            start_time = time.perf_counter()
            result = await executor_core._handle_conditional_step(
                conditional_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )
            end_time = time.perf_counter()

            execution_time = end_time - start_time
            print(
                f"ConditionalStep with output mapping execution time: {execution_time:.6f} seconds"
            )

            assert result.success is True
            assert execution_time < 0.1

    async def test_conditional_step_memory_usage_performance(self, executor_core):
        """Test ConditionalStep memory usage performance."""
        large_data = {"large_key": "x" * 10000}  # 10KB of data

        conditional_step = ConditionalStep(
            name="test_conditional",
            condition_callable=lambda data, context: "branch_a",
            branches={
                "branch_a": Pipeline(
                    name="branch_a",
                    steps=[
                        Step(
                            name="test_agent",
                            agent=Mock(return_value=large_data),
                            config=StepConfig(max_retries=1),
                        )
                    ],
                )
            },
        )

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_agent", success=True, output=large_data
            )

            start_time = time.perf_counter()
            result = await executor_core._handle_conditional_step(
                conditional_step,
                data=large_data,
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )
            end_time = time.perf_counter()

            execution_time = end_time - start_time
            print(f"ConditionalStep with large data execution time: {execution_time:.6f} seconds")

            assert result.success is True
            assert execution_time < 0.1

    async def test_conditional_step_concurrent_execution_performance(self, executor_core):
        """Test ConditionalStep concurrent execution performance."""
        conditional_step = ConditionalStep(
            name="test_conditional",
            condition_callable=lambda data, context: "branch_a",
            branches={
                "branch_a": Pipeline(
                    name="branch_a",
                    steps=[
                        Step(
                            name="test_agent",
                            agent=Mock(return_value="test_output"),
                            config=StepConfig(max_retries=1),
                        )
                    ],
                )
            },
        )

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_agent", success=True, output="test_output"
            )

            start_time = time.perf_counter()
            tasks = []
            for i in range(10):
                task = executor_core._handle_conditional_step(
                    conditional_step,
                    data=f"test_data_{i}",
                    context=None,
                    resources=None,
                    limits=None,
                    context_setter=None,
                )
                tasks.append(task)
            results = await asyncio.gather(*tasks)
            end_time = time.perf_counter()

            execution_time = end_time - start_time
            print(f"ConditionalStep concurrent execution time: {execution_time:.6f} seconds")

            assert all(result.success for result in results)
            assert execution_time < 0.5  # Should complete 10 concurrent executions in under 500ms

    async def test_conditional_step_error_handling_performance(self, executor_core):
        """Test ConditionalStep error handling performance."""
        conditional_step = ConditionalStep(
            name="test_conditional",
            condition_callable=lambda data, context: "branch_a",
            branches={
                "branch_a": Pipeline(
                    name="branch_a",
                    steps=[
                        Step(
                            name="test_agent",
                            agent=Mock(side_effect=Exception("Test error")),
                            config=StepConfig(max_retries=1),
                        )
                    ],
                )
            },
        )

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.side_effect = Exception("Test error")

            start_time = time.perf_counter()
            result = await executor_core._handle_conditional_step(
                conditional_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )
            end_time = time.perf_counter()

            execution_time = end_time - start_time
            print(f"ConditionalStep error handling execution time: {execution_time:.6f} seconds")

            assert result.success is False
            assert execution_time < 0.1

    async def test_conditional_step_optimization_impact(self, executor_core):
        """Test the impact of performance optimizations on ConditionalStep execution."""
        # Test basic execution multiple times to measure consistency
        conditional_step = ConditionalStep(
            name="test_conditional",
            condition_callable=lambda data, context: "branch_a",
            branches={
                "branch_a": Pipeline(
                    name="branch_a",
                    steps=[
                        Step(
                            name="test_agent",
                            agent=Mock(return_value="test_output"),
                            config=StepConfig(max_retries=1),
                        )
                    ],
                )
            },
        )

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_agent", success=True, output="test_output"
            )

            execution_times = []
            for i in range(5):
                start_time = time.perf_counter()
                result = await executor_core._handle_conditional_step(
                    conditional_step,
                    data=f"test_data_{i}",
                    context=None,
                    resources=None,
                    limits=None,
                    context_setter=None,
                )
                end_time = time.perf_counter()
                execution_times.append(end_time - start_time)
                assert result.success is True

            avg_execution_time = sum(execution_times) / len(execution_times)
            print(f"Average ConditionalStep execution time: {avg_execution_time:.6f} seconds")
            print(f"Execution times: {[f'{t:.6f}' for t in execution_times]}")

            # Log performance metrics (no tight assertions - CI variance)
            variance = max(execution_times) - min(execution_times)
            print(f"  Variance: {variance * 1000:.2f}ms")
            # Sanity checks only - catches major regressions
            assert avg_execution_time < 1.0, f"Average too slow: {avg_execution_time:.3f}s"
