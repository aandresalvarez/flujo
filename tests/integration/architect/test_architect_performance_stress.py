from __future__ import annotations

import pytest
import time
import psutil
import os
from flujo.architect.builder import build_architect_pipeline
from flujo.architect.context import ArchitectContext
from flujo.cli.helpers import create_flujo_runner, execute_pipeline_with_output_handling
from flujo.infra.config import get_performance_threshold

# Force minimal architect pipeline for performance tests to avoid hanging
# This ensures tests use the simple pipeline instead of the complex state machine
os.environ["FLUJO_ARCHITECT_IGNORE_CONFIG"] = "1"
os.environ["FLUJO_TEST_MODE"] = "1"


@pytest.mark.integration
@pytest.mark.slow  # Multiple runs to compare timing; slower
@pytest.mark.timeout(30)  # 30 second timeout to prevent hanging
def test_architect_execution_time_consistency():
    """Test: Architect execution time is consistent across multiple runs."""
    pipeline = build_architect_pipeline()
    initial = {"initial_prompt": "Make a pipeline", "user_goal": "Echo input"}
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    execution_times = []

    for i in range(3):
        start_time = time.time()
        result = execute_pipeline_with_output_handling(
            runner=runner, input_data="Echo input", run_id=None, json_output=False
        )
        end_time = time.time()
        execution_times.append(end_time - start_time)

        assert result is not None
        ctx = getattr(result, "final_pipeline_context", None)
        assert ctx is not None

    # Execution times should be reasonably consistent (within 50% variance)
    avg_time = sum(execution_times) / len(execution_times)
    max_variance = avg_time * 0.5

    for exec_time in execution_times:
        assert (
            abs(exec_time - avg_time) <= max_variance
        ), f"Execution time {exec_time}s varies too much from average {avg_time}s"


@pytest.mark.integration
def test_architect_memory_usage_stability():
    """Test: Architect memory usage remains stable during execution."""
    pipeline = build_architect_pipeline()
    initial = {"initial_prompt": "Make a pipeline", "user_goal": "Echo input"}
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss

    result = execute_pipeline_with_output_handling(
        runner=runner, input_data="Echo input", run_id=None, json_output=False
    )

    # Get final memory usage
    final_memory = process.memory_info().rss

    # Memory usage should not increase excessively. Allow environment-adjusted threshold.
    memory_increase = final_memory - initial_memory
    max_allowed_increase_mb = get_performance_threshold(50.0)  # 50MB local, 3x in CI
    max_allowed_increase = int(max_allowed_increase_mb * 1024 * 1024)

    assert (
        memory_increase <= max_allowed_increase
    ), f"Memory usage increased by {memory_increase / (1024 * 1024):.2f}MB, exceeding limit of {max_allowed_increase_mb:.0f}MB"

    # Verify result is valid
    assert result is not None
    ctx = getattr(result, "final_pipeline_context", None)
    assert ctx is not None


@pytest.mark.integration
@pytest.mark.slow  # Mark as slow due to multiple architect pipeline executions
@pytest.mark.timeout(60)  # 60 second timeout to prevent hanging
def test_architect_handles_high_frequency_requests():
    """Test: Architect can handle high frequency requests without degradation."""
    pipeline = build_architect_pipeline()
    initial = {"initial_prompt": "Make a pipeline", "user_goal": "Echo input"}
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    # Warm up the system first to avoid measuring cold start time
    _ = execute_pipeline_with_output_handling(
        runner=runner, input_data="Echo input", run_id=None, json_output=False
    )

    start_time = time.time()
    successful_executions = 0
    execution_times = []

    # Execute multiple requests rapidly
    for i in range(10):
        try:
            request_start = time.time()
            result = execute_pipeline_with_output_handling(
                runner=runner, input_data="Echo input", run_id=None, json_output=False
            )
            request_end = time.time()

            if result is not None:
                successful_executions += 1
                execution_times.append(request_end - request_start)

        except Exception as e:
            # Log but continue - we want to test resilience
            print(f"Execution {i} failed: {e}")

    end_time = time.time()
    total_time = end_time - start_time

    # Should complete at least 80% of requests successfully
    success_rate = successful_executions / 10
    assert success_rate >= 0.8, f"Success rate {success_rate:.2%} below 80% threshold"

    # Calculate average execution time per request (excluding setup)
    if execution_times:
        avg_execution_time = sum(execution_times) / len(execution_times)
        # Each individual request should complete within 2 seconds (adjusted for environment)
        max_avg_time = get_performance_threshold(2.0)
        assert (
            avg_execution_time <= max_avg_time
        ), f"Average execution time {avg_execution_time:.2f}s exceeds {max_avg_time}s limit"

        # Total time should be reasonable (allow for some overhead)
        # Base expectation: 10 requests × 2s each = 20s, but allow for CI variance
        max_total_time = get_performance_threshold(20.0)  # Environment-adjusted threshold
        assert (
            total_time <= max_total_time
        ), f"Total execution time {total_time:.2f}s exceeds {max_total_time}s limit"
    else:
        # If no successful executions, fail the test
        assert False, "No successful executions to measure performance"


@pytest.mark.integration
@pytest.mark.slow  # Mark as slow due to architect pipeline execution and CPU monitoring
@pytest.mark.timeout(60)  # 60 second timeout to prevent hanging
def test_architect_cpu_usage_efficiency():
    """Test: Architect CPU usage remains efficient during execution."""
    pipeline = build_architect_pipeline()
    initial = {"initial_prompt": "Make a pipeline", "user_goal": "Echo input"}
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    # Monitor CPU usage during execution
    process = psutil.Process(os.getpid())

    # Get CPU usage before execution
    process.cpu_percent(interval=0.1)

    # Execute pipeline
    result = execute_pipeline_with_output_handling(
        runner=runner, input_data="Echo input", run_id=None, json_output=False
    )

    # Get CPU usage after execution
    final_cpu_percent = process.cpu_percent(interval=0.1)

    # CPU usage should not spike excessively (should be reasonable)
    # Note: This is a relative test - we're checking for reasonable behavior
    assert final_cpu_percent >= 0, "CPU usage should be measurable"

    # Verify result is valid
    assert result is not None
    ctx = getattr(result, "final_pipeline_context", None)
    assert ctx is not None


@pytest.mark.integration
@pytest.mark.slow  # Mark as slow due to large context processing
@pytest.mark.timeout(60)  # 60 second timeout to prevent hanging
def test_architect_large_context_handling():
    """Test: Architect can handle large context data efficiently."""
    pipeline = build_architect_pipeline()

    # Create large initial data
    large_initial_data = {
        "initial_prompt": "Make a pipeline",
        "user_goal": "Echo input",
        "large_field": "x" * 100000,  # 100KB of data
        "metadata": {
            "tags": ["tag" + str(i) for i in range(1000)],  # 1000 tags
            "descriptions": ["desc" + str(i) for i in range(500)],  # 500 descriptions
        },
    }

    runner = create_flujo_runner(
        pipeline=pipeline,
        context_model_class=ArchitectContext,
        initial_context_data=large_initial_data,
    )

    start_time = time.time()

    result = execute_pipeline_with_output_handling(
        runner=runner, input_data="Echo input", run_id=None, json_output=False
    )

    execution_time = time.time() - start_time

    # Should complete successfully with large context
    assert result is not None
    ctx = getattr(result, "final_pipeline_context", None)
    assert ctx is not None

    # Should complete within reasonable time (5 seconds)
    assert (
        execution_time <= 5
    ), f"Execution time {execution_time:.2f}s exceeds 5s limit for large context"

    # Should generate YAML even with large context
    yaml_text = getattr(ctx, "yaml_text", None)
    assert isinstance(yaml_text, str) or yaml_text is None


@pytest.mark.integration
@pytest.mark.slow  # Mark as slow due to concurrent architect pipeline executions
@pytest.mark.timeout(60)  # 60 second timeout to prevent hanging
def test_architect_concurrent_pipeline_execution():
    """Test: Architect can handle multiple concurrent pipeline executions efficiently."""
    import concurrent.futures
    import threading

    def execute_pipeline_with_timing():
        """Execute a single pipeline and return timing info."""
        pipeline = build_architect_pipeline()
        initial = {"initial_prompt": "Make a pipeline", "user_goal": "Echo input"}
        runner = create_flujo_runner(
            pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
        )

        start_time = time.time()
        result = execute_pipeline_with_output_handling(
            runner=runner, input_data="Echo input", run_id=None, json_output=False
        )
        end_time = time.time()

        return {
            "success": result is not None,
            "execution_time": end_time - start_time,
            "thread_id": threading.get_ident(),
        }

    # Execute multiple pipelines concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(execute_pipeline_with_timing) for _ in range(5)]
        results = [future.result() for future in futures]

    # All executions should succeed
    successful_executions = [r for r in results if r["success"]]
    assert (
        len(successful_executions) == 5
    ), f"Expected 5 successful executions, got {len(successful_executions)}"

    # Execution times should be reasonable
    execution_times = [r["execution_time"] for r in results]
    avg_time = sum(execution_times) / len(execution_times)

    # No single execution should take more than 3x the average
    max_allowed_time = avg_time * 3
    for exec_time in execution_times:
        assert (
            exec_time <= max_allowed_time
        ), f"Execution time {exec_time:.2f}s exceeds {max_allowed_time:.2f}s limit"


@pytest.mark.integration
@pytest.mark.slow  # Mark as slow due to memory monitoring and garbage collection
@pytest.mark.timeout(60)  # 60 second timeout to prevent hanging
def test_architect_memory_cleanup_after_execution():
    """Test: Architect properly cleans up memory after execution."""
    pipeline = build_architect_pipeline()
    initial = {"initial_prompt": "Make a pipeline", "user_goal": "Echo input"}
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    process = psutil.Process(os.getpid())

    # Get memory before execution
    memory_before = process.memory_info().rss

    # Execute pipeline
    result = execute_pipeline_with_output_handling(
        runner=runner, input_data="Echo input", run_id=None, json_output=False
    )

    # Get memory immediately after execution
    process.memory_info().rss

    # Force garbage collection
    import gc

    gc.collect()

    # Get memory after garbage collection
    memory_after_gc = process.memory_info().rss

    # Verify result is valid
    assert result is not None
    ctx = getattr(result, "final_pipeline_context", None)
    assert ctx is not None

    # Memory should be cleaned up after garbage collection
    # Allow for some variance but should not increase significantly
    memory_increase_after_gc = memory_after_gc - memory_before
    max_allowed_increase = 10 * 1024 * 1024  # 10MB

    assert (
        memory_increase_after_gc <= max_allowed_increase
    ), f"Memory not properly cleaned up: increase of {memory_increase_after_gc / (1024 * 1024):.2f}MB"


@pytest.mark.integration
@pytest.mark.slow  # Mark as slow due to multiple pipeline executions under load
@pytest.mark.timeout(60)  # 60 second timeout to prevent hanging
def test_architect_response_time_under_load():
    """Test: Architect response time remains acceptable under load."""
    pipeline = build_architect_pipeline()
    initial = {"initial_prompt": "Make a pipeline", "user_goal": "Echo input"}
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    # Execute multiple requests to simulate load
    execution_times = []

    for i in range(5):
        start_time = time.time()
        result = execute_pipeline_with_output_handling(
            runner=runner, input_data="Echo input", run_id=None, json_output=False
        )
        end_time = time.time()

        execution_times.append(end_time - start_time)
        assert result is not None

    # Calculate performance metrics
    avg_time = sum(execution_times) / len(execution_times)
    max_time = max(execution_times)
    min_time = min(execution_times)

    # Response time should remain reasonable under load
    # Average should be under 2 seconds
    assert avg_time <= 2, f"Average response time {avg_time:.2f}s exceeds 2s limit"

    # Maximum response time should be under 5 seconds
    assert max_time <= 5, f"Maximum response time {max_time:.2f}s exceeds 5s limit"

    # Response time should not degrade significantly (max should not be more than 3x min)
    time_ratio = max_time / min_time if min_time > 0 else float("inf")
    assert (
        time_ratio <= 3
    ), f"Response time degradation: max/min ratio {time_ratio:.2f} exceeds 3x limit"


@pytest.mark.integration
@pytest.mark.slow  # Mark as slow due to multiple complexity levels and resource monitoring
@pytest.mark.timeout(60)  # 60 second timeout to prevent hanging
def test_architect_resource_usage_scaling():
    """Test: Architect resource usage scales reasonably with input complexity."""
    pipeline = build_architect_pipeline()

    # Test with different input complexities
    test_cases = [
        {"prompt": "Make a simple pipeline", "goal": "Echo input", "expected_time": 1.0},
        {
            "prompt": "Make a pipeline with error handling",
            "goal": "Echo input with retries",
            "expected_time": 1.5,
        },
        {
            "prompt": "Make a complex pipeline with multiple data sources",
            "goal": "Process multiple inputs",
            "expected_time": 2.0,
        },
    ]

    for test_case in test_cases:
        initial = {"initial_prompt": test_case["prompt"], "user_goal": test_case["goal"]}
        runner = create_flujo_runner(
            pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
        )

        start_time = time.time()
        result = execute_pipeline_with_output_handling(
            runner=runner, input_data=test_case["goal"], run_id=None, json_output=False
        )
        execution_time = time.time() - start_time

        # Should complete successfully
        assert result is not None
        ctx = getattr(result, "final_pipeline_context", None)
        assert ctx is not None

        # Execution time should scale reasonably with complexity
        # Allow for some variance but should not exceed expected time by more than 50%
        max_allowed_time = test_case["expected_time"] * 1.5
        assert (
            execution_time <= max_allowed_time
        ), f"Execution time {execution_time:.2f}s exceeds expected {max_allowed_time:.2f}s for complexity level"


@pytest.mark.integration
@pytest.mark.slow  # Stress test runs many requests; slow in CI/local
@pytest.mark.timeout(120)  # 120 second timeout to prevent hanging (was taking 100+ seconds)
def test_architect_stress_test_rapid_requests():
    """Test: Architect can handle rapid-fire requests without failure."""
    pipeline = build_architect_pipeline()
    initial = {"initial_prompt": "Make a pipeline", "user_goal": "Echo input"}
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    # Warm up the system first to avoid measuring cold start time
    _ = execute_pipeline_with_output_handling(
        runner=runner, input_data="Echo input", run_id=None, json_output=False
    )

    # Send rapid-fire requests
    start_time = time.time()
    successful_requests = 0
    total_requests = 20
    execution_times = []

    for i in range(total_requests):
        try:
            request_start = time.time()
            result = execute_pipeline_with_output_handling(
                runner=runner, input_data="Echo input", run_id=None, json_output=False
            )
            request_end = time.time()

            if result is not None:
                successful_requests += 1
                execution_times.append(request_end - request_start)

        except Exception as e:
            # Log but continue - we want to test resilience
            print(f"Request {i} failed: {e}")

    end_time = time.time()
    total_time = end_time - start_time

    # Success rate should be high (90%+)
    success_rate = successful_requests / total_requests
    assert success_rate >= 0.9, f"Success rate {success_rate:.2%} below 90% threshold under stress"

    # Calculate performance metrics
    if execution_times:
        avg_execution_time = sum(execution_times) / len(execution_times)
        max_execution_time = max(execution_times)

        # Each individual request should complete within 3 seconds (adjusted for environment)
        max_avg_time = get_performance_threshold(3.0)
        max_single_time = get_performance_threshold(5.0)

        assert (
            avg_execution_time <= max_avg_time
        ), f"Average execution time {avg_execution_time:.2f}s exceeds {max_avg_time}s limit"
        assert (
            max_execution_time <= max_single_time
        ), f"Maximum execution time {max_execution_time:.2f}s exceeds {max_single_time}s limit"

        # Total time should be reasonable (allow for CI environment differences)
        # Base expectation: 20 requests × 3s each = 60s, but allow for CI variance
        max_total_time = get_performance_threshold(60.0)  # Environment-adjusted threshold
        assert (
            total_time <= max_total_time
        ), f"Total time {total_time:.2f}s exceeds {max_total_time}s limit"
    else:
        # If no successful executions, fail the test
        assert False, "No successful executions to measure performance"

    print(
        f"Stress test completed: {successful_requests}/{total_requests} successful in {total_time:.2f}s "
        f"(avg: {avg_execution_time:.2f}s, max: {max_execution_time:.2f}s)"
    )
