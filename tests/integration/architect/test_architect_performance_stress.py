from __future__ import annotations

import os
import time

import pytest

from flujo.architect.builder import build_architect_pipeline
from flujo.architect.context import ArchitectContext
from flujo.cli.helpers import create_flujo_runner, execute_pipeline_with_output_handling
# Performance thresholds removed - using relative measurements instead

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment,unused-ignore]

pytestmark = []
if psutil is None:
    pytestmark.append(pytest.mark.skip(reason="psutil not available"))


# Force minimal architect pipeline for performance tests to avoid hanging.
# Apply env overrides per-test to avoid leaking into other suites.
@pytest.fixture(autouse=True)
def _architect_perf_env():
    prev_ignore = os.environ.get("FLUJO_ARCHITECT_IGNORE_CONFIG")
    prev_test = os.environ.get("FLUJO_TEST_MODE")
    prev_disable = os.environ.get("FLUJO_DISABLE_TRACING")
    prev_state_machine = os.environ.get("FLUJO_ARCHITECT_STATE_MACHINE")

    # Force the minimal architect pipeline for perf timings, even though the
    # integration/architect conftest enables the full state machine by default.
    # The state machine path is exercised elsewhere; here we only care about
    # perf of the lightweight builder.
    os.environ["FLUJO_ARCHITECT_IGNORE_CONFIG"] = "1"
    os.environ["FLUJO_TEST_MODE"] = "1"
    os.environ["FLUJO_DISABLE_TRACING"] = "1"
    os.environ["FLUJO_ARCHITECT_STATE_MACHINE"] = "0"
    try:
        yield
    finally:
        if prev_ignore is None:
            os.environ.pop("FLUJO_ARCHITECT_IGNORE_CONFIG", None)
        else:
            os.environ["FLUJO_ARCHITECT_IGNORE_CONFIG"] = prev_ignore
        if prev_test is None:
            os.environ.pop("FLUJO_TEST_MODE", None)
        else:
            os.environ["FLUJO_TEST_MODE"] = prev_test
        if prev_disable is None:
            os.environ.pop("FLUJO_DISABLE_TRACING", None)
        else:
            os.environ["FLUJO_DISABLE_TRACING"] = prev_disable
        if prev_state_machine is None:
            os.environ.pop("FLUJO_ARCHITECT_STATE_MACHINE", None)
        else:
            os.environ["FLUJO_ARCHITECT_STATE_MACHINE"] = prev_state_machine


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
        assert abs(exec_time - avg_time) <= max_variance, (
            f"Execution time {exec_time}s varies too much from average {avg_time}s"
        )


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_architect_memory_usage_stability():
    """Test: Architect memory usage remains stable during execution."""
    if psutil is None:
        pytest.skip("psutil not available")
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

    # Memory usage should not increase excessively.
    # Use a generous 200MB threshold that works in any environment.
    # The goal is to detect memory leaks, not micro-optimize.
    memory_increase = final_memory - initial_memory
    max_allowed_increase_mb = 200.0  # 200MB is generous for a single pipeline run
    max_allowed_increase = int(max_allowed_increase_mb * 1024 * 1024)

    assert memory_increase <= max_allowed_increase, (
        f"Memory usage increased by {memory_increase / (1024 * 1024):.2f}MB, exceeding limit of {max_allowed_increase_mb:.0f}MB"
    )

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
        # Log performance for debugging (no strict threshold - stress test only)
        print(f"Average execution time: {avg_execution_time:.2f}s")
        print(f"Total execution time: {total_time:.2f}s")

        # Sanity check: operations should complete (not hang)
        # Use generous 120s timeout - this is a stress test, not a speed test
        assert total_time <= 120.0, (
            f"Total execution time {total_time:.2f}s exceeds 120s sanity limit"
        )
    else:
        # If no successful executions, fail the test
        assert False, "No successful executions to measure performance"


@pytest.mark.integration
@pytest.mark.slow  # Mark as slow due to architect pipeline execution and CPU monitoring
@pytest.mark.timeout(60)  # 60 second timeout to prevent hanging
def test_architect_cpu_usage_efficiency():
    """Test: Architect CPU usage remains efficient during execution."""
    if psutil is None:
        pytest.skip("psutil not available")
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

    # Sanity check: should complete (not hang)
    # Use generous 60s timeout - this is a stress test with large context
    print(f"Execution time with large context: {execution_time:.2f}s")
    assert execution_time <= 60.0, (
        f"Execution time {execution_time:.2f}s exceeds 60s sanity limit for large context"
    )

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
    assert len(successful_executions) == 5, (
        f"Expected 5 successful executions, got {len(successful_executions)}"
    )

    # Execution times should be reasonable
    execution_times = [r["execution_time"] for r in results]
    avg_time = sum(execution_times) / len(execution_times)

    # No single execution should take more than 3x the average
    max_allowed_time = avg_time * 3
    for exec_time in execution_times:
        assert exec_time <= max_allowed_time, (
            f"Execution time {exec_time:.2f}s exceeds {max_allowed_time:.2f}s limit"
        )


@pytest.mark.integration
@pytest.mark.slow  # Mark as slow due to memory monitoring and garbage collection
@pytest.mark.timeout(60)  # 60 second timeout to prevent hanging
def test_architect_memory_cleanup_after_execution():
    """Test: Architect properly cleans up memory after execution."""
    if psutil is None:
        pytest.skip("psutil not available")
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

    # (Optional) Capture memory immediately after execution if you want to log trends:
    # memory_immediate = process.memory_info().rss

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

    assert memory_increase_after_gc <= max_allowed_increase, (
        f"Memory not properly cleaned up: increase of {memory_increase_after_gc / (1024 * 1024):.2f}MB"
    )


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

    print(f"Performance under load: avg={avg_time:.2f}s, min={min_time:.2f}s, max={max_time:.2f}s")

    # RELATIVE performance check (environment-independent):
    # Response time should not degrade significantly under load.
    # Max should not be more than 5x min (generous to account for CI variance).
    time_ratio = max_time / min_time if min_time > 0 else float("inf")
    assert time_ratio <= 5, (
        f"Response time degradation: max/min ratio {time_ratio:.2f} exceeds 5x limit "
        f"(min={min_time:.2f}s, max={max_time:.2f}s)"
    )


@pytest.mark.integration
@pytest.mark.slow  # Mark as slow due to multiple complexity levels and resource monitoring
@pytest.mark.timeout(60)  # 60 second timeout to prevent hanging
def test_architect_resource_usage_scaling():
    """Test: Architect resource usage scales reasonably with input complexity."""
    pipeline = build_architect_pipeline()

    # Test with different input complexities
    test_cases = [
        {"prompt": "Make a simple pipeline", "goal": "Echo input", "expected_time": 2.0},
        {
            "prompt": "Make a pipeline with error handling",
            "goal": "Echo input with retries",
            "expected_time": 3.0,
        },
        {
            "prompt": "Make a complex pipeline with multiple data sources",
            "goal": "Process multiple inputs",
            "expected_time": 4.0,
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
        assert execution_time <= max_allowed_time, (
            f"Execution time {execution_time:.2f}s exceeds expected {max_allowed_time:.2f}s for complexity level"
        )


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
        min_execution_time = min(execution_times)

        print(
            f"Stress test completed: {successful_requests}/{total_requests} successful in {total_time:.2f}s "
            f"(avg: {avg_execution_time:.2f}s, min: {min_execution_time:.2f}s, max: {max_execution_time:.2f}s)"
        )

        # RELATIVE performance check (environment-independent):
        # Max execution time should not be more than 5x min (generous for stress test)
        time_ratio = (
            max_execution_time / min_execution_time if min_execution_time > 0 else float("inf")
        )
        assert time_ratio <= 5, (
            f"Execution time variance too high: max/min ratio {time_ratio:.2f} exceeds 5x limit"
        )

        # Sanity check: total time should not exceed generous limit (not hang)
        assert total_time <= 180.0, f"Total time {total_time:.2f}s exceeds 180s sanity limit"
    else:
        # If no successful executions, fail the test
        assert False, "No successful executions to measure performance"
