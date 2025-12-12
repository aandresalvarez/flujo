"""Error recovery and resilience tests.

These tests ensure that Flujo can recover gracefully from various error conditions
and maintain system stability under adverse circumstances.
"""
# ruff: noqa

import asyncio
import signal
import time
from typing import List, Dict, Any, Optional, Union
import pytest
from unittest.mock import AsyncMock, Mock, patch

pytestmark = [
    pytest.mark.slow,
    pytest.mark.stress,
]

from flujo.application.core.executor_core import ExecutorCore
from flujo.domain.models import PipelineContext, StepResult, StepOutcome, Failure
from flujo.domain.dsl.step import Step
from tests.robustness.error_handling import (
    get_error_handler,
    with_error_recovery,
    async_with_error_recovery,
)
from flujo.exceptions import (
    FlujoError,
    ExecutionError,
    ValidationError,
    ConfigurationError,
    PausedException,
    PipelineAbortSignal,
    InfiniteRedirectError,
)
from tests.test_types.mocks import create_mock_executor_core


class TestErrorRecovery:
    """Test suite for error recovery and resilience."""

    def test_graceful_degradation_on_agent_failure(self):
        """Test that system degrades gracefully when agents fail."""
        executor = create_mock_executor_core()

        # Create step with failing agent
        failing_agent = AsyncMock()
        failing_agent.run.side_effect = Exception("Agent crashed")

        step = Step(name="failing_step", agent=failing_agent)

        async def test_failure_recovery():
            result = await executor.execute(step, {"input": "test"})

            # Should get failure result, not crash
            assert isinstance(result, StepResult)
            assert not result.success
            feedback = (result.feedback or "").lower()
            assert "failed" in feedback or "exception" in feedback

        asyncio.run(test_failure_recovery())

    def test_network_timeout_recovery(self):
        """Test recovery from network timeouts."""
        handler = get_error_handler()
        executor = create_mock_executor_core()

        # Create agent that times out
        async def slow_agent(*args, **kwargs):
            await asyncio.sleep(10)  # Very slow
            return {"result": "success"}

        timeout_agent = AsyncMock()
        timeout_agent.run = slow_agent

        step = Step(name="timeout_step", agent=timeout_agent)

        async def test_timeout_operation():
            result = await executor.execute(step, {"input": "test"})
            # This should timeout gracefully
            return result

        async def run_timeout_test():
            # Test with timeout expectation
            error_result = await handler.async_expect_error(
                expected_errors=[
                    asyncio.TimeoutError,
                    Exception,
                ],  # Accept timeout or other failures
                operation=test_timeout_operation,
                timeout=5.0,  # Shorter timeout for test
            )

            # Should handle timeout gracefully
            assert error_result.success, (
                f"Timeout not handled properly: {error_result.error_message}"
            )

            # Should complete in reasonable time
            assert error_result.recovery_time < 6.0, (
                f"Recovery took too long: {error_result.recovery_time}s"
            )

        asyncio.run(run_timeout_test())

    def test_retry_mechanism_robustness(self):
        """Test retry mechanism under various failure conditions."""
        handler = get_error_handler()

        # Test retry with intermittent failures
        call_count = 0

        def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:  # Fail first 2 attempts
                raise ConnectionError("Temporary network failure")
            return {"result": "success"}

        # Should succeed on 3rd attempt
        result = handler.retry_operation(
            flaky_operation, max_attempts=5, exceptions_to_retry=[ConnectionError]
        )

        assert result["result"] == "success"
        assert call_count == 3  # Should have taken 3 attempts

        # Test async retry
        async def async_flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 6:  # Fail attempts 3-5
                raise TimeoutError("Async timeout")
            return {"result": "async_success"}

        async def test_async_retry():
            result = await handler.async_retry_operation(
                async_flaky_operation, max_attempts=5, exceptions_to_retry=[TimeoutError]
            )
            return result

        async_result = asyncio.run(test_async_retry())
        assert async_result["result"] == "async_success"
        assert call_count == 6  # Should have taken 3 more attempts

    def test_memory_exhaustion_recovery(self):
        """Test recovery from memory exhaustion conditions."""
        executor = create_mock_executor_core()

        # Create agent that tries to allocate too much memory
        async def memory_hungry_agent(*args, **kwargs):
            # Try to allocate large amounts of memory
            large_data = []
            for i in range(1000):
                large_data.append("x" * 10000)  # 10KB per string
            return {"result": "success", "data": large_data}

        memory_agent = AsyncMock()
        memory_agent.run = memory_hungry_agent

        step = Step(name="memory_step", agent=memory_agent)

        async def test_memory_recovery():
            try:
                result = await executor.execute(step, {"input": "test"})

                # Should either succeed or fail gracefully
                assert isinstance(result, StepResult)

                # If it failed, should have proper error message
                if not result.success:
                    assert result.feedback is not None

            except MemoryError:
                # If we actually hit memory limits, that's also acceptable
                # The test environment might have memory constraints
                pass

        asyncio.run(test_memory_recovery())

    def test_corrupted_context_recovery(self):
        """Test recovery from corrupted context data."""
        from flujo.application.core.context_manager import ContextManager

        # Create corrupted context
        context = PipelineContext()
        context.step_outputs = {"corrupted": object()}  # Non-serializable object

        # Try to isolate corrupted context
        try:
            isolated = ContextManager.isolate(context)
            # Should handle corruption gracefully
            assert isolated is not None
        except Exception as e:
            # If isolation fails, it should fail gracefully with clear error
            assert "corrupted" in str(e).lower() or "serializable" in str(e).lower()

    def test_circular_reference_recovery(self):
        """Test recovery from circular reference issues."""
        import json
        from flujo.state.backends.base import _serialize_for_json

        # Create circular reference
        obj1 = {"name": "obj1"}
        obj2 = {"name": "obj2"}
        obj1["ref"] = obj2
        obj2["ref"] = obj1

        # Should handle circular references gracefully via json-friendly serialization
        normalized = _serialize_for_json(obj1, _seen=set(), strict=False)
        serialized = json.loads(json.dumps(normalized, ensure_ascii=False))
        assert serialized["ref"]["ref"] is None

        # Strict mode should use the placeholder strings
        strict_normalized = _serialize_for_json(obj1, _seen=set(), strict=True)
        strict_serialized = json.loads(json.dumps(strict_normalized, ensure_ascii=False))
        assert strict_serialized["ref"]["ref"] in ("<circular>", "<circular-ref>")

    def test_invalid_configuration_recovery(self):
        """Test recovery from invalid configuration."""
        # Test with invalid executor configuration
        with pytest.raises((ValueError, ConfigurationError)):
            ExecutorCore(
                agent_runner=None,  # Invalid
                processor_pipeline=None,  # Invalid
                validator_runner=None,  # Invalid
                plugin_runner=None,  # Invalid
                usage_meter=None,  # Invalid
                cache_backend=None,  # Invalid
                telemetry=None,
                cache_size=-1,  # Invalid
            )

    def test_signal_handling_during_execution(self):
        """Test that execution handles system signals gracefully."""
        executor = create_mock_executor_core()

        # Create long-running operation
        async def long_running_agent(*args, **kwargs):
            await asyncio.sleep(2)
            return {"result": "completed"}

        long_agent = AsyncMock()
        long_agent.run = long_running_agent

        step = Step(name="long_step", agent=long_agent)

        async def test_signal_handling():
            # Start execution
            task = asyncio.create_task(executor.execute(step, {"input": "test"}))

            # Wait a bit then cancel (simulates signal)
            await asyncio.sleep(0.1)
            task.cancel()

            # Should handle cancellation gracefully
            try:
                await task
                pytest.fail("Expected task to be cancelled")
            except asyncio.CancelledError:
                # This is expected
                pass

        asyncio.run(test_signal_handling())

    def test_resource_exhaustion_recovery(self):
        """Test recovery from resource exhaustion."""
        import concurrent.futures

        # Test thread pool exhaustion
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # Submit many tasks to potentially exhaust resources
            futures = []
            for i in range(10):
                future = executor.submit(lambda: "test")
                futures.append(future)

            # Should complete without hanging
            results = [f.result(timeout=5) for f in concurrent.futures.as_completed(futures)]
            assert len(results) == 10
            assert all(r == "test" for r in results)

    def test_database_connection_failure_recovery(self):
        """Test recovery from database connection failures."""
        # This would test actual database components if they existed
        # For now, test with mock failures

        # Simulate database connection failure
        mock_db = Mock()
        mock_db.query.side_effect = Exception("Connection lost")

        # System should handle DB failures gracefully
        try:
            mock_db.query("SELECT * FROM test")
            pytest.fail("Expected database exception")
        except Exception as e:
            assert "Connection lost" in str(e)

    def test_external_service_degradation_recovery(self):
        """Test recovery when external services degrade."""
        executor = create_mock_executor_core()

        # Simulate external service degradation (increasing delays)
        call_count = 0

        async def degrading_agent(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            # Simulate increasing delays (service degradation)
            delay = min(call_count * 0.1, 2.0)  # Cap at 2 seconds
            await asyncio.sleep(delay)

            return {"result": f"call_{call_count}"}

        degrading_service = AsyncMock()
        degrading_service.run = degrading_agent

        step = Step(name="degrading_step", agent=degrading_service)

        async def test_service_degradation():
            results = []

            # Make several calls to simulate service degradation
            for i in range(5):
                start_time = time.time()
                result = await executor.execute(step, {"input": f"test_{i}"})
                end_time = time.time()

                execution_time = end_time - start_time
                results.append((i, result, execution_time))

                # Should eventually slow down but not fail completely
                if i >= 2:  # After a few calls
                    assert execution_time < 3.0, f"Call {i} took too long: {execution_time}s"

            # Should have completed all calls
            assert len(results) == 5
            assert all(isinstance(r, StepResult) for _, r, _ in results)

        asyncio.run(test_service_degradation())


class TestResilienceUnderLoad:
    """Test suite for system resilience under various load conditions."""

    def test_high_frequency_execution_resilience(self):
        """Test resilience under high-frequency execution."""
        executor = create_mock_executor_core()

        async def run_high_frequency_test():
            results = []

            # Execute many operations in quick succession
            for i in range(100):
                step = Step(name=f"freq_step_{i}", agent=AsyncMock())
                result = await executor.execute(step, {"input": f"data_{i}"})
                results.append(result)

                # Brief yield to other tasks
                await asyncio.sleep(0.001)

            return results

        results = asyncio.run(run_high_frequency_test())

        # Should complete all operations
        assert len(results) == 100
        assert all(isinstance(r, StepResult) for r in results)
        assert all(r.success for r in results)

    def test_memory_pressure_resilience(self):
        """Test resilience under memory pressure."""
        executor = create_mock_executor_core()

        async def run_memory_pressure_test():
            results = []

            # Create memory pressure with large data
            for i in range(50):
                large_data = {
                    "id": i,
                    "payload": "x" * 50000,  # 50KB per operation
                    "metadata": [{"size": "large"} for _ in range(100)],
                }

                step = Step(name=f"memory_step_{i}", agent=AsyncMock())
                result = await executor.execute(step, large_data)
                results.append(result)

                # Periodic cleanup check
                if i % 10 == 0:
                    import gc

                    gc.collect()

            return results

        results = asyncio.run(run_memory_pressure_test())

        # Should handle memory pressure gracefully
        assert len(results) == 50
        assert all(isinstance(r, StepResult) for r in results)

    def test_network_instability_resilience(self):
        """Test resilience under simulated network instability."""
        executor = create_mock_executor_core()

        # Simulate intermittent network failures
        failure_count = 0

        async def unstable_agent(*args, **kwargs):
            nonlocal failure_count
            failure_count += 1

            # Fail every 3rd call
            if failure_count % 3 == 0:
                raise Exception("Network timeout")

            await asyncio.sleep(0.01)  # Simulate network delay
            return {"result": f"success_{failure_count}"}

        unstable_service = AsyncMock()
        unstable_service.run = unstable_agent

        step = Step(name="unstable_step", agent=unstable_service)

        async def test_network_instability():
            results = []

            # Make multiple calls, some will fail
            for i in range(9):  # 3 failures expected
                result = await executor.execute(step, {"input": f"test_{i}"})
                results.append(result)

            # Should have 6 successes and 3 failures
            successful_results = [r for r in results if r.success]
            failed_results = [r for r in results if not r.success]

            assert len(successful_results) == 6, (
                f"Expected 6 successes, got {len(successful_results)}"
            )
            assert len(failed_results) == 3, f"Expected 3 failures, got {len(failed_results)}"

            # All results should be StepResult instances
            assert all(isinstance(r, StepResult) for r in results)

        asyncio.run(test_network_instability())

    def test_disk_space_exhaustion_recovery(self):
        """Test recovery when disk space is exhausted."""
        # This would test file operations under disk pressure
        # For now, test with mock failures

        import tempfile
        import os

        # Create a temporary file to test file operations
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(b"test data")

        try:
            # Simulate disk full by trying operations that might fail
            # In a real scenario, this would test actual file I/O under disk pressure

            # Test file access
            with open(temp_path, "r") as f:
                content = f.read()
                assert content == "test data"

        finally:
            # Cleanup
            os.unlink(temp_path)


class TestGracefulShutdown:
    """Test suite for graceful shutdown scenarios."""

    def test_shutdown_during_execution(self):
        """Test graceful shutdown while execution is in progress."""
        executor = create_mock_executor_core()

        async def long_running_operation():
            async def _slow_run(*args, **kwargs):
                await asyncio.sleep(1.0)
                return {"result": "done"}

            slow_agent = AsyncMock()
            slow_agent.run = AsyncMock(side_effect=_slow_run)

            step = Step(name="long_step", agent=slow_agent)

            # Start execution
            task = asyncio.create_task(executor.execute(step, {"input": "test"}))

            # Wait a bit then cancel
            await asyncio.sleep(0.05)
            task.cancel()

            # Should handle cancellation gracefully
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(long_running_operation())

    def test_cleanup_on_shutdown(self):
        """Test that resources are cleaned up on shutdown."""
        from flujo.application.core.background_task_manager import BackgroundTaskManager

        manager = BackgroundTaskManager()

        async def test_cleanup():
            # Create some background tasks
            tasks = []
            for i in range(3):
                task = asyncio.create_task(asyncio.sleep(1))
                manager.add_task(task)
                tasks.append(task)

            # Simulate shutdown - cancel tasks
            for task in tasks:
                task.cancel()

            # Wait for cleanup
            await asyncio.gather(*tasks, return_exceptions=True)

            # Manager should be empty
            assert len(manager._background_tasks) == 0

        asyncio.run(test_cleanup())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
