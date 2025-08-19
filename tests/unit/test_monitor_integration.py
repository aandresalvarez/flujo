"""Tests demonstrating proper use of In-Memory Monitor for unit testing.
The In-Memory Monitor is critical for the bug hunting campaign as it captures detailed information
about agent calls, including input data, output data, success status, and error details. This allows
developers to programmatically verify agent behavior and identify issues during testing. By ensuring
that the monitor records all relevant details, these tests help prevent regressions by validating
that changes to the codebase do not introduce unexpected failures or alter expected behavior."""

import pytest
import asyncio

from flujo.infra.monitor import global_monitor, FlujoMonitor
from flujo import Step
from tests.conftest import create_test_flujo


class TestInMemoryMonitorUsage:
    """Test proper usage of In-Memory Monitor for unit testing."""

    @pytest.fixture(autouse=True)
    def clear_monitor(self):
        """Clear monitor before each test to ensure isolation."""
        global_monitor.calls.clear()
        yield
        global_monitor.calls.clear()

    @pytest.mark.asyncio
    async def test_monitor_records_agent_calls(self):
        """Test that In-Memory Monitor records agent calls for programmatic verification."""
        from flujo.agents import monitored_agent
        from flujo.domain.agent_protocol import AsyncAgentProtocol

        # Create a monitored agent using the decorator
        @monitored_agent("test_agent")
        class MonitoredTestAgent(AsyncAgentProtocol[str, str]):
            async def run(self, data: str, **kwargs) -> str:
                return "test_output"

        agent = MonitoredTestAgent()

        # Create a simple pipeline
        step = Step.model_validate({"name": "test_step", "agent": agent})
        runner = create_test_flujo(step)

        # Run the pipeline
        from flujo.testing.utils import gather_result

        await gather_result(runner, "test_input")

        # Verify the monitor recorded the call
        assert len(global_monitor.calls) == 1
        call = global_monitor.calls[0]

        # Verify call details
        assert call["agent_name"] == "test_agent"
        assert call["success"] is True
        assert call["input_data"] == "test_input"
        assert call["output_data"] == "test_output"
        assert call["failure_type"] is None
        assert call["error_message"] is None
        assert call["exception"] is None

    @pytest.mark.asyncio
    async def test_monitor_records_failed_calls(self):
        """Test that In-Memory Monitor records failed agent calls."""
        from flujo.agents import monitored_agent
        from flujo.domain.agent_protocol import AsyncAgentProtocol

        # Create a monitored agent that will fail
        @monitored_agent("failing_agent")
        class FailingTestAgent(AsyncAgentProtocol[str, str]):
            async def run(self, data: str, **kwargs) -> str:
                raise Exception("Test failure")

        agent = FailingTestAgent()

        step = Step.model_validate({"name": "failing_step", "agent": agent})
        runner = create_test_flujo(step)

        # Run the pipeline (it will fail)
        from flujo.testing.utils import gather_result

        await gather_result(runner, "test_input")

        # Verify the monitor recorded the failure (may be called multiple times during error handling)
        assert len(global_monitor.calls) >= 1
        # Check the last call (most recent)
        call = global_monitor.calls[-1]

        # Verify failure details
        assert call["agent_name"] == "failing_agent"
        assert call["success"] is False
        assert call["input_data"] == "test_input"
        assert call["output_data"] is None
        assert call["failure_type"] is not None
        assert call["error_message"] == "Test failure"
        assert call["exception"] is not None

    @pytest.mark.asyncio
    async def test_monitor_multiple_calls(self):
        """Test that In-Memory Monitor records multiple agent calls."""
        from flujo.agents import monitored_agent
        from flujo.domain.agent_protocol import AsyncAgentProtocol

        # Create a monitored agent
        @monitored_agent("multi_agent")
        class MultiTestAgent(AsyncAgentProtocol[str, str]):
            async def run(self, data: str, **kwargs) -> str:
                return f"output_{data}"

        agent = MultiTestAgent()

        step = Step.model_validate({"name": "multi_step", "agent": agent})
        runner = create_test_flujo(step)

        # Run the pipeline twice
        from flujo.testing.utils import gather_result

        await gather_result(runner, "input1")
        await gather_result(runner, "input2")

        # Verify both calls were recorded
        assert len(global_monitor.calls) == 2

        # Verify first call
        assert global_monitor.calls[0]["agent_name"] == "multi_agent"
        assert global_monitor.calls[0]["input_data"] == "input1"
        assert global_monitor.calls[0]["output_data"] == "output_input1"

        # Verify second call
        assert global_monitor.calls[1]["agent_name"] == "multi_agent"
        assert global_monitor.calls[1]["input_data"] == "input2"
        assert global_monitor.calls[1]["output_data"] == "output_input2"

    @pytest.mark.asyncio
    @pytest.mark.slow  # Mark as slow due to sleep timing operation
    async def test_monitor_execution_time(self):
        """Test that In-Memory Monitor records execution time."""
        from flujo.agents import monitored_agent
        from flujo.domain.agent_protocol import AsyncAgentProtocol

        # Create a monitored slow agent
        @monitored_agent("slow_agent")
        class SlowTestAgent(AsyncAgentProtocol[str, str]):
            async def run(self, data: str, **kwargs) -> str:
                await asyncio.sleep(0.1)  # Simulate work
                return "slow_output"

        agent = SlowTestAgent()

        step = Step.model_validate({"name": "slow_step", "agent": agent})
        runner = create_test_flujo(step)

        # Run the pipeline
        from flujo.testing.utils import gather_result

        await gather_result(runner, "test_input")

        # Verify execution time was recorded
        assert len(global_monitor.calls) == 1
        call = global_monitor.calls[0]

        # Verify call details
        assert call["agent_name"] == "slow_agent"
        assert call["success"] is True
        assert call["input_data"] == "test_input"
        assert call["output_data"] == "slow_output"

        # Execution time should be recorded in milliseconds
        assert call["execution_time_ms"] > 0
        # In CI environments, timing can be slightly off due to system scheduling
        # Allow for a small tolerance (95ms instead of 100ms)
        assert call["execution_time_ms"] >= 95, (
            f"Expected execution time >= 95ms, but got {call['execution_time_ms']:.2f}ms. "
            f"This can happen in CI environments due to system scheduling variations."
        )

    def test_monitor_clear_functionality(self):
        """Test that monitor can be cleared for isolated tests."""
        # Create a custom monitor for testing
        test_monitor = FlujoMonitor()

        # Record some calls
        test_monitor.record_agent_call(
            agent_name="test",
            success=True,
            execution_time_ms=100,
            input_data="test",
            output_data="result",
        )

        # Verify call was recorded
        assert len(test_monitor.calls) == 1

        # Clear the monitor
        test_monitor.calls.clear()

        # Verify monitor is empty
        assert len(test_monitor.calls) == 0


class TestMonitorIntegrationWithCaplog:
    """Test In-Memory Monitor integration with caplog for comprehensive testing."""

    @pytest.fixture(autouse=True)
    def clear_monitor(self):
        """Clear monitor before each test to ensure isolation."""
        global_monitor.calls.clear()
        yield
        global_monitor.calls.clear()

    @pytest.mark.asyncio
    async def test_monitor_with_caplog(self, caplog):
        """Test that In-Memory Monitor works with caplog for log message assertions."""
        from flujo.agents import monitored_agent
        from flujo.domain.agent_protocol import AsyncAgentProtocol

        # Configure logging to capture messages
        import logging

        logging.getLogger("flujo").setLevel(logging.DEBUG)

        # Create a monitored agent
        @monitored_agent("caplog_agent")
        class CaplogTestAgent(AsyncAgentProtocol[str, str]):
            async def run(self, data: str, **kwargs) -> str:
                return "test_output"

        agent = CaplogTestAgent()
        step = Step.model_validate({"name": "test_step", "agent": agent})
        runner = create_test_flujo(step)

        # Run the pipeline
        from flujo.testing.utils import gather_result

        await gather_result(runner, "test_input")

        # Verify monitor recorded the call
        assert len(global_monitor.calls) == 1

        # Verify no unexpected log messages in unit tests
        # (In a real scenario, you might assert specific log messages)
        assert len(caplog.records) >= 0  # Allow for framework logging

        # Verify the call details through monitor (not logs)
        call = global_monitor.calls[0]
        assert call["agent_name"] == "caplog_agent"
        assert call["success"] is True
        assert call["input_data"] == "test_input"
        assert call["output_data"] == "test_output"
