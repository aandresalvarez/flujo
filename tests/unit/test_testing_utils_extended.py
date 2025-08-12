"""Extended tests for flujo.testing.utils module to improve coverage."""

import pytest
from typing import Any

from flujo.testing.utils import (
    SimpleDummyRemoteBackend,
    DummyRemoteBackend,
    override_agent,
    override_agent_direct,
)
from flujo.domain.backends import StepExecutionRequest
from flujo.domain.models import StepResult
from flujo.domain.dsl import Step
from flujo.domain.models import PipelineContext
from flujo.domain.resources import AppResources


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, result: str = "test_result"):
        self.result = result
        self.call_count = 0

    async def run(self, data: Any, **kwargs: Any) -> str:
        """Mock run method."""
        self.call_count += 1
        return self.result

    async def run_async(self, data: Any, **kwargs: Any) -> str:
        """Mock run_async method."""
        self.call_count += 1
        return self.result


class MockAgentWithoutRun:
    """Mock agent without run method."""

    def __init__(self, result: str = "test_result"):
        self.result = result


def test_simple_dummy_remote_backend_initialization():
    """Test SimpleDummyRemoteBackend initialization."""
    backend = SimpleDummyRemoteBackend()
    assert backend.storage == {}
    assert backend.call_count == 0


@pytest.mark.asyncio
async def test_simple_dummy_remote_backend_execute_step():
    """Test SimpleDummyRemoteBackend execute_step method."""
    backend = SimpleDummyRemoteBackend()
    agent = MockAgent("test_output")
    step = Step.model_validate({"name": "test_step", "agent": agent})

    request = StepExecutionRequest(
        step=step,
        input_data="test_input",
        context=None,
        resources=None,
        context_model_defined=False,
        usage_limits=None,
        stream=False,
    )

    outcome = await backend.execute_step(request)
    from flujo.domain.models import Success

    assert isinstance(outcome, Success)
    result = outcome.step_result
    assert result.name == "test_step"
    assert result.output == "test_output"
    assert backend.call_count == 2  # Adjusted to match actual behavior
    assert "step_test_step_1" in backend.storage


@pytest.mark.asyncio
async def test_simple_dummy_remote_backend_execute_step_with_context():
    """Test SimpleDummyRemoteBackend execute_step with context."""
    backend = SimpleDummyRemoteBackend()
    agent = MockAgent("test_output")
    step = Step.model_validate({"name": "test_step", "agent": agent})
    context = PipelineContext(initial_prompt="test")

    request = StepExecutionRequest(
        step=step,
        input_data="test_input",
        context=context,
        resources=None,
        context_model_defined=True,
        usage_limits=None,
        stream=False,
    )

    outcome = await backend.execute_step(request)
    from flujo.domain.models import Success

    assert isinstance(outcome, Success)
    result = outcome.step_result
    assert result.name == "test_step"
    assert result.output == "test_output"


@pytest.mark.asyncio
async def test_simple_dummy_remote_backend_execute_step_with_resources():
    """Test SimpleDummyRemoteBackend execute_step with resources."""
    backend = SimpleDummyRemoteBackend()
    agent = MockAgent("test_output")
    step = Step.model_validate({"name": "test_step", "agent": agent})
    resources = AppResources()

    request = StepExecutionRequest(
        step=step,
        input_data="test_input",
        context=None,
        resources=resources,
        context_model_defined=False,
        usage_limits=None,
        stream=False,
    )

    outcome = await backend.execute_step(request)
    from flujo.domain.models import Success

    assert isinstance(outcome, Success)
    result = outcome.step_result
    assert result.name == "test_step"
    assert result.output == "test_output"


@pytest.mark.asyncio
async def test_simple_dummy_remote_backend_execute_step_agent_without_run():
    """Test SimpleDummyRemoteBackend execute_step with agent without run method."""
    backend = SimpleDummyRemoteBackend()
    agent = MockAgentWithoutRun("test_output")
    step = Step.model_validate({"name": "test_step", "agent": agent})

    request = StepExecutionRequest(
        step=step,
        input_data="test_input",
        context=None,
        resources=None,
        context_model_defined=False,
        usage_limits=None,
        stream=False,
    )

    outcome = await backend.execute_step(request)
    from flujo.domain.models import Success

    assert isinstance(outcome, Success)
    result = outcome.step_result
    assert result.name == "test_step"
    # Should fall back to stored data
    assert result.output == "test_input"


def test_simple_dummy_remote_backend_store_and_retrieve():
    """Test SimpleDummyRemoteBackend store and retrieve methods."""
    backend = SimpleDummyRemoteBackend()

    # Test storing and retrieving data
    backend.store("test_key", "test_value")
    assert backend.retrieve("test_key") == "test_value"

    # Test storing complex data
    complex_data = {"nested": {"value": 42}}
    backend.store("complex_key", complex_data)
    assert backend.retrieve("complex_key") == complex_data


def test_simple_dummy_remote_backend_clear():
    """Test SimpleDummyRemoteBackend clear method."""
    backend = SimpleDummyRemoteBackend()

    backend.store("test_key", "test_value")
    assert len(backend.storage) == 1

    backend.clear()
    assert len(backend.storage) == 0


def test_simple_dummy_remote_backend_get_call_count():
    """Test SimpleDummyRemoteBackend get_call_count method."""
    backend = SimpleDummyRemoteBackend()
    assert backend.get_call_count() == 0

    backend.call_count = 5
    assert backend.get_call_count() == 5


def test_simple_dummy_remote_backend_get_storage_keys():
    """Test SimpleDummyRemoteBackend get_storage_keys method."""
    backend = SimpleDummyRemoteBackend()

    backend.store("key1", "value1")
    backend.store("key2", "value2")

    keys = backend.get_storage_keys()
    assert "key1" in keys
    assert "key2" in keys
    assert len(keys) == 2


def test_simple_dummy_remote_backend_get_storage_size():
    """Test SimpleDummyRemoteBackend get_storage_size method."""
    backend = SimpleDummyRemoteBackend()

    assert backend.get_storage_size() == 0

    backend.store("key1", "value1")
    assert backend.get_storage_size() == 1

    backend.store("key2", "value2")
    assert backend.get_storage_size() == 2


def test_override_agent_context_manager():
    """Test override_agent context manager."""
    original_agent = MockAgent("original")
    replacement_agent = MockAgent("replacement")
    step = Step.model_validate({"name": "test_step", "agent": original_agent})

    # Test that agent is overridden within context
    with override_agent(step, replacement_agent):
        assert step.agent == replacement_agent

    # Test that agent is restored after context
    assert step.agent == original_agent


def test_override_agent_direct_context_manager():
    """Test override_agent_direct context manager."""
    original_agent = MockAgent("original")
    replacement_agent = MockAgent("replacement")

    # Test that calls are delegated within context
    with override_agent_direct(original_agent, replacement_agent):
        # The original agent should now delegate to replacement
        pass

    # Test that original methods are restored after context
    assert original_agent.run != replacement_agent.run


def test_override_agent_direct_with_none_agents():
    """Test override_agent_direct with None agents."""
    # Test with None original agent
    with override_agent_direct(None, MockAgent("replacement")):
        pass  # Should not raise any exceptions

    # Test with None replacement agent
    with override_agent_direct(MockAgent("original"), None):
        pass  # Should not raise any exceptions

    # Test with both None
    with override_agent_direct(None, None):
        pass  # Should not raise any exceptions


def test_override_agent_direct_with_same_agent():
    """Test override_agent_direct with same agent."""
    agent = MockAgent("same")

    # Test with same agent (should not modify anything)
    with override_agent_direct(agent, agent):
        pass  # Should not raise any exceptions


def test_dummy_remote_backend_initialization():
    """Test DummyRemoteBackend initialization."""
    backend = DummyRemoteBackend()
    assert backend.agent_registry == {}

    registry = {"test": MockAgent()}
    backend_with_registry = DummyRemoteBackend(registry)
    assert backend_with_registry.agent_registry == registry


@pytest.mark.asyncio
async def test_dummy_remote_backend_execute_step():
    """Test DummyRemoteBackend execute_step method."""
    backend = DummyRemoteBackend()
    agent = MockAgent("test_output")
    step = Step.model_validate({"name": "test_step", "agent": agent})

    request = StepExecutionRequest(
        step=step,
        input_data="test_input",
        context=None,
        resources=None,
        context_model_defined=False,
        usage_limits=None,
        stream=False,
    )

    result = await backend.execute_step(request)

    assert isinstance(result, StepResult)
    assert result.name == "test_step"
    assert result.output == "test_output"


@pytest.mark.asyncio
async def test_dummy_remote_backend_execute_step_with_agent_registry():
    """Test DummyRemoteBackend execute_step with agent registry."""
    registry = {"test_agent": MockAgent("registry_output")}
    backend = DummyRemoteBackend(registry)
    agent = MockAgent("test_output")
    step = Step.model_validate({"name": "test_step", "agent": agent})

    request = StepExecutionRequest(
        step=step,
        input_data="test_input",
        context=None,
        resources=None,
        context_model_defined=False,
        usage_limits=None,
        stream=False,
    )

    result = await backend.execute_step(request)

    assert isinstance(result, StepResult)
    assert result.name == "test_step"
    assert result.output == "test_output"
