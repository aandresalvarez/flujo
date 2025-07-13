"""Tests for the override_agent context manager."""

from flujo.domain import Step
from flujo.testing import override_agent, StubAgent


class MockAgent:
    """Mock agent for override tests."""

    def __init__(self, name: str):
        self.name = name
        self.call_count = 0

    async def run(self, data: str, **kwargs) -> str:
        self.call_count += 1
        return f"{self.name}: {data}"


def test_override_agent_basic_functionality():
    """Test basic agent override functionality."""
    original_agent = MockAgent("original")
    replacement_agent = MockAgent("replacement")

    step = Step.model_validate({"name": "test_step", "agent": original_agent})

    # Verify original agent is set
    assert step.agent is original_agent

    # Use context manager to override
    with override_agent(step, replacement_agent):
        assert step.agent is replacement_agent

    # Verify original agent is restored
    assert step.agent is original_agent


def test_override_agent_with_exception():
    """Test that agent is restored even when an exception occurs."""
    original_agent = MockAgent("original")
    replacement_agent = MockAgent("replacement")

    step = Step.model_validate({"name": "test_step", "agent": original_agent})

    # Verify original agent is set
    assert step.agent is original_agent

    # Use context manager and raise an exception
    try:
        with override_agent(step, replacement_agent):
            assert step.agent is replacement_agent
            raise RuntimeError("Test exception")
    except RuntimeError:
        pass

    # Verify original agent is still restored
    assert step.agent is original_agent


def test_override_agent_none_agent():
    """Test overriding with None agent."""
    original_agent = MockAgent("original")

    step = Step.model_validate({"name": "test_step", "agent": original_agent})

    # Use context manager to override with None
    with override_agent(step, None):
        assert step.agent is None

    # Verify original agent is restored
    assert step.agent is original_agent


def test_override_agent_multiple_steps():
    """Test overriding agents in multiple steps."""
    agent1 = MockAgent("agent1")
    agent2 = MockAgent("agent2")
    replacement1 = MockAgent("replacement1")
    replacement2 = MockAgent("replacement2")

    step1 = Step.model_validate({"name": "step1", "agent": agent1})
    step2 = Step.model_validate({"name": "step2", "agent": agent2})

    # Override both steps
    with override_agent(step1, replacement1):
        with override_agent(step2, replacement2):
            assert step1.agent is replacement1
            assert step2.agent is replacement2

    # Verify both are restored
    assert step1.agent is agent1
    assert step2.agent is agent2


def test_override_agent_integration_with_arun():
    """Test that the overridden agent is actually used when calling arun."""
    original_agent = MockAgent("original")
    replacement_agent = MockAgent("replacement")

    step = Step.model_validate({"name": "test_step", "agent": original_agent})

    # Test with original agent
    import asyncio

    result = asyncio.run(step.arun("test_data"))
    assert result == "original: test_data"
    assert original_agent.call_count == 1
    assert replacement_agent.call_count == 0

    # Test with overridden agent
    with override_agent(step, replacement_agent):
        result = asyncio.run(step.arun("test_data"))
        assert result == "replacement: test_data"
        assert original_agent.call_count == 1  # Unchanged
        assert replacement_agent.call_count == 1

    # Test that original agent is restored
    result = asyncio.run(step.arun("test_data"))
    assert result == "original: test_data"
    assert original_agent.call_count == 2
    assert replacement_agent.call_count == 1  # Unchanged


def test_override_agent_with_stub_agent():
    """Test overriding with StubAgent for testing scenarios."""
    original_agent = MockAgent("original")
    stub_agent = StubAgent(["stub_output_1", "stub_output_2"])

    step = Step.model_validate({"name": "test_step", "agent": original_agent})

    with override_agent(step, stub_agent):
        import asyncio

        result = asyncio.run(step.arun("test_data"))
        assert result == "stub_output_1"
        assert stub_agent.call_count == 1
        assert stub_agent.inputs == ["test_data"]

    # Verify original agent is restored
    assert step.agent is original_agent
