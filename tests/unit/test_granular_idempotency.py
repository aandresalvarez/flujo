"""Unit tests for idempotency enforcement in GranularAgentStepExecutor."""

import pytest
from unittest.mock import MagicMock
from flujo.application.core.policies.granular_policy import GranularAgentStepExecutor
from flujo.exceptions import ConfigurationError


class MockTool:
    def __init__(self, name, func, requires_key=True):
        self.__name__ = name
        self.function = func
        self.requires_idempotency_key = requires_key


@pytest.mark.asyncio
async def test_idempotency_wrapping_fails_if_key_missing() -> None:
    """Verify that tool calls fail if idempotency_key is missing when required."""
    executor = GranularAgentStepExecutor()

    # Mock a tool function
    async def my_tool(payload: dict, **kwargs: object) -> str:
        return "success"

    # Mock an agent with the tool in a dict (pydantic-ai style)
    mock_agent = MagicMock()
    mock_agent.tools = {"my_tool": MockTool("my_tool", my_tool, requires_key=True)}

    wrapped_agent = executor._enforce_idempotency_on_agent(mock_agent, "test-key")

    # The wrapped tool should be in wrapped_agent.tools["my_tool"].function
    wrapped_tool_func = wrapped_agent.tools["my_tool"].function

    # Calling without idempotency_key should raise ConfigurationError
    with pytest.raises(ConfigurationError) as exc_info:
        await wrapped_tool_func({"data": "foo"})

    assert "requires 'idempotency_key'" in str(exc_info.value)


@pytest.mark.asyncio
async def test_idempotency_wrapping_succeeds_with_correct_key() -> None:
    """Verify that tool calls succeed if correct idempotency_key is provided."""
    executor = GranularAgentStepExecutor()

    # Mock a tool function
    async def my_tool(payload: dict, **kwargs: object) -> str:
        return "success"

    # Mock an agent
    mock_agent = MagicMock()
    mock_agent.tools = {"my_tool": MockTool("my_tool", my_tool, requires_key=True)}

    key = "correct-key"
    wrapped_agent = executor._enforce_idempotency_on_agent(mock_agent, key)
    wrapped_tool_func = wrapped_agent.tools["my_tool"].function

    # Calling with correct key in payload should succeed
    result = await wrapped_tool_func({"data": "foo", "idempotency_key": key})
    assert result == "success"

    # Calling with correct key in kwargs should succeed
    result = await wrapped_tool_func({"data": "foo"}, idempotency_key=key)
    assert result == "success"


@pytest.mark.asyncio
async def test_idempotency_wrapping_fails_with_wrong_key() -> None:
    """Verify that tool calls fail if wrong idempotency_key is provided."""
    executor = GranularAgentStepExecutor()

    # Mock a tool function
    async def my_tool(payload: dict, **kwargs: object) -> str:
        return "success"

    # Mock an agent
    mock_agent = MagicMock()
    mock_agent.tools = {"my_tool": MockTool("my_tool", my_tool, requires_key=True)}

    expected_key = "expected-key"
    wrapped_agent = executor._enforce_idempotency_on_agent(mock_agent, expected_key)
    wrapped_tool_func = wrapped_agent.tools["my_tool"].function

    # Calling with wrong key in payload should raise ConfigurationError
    with pytest.raises(ConfigurationError) as exc_info:
        await wrapped_tool_func({"data": "foo", "idempotency_key": "wrong-key"})

    assert "Idempotency key mismatch" in str(exc_info.value)


@pytest.mark.asyncio
async def test_idempotency_wrapping_skips_if_not_required() -> None:
    """Verify that tool calls succeed if idempotency_key is missing but not required."""
    executor = GranularAgentStepExecutor()

    # Mock a tool function
    async def my_tool(payload: dict, **kwargs: object) -> str:
        return "success"

    # Mock an agent with requires_key=False
    mock_agent = MagicMock()
    mock_agent.tools = {"my_tool": MockTool("my_tool", my_tool, requires_key=False)}

    wrapped_agent = executor._enforce_idempotency_on_agent(mock_agent, "some-key")
    wrapped_tool_func = wrapped_agent.tools["my_tool"].function

    # Calling without key should succeed
    result = await wrapped_tool_func({"data": "foo"})
    assert result == "success"
