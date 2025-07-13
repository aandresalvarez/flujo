"""Tests for flujo.domain.streaming_protocol module."""

import pytest
from typing import AsyncIterator, Any

from flujo.domain.streaming_protocol import StreamingAgentProtocol


class MockStreamingAgent:
    """Mock agent that implements StreamingAgentProtocol."""

    async def stream(self, data: Any, **kwargs: Any) -> AsyncIterator[Any]:
        """Mock streaming implementation."""
        yield f"chunk1: {data}"
        yield f"chunk2: {data}"
        yield f"final: {data}"


class MockStreamingAgentWithKwargs:
    """Mock agent that uses kwargs in stream method."""

    async def stream(self, data: Any, **kwargs: Any) -> AsyncIterator[Any]:
        """Mock streaming implementation with kwargs."""
        prefix = kwargs.get("prefix", "default")
        yield f"{prefix}: {data}"


def test_streaming_protocol_import():
    """Test that StreamingAgentProtocol can be imported."""
    assert StreamingAgentProtocol is not None


def test_streaming_protocol_implementation():
    """Test that a class can implement StreamingAgentProtocol."""
    agent = MockStreamingAgent()

    # Verify it has the required method
    assert hasattr(agent, "stream")
    assert callable(agent.stream)


@pytest.mark.asyncio
async def test_streaming_agent_functionality():
    """Test that a streaming agent works as expected."""
    agent = MockStreamingAgent()

    chunks = []
    async for chunk in agent.stream("test_data"):
        chunks.append(chunk)

    assert len(chunks) == 3
    assert chunks[0] == "chunk1: test_data"
    assert chunks[1] == "chunk2: test_data"
    assert chunks[2] == "final: test_data"


@pytest.mark.asyncio
async def test_streaming_agent_with_kwargs():
    """Test that a streaming agent works with kwargs."""
    agent = MockStreamingAgentWithKwargs()

    chunks = []
    async for chunk in agent.stream("test_data", prefix="custom"):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0] == "custom: test_data"


@pytest.mark.asyncio
async def test_streaming_agent_default_kwargs():
    """Test that a streaming agent works with default kwargs."""
    agent = MockStreamingAgentWithKwargs()

    chunks = []
    async for chunk in agent.stream("test_data"):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0] == "default: test_data"
