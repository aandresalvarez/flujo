"""
Test for the critical streaming bytes corruption bug and type-safe protocols.

This test demonstrates the bug where streaming bytes payloads are corrupted
during reassembly in the UltraStepExecutor, and validates the new type-safe
streaming protocols.
"""

import pytest
from unittest.mock import MagicMock
from typing import AsyncIterator, Any, Union

from flujo.application.core.executor_core import ExecutorCore as UltraStepExecutor
from flujo.domain.dsl.step import Step
from flujo.domain.models import StepResult
from flujo.domain.resources import AppResources
from flujo.domain.streaming_protocol import (
    TextOnlyStreamingAgentProtocol,
    BinaryOnlyStreamingAgentProtocol,
    StreamingAgentProtocol,
)
from tests.test_types.fixtures import execute_simple_step


class StubTextStreamingAgent(TextOnlyStreamingAgentProtocol):
    """A stub agent that explicitly streams text data."""

    def __init__(self, chunks: list[str]):
        self.chunks = chunks

    async def stream(self, data: Any, **kwargs) -> AsyncIterator[str]:
        """Stream the configured text chunks."""
        for chunk in self.chunks:
            yield chunk


class StubBinaryStreamingAgent(BinaryOnlyStreamingAgentProtocol):
    """A stub agent that explicitly streams binary data."""

    def __init__(self, chunks: list[bytes]):
        self.chunks = chunks

    async def stream(self, data: Any, **kwargs) -> AsyncIterator[bytes]:
        """Stream the configured binary chunks."""
        for chunk in self.chunks:
            yield chunk


class StubLegacyStreamingAgent(StreamingAgentProtocol):
    """A stub agent using the legacy protocol for backward compatibility."""

    def __init__(self, chunks: list[Union[str, bytes]]):
        self.chunks = chunks

    async def stream(self, data: Any, **kwargs) -> AsyncIterator[Union[str, bytes]]:
        """Stream the configured chunks."""
        for chunk in self.chunks:
            yield chunk


class TestStreamingBytesBug:
    """Test suite for the streaming bytes corruption bug and type-safe protocols."""

    @pytest.fixture
    def executor(self):
        """Create a fresh UltraStepExecutor instance."""
        return UltraStepExecutor()

    @pytest.fixture
    def mock_step(self):
        """Create a mock step for testing."""
        from flujo.testing.utils import StubAgent

        # Create a real Step instance instead of a mock
        step = Step(
            name="test_step",
            agent=StubAgent(["test output"]),  # Default agent, will be overridden in tests
            persist_feedback_to_context=None,  # Explicitly set to None
        )
        return step

    @pytest.fixture
    def mock_resources(self):
        """Create mock resources."""
        return MagicMock(spec=AppResources)

    @pytest.mark.asyncio
    async def test_text_streaming_agent_protocol(self, executor, mock_step, mock_resources):
        """Test that TextStreamingAgentProtocol works correctly."""
        # Arrange
        string_chunks = ["hello", " ", "world", "!"]
        agent = StubTextStreamingAgent(string_chunks)
        mock_step.agent = agent

        # Act
        result = await execute_simple_step(
            executor,
            step=mock_step,
            data="test input",
            context=None,
            resources=mock_resources,
            stream=True,
        )

        # Assert
        assert isinstance(result, StepResult)
        assert result.output == "hello world!"
        assert isinstance(result.output, str)

    @pytest.mark.asyncio
    async def test_binary_streaming_agent_protocol(self, executor, mock_step, mock_resources):
        """Test that BinaryStreamingAgentProtocol works correctly."""
        # Arrange
        bytes_chunks = [b"data1", b"data2", b"data3"]
        agent = StubBinaryStreamingAgent(bytes_chunks)
        mock_step.agent = agent

        # Act
        result = await execute_simple_step(
            executor,
            step=mock_step,
            data="test input",
            context=None,
            resources=mock_resources,
            stream=True,
        )

        # Assert
        assert isinstance(result, StepResult)
        assert result.output == b"data1data2data3"
        assert isinstance(result.output, bytes)

    @pytest.mark.asyncio
    async def test_legacy_streaming_agent_fallback(self, executor, mock_step, mock_resources):
        """Test that legacy streaming agents still work with runtime type checking."""
        # Arrange
        string_chunks = ["hello", " ", "world", "!"]
        agent = StubLegacyStreamingAgent(string_chunks)
        mock_step.agent = agent

        # Act
        result = await execute_simple_step(
            executor,
            step=mock_step,
            data="test input",
            context=None,
            resources=mock_resources,
            stream=True,
        )

        # Assert
        assert isinstance(result, StepResult)
        assert result.output == "hello world!"
        assert isinstance(result.output, str)

    @pytest.mark.asyncio
    async def test_legacy_binary_streaming_agent_fallback(
        self, executor, mock_step, mock_resources
    ):
        """Test that legacy binary streaming agents work with runtime type checking."""
        # Arrange
        bytes_chunks = [b"data1", b"data2", b"data3"]
        agent = StubLegacyStreamingAgent(bytes_chunks)
        mock_step.agent = agent

        # Act
        result = await execute_simple_step(
            executor,
            step=mock_step,
            data="test input",
            context=None,
            resources=mock_resources,
            stream=True,
        )

        # Assert
        assert isinstance(result, StepResult)
        assert result.output == b"data1data2data3"
        assert isinstance(result.output, bytes)

    @pytest.mark.asyncio
    async def test_mixed_stream_types_handled_gracefully(self, executor, mock_step, mock_resources):
        """Test that mixed stream types are handled gracefully."""
        # Arrange - This should fall back to str(chunks) for mixed types
        mixed_chunks = ["text", b"binary", "more_text"]
        agent = StubLegacyStreamingAgent(mixed_chunks)
        mock_step.agent = agent

        # Act
        result = await execute_simple_step(
            executor,
            step=mock_step,
            data="test input",
            context=None,
            resources=mock_resources,
            stream=True,
        )

        # Assert - Mixed types should fall back to string representation
        assert isinstance(result, StepResult)
        # This should be the string representation of the list
        expected_str = str(mixed_chunks)
        assert result.output == expected_str
        assert isinstance(result.output, str)

    @pytest.mark.asyncio
    async def test_empty_stream_handled_correctly(self, executor, mock_step, mock_resources):
        """Test that empty streams are handled correctly."""
        # Arrange
        empty_chunks = []
        agent = StubTextStreamingAgent(empty_chunks)
        mock_step.agent = agent

        # Act
        result = await execute_simple_step(
            executor,
            step=mock_step,
            data="test input",
            context=None,
            resources=mock_resources,
            stream=True,
        )

        # Assert
        assert isinstance(result, StepResult)
        assert result.output == ""
        assert isinstance(result.output, str)

    @pytest.mark.asyncio
    async def test_single_bytes_chunk(self, executor, mock_step, mock_resources):
        """Test handling of a single bytes chunk."""
        # Arrange
        single_bytes_chunk = [b"single_chunk"]
        agent = StubBinaryStreamingAgent(single_bytes_chunk)
        mock_step.agent = agent

        # Act
        result = await execute_simple_step(
            executor,
            step=mock_step,
            data="test input",
            context=None,
            resources=mock_resources,
            stream=True,
        )

        # Assert
        assert isinstance(result, StepResult)
        assert result.output == b"single_chunk"
        assert isinstance(result.output, bytes)

    @pytest.mark.asyncio
    async def test_large_bytes_stream(self, executor, mock_step, mock_resources):
        """Test handling of a large bytes stream to ensure performance."""
        # Arrange
        large_bytes_chunks = [b"chunk" * 1000 for _ in range(10)]
        agent = StubBinaryStreamingAgent(large_bytes_chunks)
        mock_step.agent = agent

        # Act
        result = await execute_simple_step(
            executor,
            step=mock_step,
            data="test input",
            context=None,
            resources=mock_resources,
            stream=True,
        )

        # Assert
        assert isinstance(result, StepResult)
        expected_bytes = b"chunk" * 1000 * 10
        assert result.output == expected_bytes
        assert isinstance(result.output, bytes)
        assert (
            len(result.output) == 50000
        )  # Each 'chunk' is a 5-byte binary string (b"chunk"), repeated 1000 times per chunk and 10 chunks in total.

    @pytest.mark.asyncio
    async def test_protocol_type_safety(self):
        """Test that the new protocols provide proper type safety."""
        # Test that TextOnlyStreamingAgentProtocol is properly typed
        text_agent = StubTextStreamingAgent(["hello"])
        assert isinstance(text_agent, TextOnlyStreamingAgentProtocol)

        # Test that BinaryOnlyStreamingAgentProtocol is properly typed
        binary_agent = StubBinaryStreamingAgent([b"data"])
        assert isinstance(binary_agent, BinaryOnlyStreamingAgentProtocol)

        # Test that legacy protocol still works
        legacy_agent = StubLegacyStreamingAgent(["hello"])
        assert isinstance(legacy_agent, StreamingAgentProtocol)
