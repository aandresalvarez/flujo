"""
Test for the critical streaming bytes corruption bug.

This test demonstrates the bug where streaming bytes payloads are corrupted
during reassembly in the UltraStepExecutor.
"""

import pytest
from unittest.mock import MagicMock
from typing import AsyncIterator
from typing import Any

from flujo.application.core.ultra_executor import UltraStepExecutor
from flujo.domain.dsl.step import Step
from flujo.domain.models import StepResult
from flujo.domain.resources import AppResources


class StubStreamingAgent:
    """A stub agent that can be configured to yield specific chunks."""

    def __init__(self, chunks: list[str | bytes]):
        self.chunks = chunks

    async def stream(self, data: Any, **kwargs) -> AsyncIterator[str | bytes]:
        """Stream the configured chunks."""
        for chunk in self.chunks:
            yield chunk


class TestStreamingBytesBug:
    """Test suite for the streaming bytes corruption bug."""

    @pytest.fixture
    def executor(self):
        """Create a fresh UltraStepExecutor instance."""
        return UltraStepExecutor()

    @pytest.fixture
    def mock_step(self):
        """Create a mock step for testing."""
        step = MagicMock(spec=Step)
        step.name = "test_step"
        step.processors = None
        step.config = MagicMock()
        step.config.max_retries = 3
        return step

    @pytest.fixture
    def mock_resources(self):
        """Create mock resources."""
        return MagicMock(spec=AppResources)

    @pytest.mark.asyncio
    async def test_string_stream_works_correctly(self, executor, mock_step, mock_resources):
        """Test that string streams work correctly (existing functionality)."""
        # Arrange
        string_chunks = ["hello", " ", "world", "!"]
        agent = StubStreamingAgent(string_chunks)
        mock_step.agent = agent

        # Act
        result = await executor.execute_step(
            step=mock_step, data="test input", context=None, resources=mock_resources, stream=True
        )

        # Assert
        assert isinstance(result, StepResult)
        assert result.output == "hello world!"
        assert isinstance(result.output, str)

    @pytest.mark.asyncio
    async def test_bytes_stream_corruption_bug(self, executor, mock_step, mock_resources):
        """Test that demonstrates the bytes corruption bug."""
        # Arrange
        bytes_chunks = [b"data1", b"data2", b"data3"]
        agent = StubStreamingAgent(bytes_chunks)
        mock_step.agent = agent

        # Act
        result = await executor.execute_step(
            step=mock_step, data="test input", context=None, resources=mock_resources, stream=True
        )

        # Assert - This will fail with the current bug
        assert isinstance(result, StepResult)
        # The bug causes this to be a string representation instead of concatenated bytes
        assert result.output == b"data1data2data3"
        assert isinstance(result.output, bytes)

    @pytest.mark.asyncio
    async def test_mixed_stream_types_handled_gracefully(self, executor, mock_step, mock_resources):
        """Test that mixed stream types are handled gracefully."""
        # Arrange - This should fall back to str(chunks) for mixed types
        mixed_chunks = ["text", b"binary", "more_text"]
        agent = StubStreamingAgent(mixed_chunks)
        mock_step.agent = agent

        # Act
        result = await executor.execute_step(
            step=mock_step, data="test input", context=None, resources=mock_resources, stream=True
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
        agent = StubStreamingAgent(empty_chunks)
        mock_step.agent = agent

        # Act
        result = await executor.execute_step(
            step=mock_step, data="test input", context=None, resources=mock_resources, stream=True
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
        agent = StubStreamingAgent(single_bytes_chunk)
        mock_step.agent = agent

        # Act
        result = await executor.execute_step(
            step=mock_step, data="test input", context=None, resources=mock_resources, stream=True
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
        agent = StubStreamingAgent(large_bytes_chunks)
        mock_step.agent = agent

        # Act
        result = await executor.execute_step(
            step=mock_step, data="test input", context=None, resources=mock_resources, stream=True
        )

        # Assert
        assert isinstance(result, StepResult)
        expected_bytes = b"chunk" * 1000 * 10
        assert result.output == expected_bytes
        assert isinstance(result.output, bytes)
        assert len(result.output) == 50000  # 5 bytes * 1000 * 10
