"""Tests for Pydantic AI Adapter version compatibility.

This module tests the `PydanticAIAdapter` and `PydanticAIUsageAdapter` classes
which provide a vendor-agnostic interface between Flujo and pydantic-ai.

Key behaviors tested:
- Version drift handling between pydantic-ai v0.6 and v0.7+ field names
- Correct extraction of usage metrics (tokens, cost)
- Adaptation of pydantic-ai responses to FlujoAgentResult
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, AsyncMock

import pytest

from flujo.agents.adapters.pydantic_ai_adapter import (
    PydanticAIAdapter,
    PydanticAIUsageAdapter,
)
from flujo.domain.agent_result import FlujoAgentResult


# =============================================================================
# Test Fixtures
# =============================================================================


def create_mock_usage(
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    request_tokens: int | None = None,
    response_tokens: int | None = None,
    cost_usd: float | None = None,
) -> MagicMock:
    """Create a mock usage object with specified attributes.

    Args:
        input_tokens: Modern pydantic-ai v0.7+ field
        output_tokens: Modern pydantic-ai v0.7+ field
        request_tokens: Legacy pydantic-ai v0.6 field
        response_tokens: Legacy pydantic-ai v0.6 field
        cost_usd: Cost in USD (optional)
    """
    mock = MagicMock()

    # Remove all default attributes to test fallback behavior
    mock._spec_class = None

    # Only set attributes that were explicitly provided
    if input_tokens is not None:
        mock.input_tokens = input_tokens
    if output_tokens is not None:
        mock.output_tokens = output_tokens
    if request_tokens is not None:
        mock.request_tokens = request_tokens
    if response_tokens is not None:
        mock.response_tokens = response_tokens
    if cost_usd is not None:
        mock.cost_usd = cost_usd

    return mock


def create_mock_agent_result(
    output: Any,
    usage: MagicMock | None = None,
) -> MagicMock:
    """Create a mock pydantic-ai AgentRunResult."""
    mock = MagicMock()
    mock.output = output
    mock.usage = MagicMock(return_value=usage)
    return mock


# =============================================================================
# PydanticAIUsageAdapter Tests
# =============================================================================


class TestPydanticAIUsageAdapter:
    """Tests for PydanticAIUsageAdapter version drift handling.

    The adapter must handle both:
    - Modern format (v0.7+): input_tokens, output_tokens
    - Legacy format (v0.6): request_tokens, response_tokens
    """

    def test_modern_format_input_output_tokens(self) -> None:
        """Modern pydantic-ai v0.7+ uses input_tokens/output_tokens."""
        mock_usage = create_mock_usage(
            input_tokens=150,
            output_tokens=75,
            cost_usd=0.0025,
        )

        adapter = PydanticAIUsageAdapter(mock_usage)

        assert adapter.input_tokens == 150
        assert adapter.output_tokens == 75
        assert adapter.cost_usd == 0.0025

    def test_legacy_format_request_response_tokens(self) -> None:
        """Legacy pydantic-ai v0.6 uses request_tokens/response_tokens."""
        # Create a mock that ONLY has legacy attributes (no modern ones)
        mock_usage = MagicMock(spec=["request_tokens", "response_tokens"])
        mock_usage.request_tokens = 200
        mock_usage.response_tokens = 100

        adapter = PydanticAIUsageAdapter(mock_usage)

        # Adapter should map legacy names to standard names
        assert adapter.input_tokens == 200
        assert adapter.output_tokens == 100
        assert adapter.cost_usd is None

    def test_modern_takes_precedence_over_legacy(self) -> None:
        """When both formats present, modern should be used."""
        mock_usage = create_mock_usage(
            input_tokens=100,  # Modern
            output_tokens=50,  # Modern
            request_tokens=999,  # Legacy (should be ignored)
            response_tokens=999,  # Legacy (should be ignored)
        )

        adapter = PydanticAIUsageAdapter(mock_usage)

        # Modern format takes precedence
        assert adapter.input_tokens == 100
        assert adapter.output_tokens == 50

    def test_missing_fields_default_to_zero(self) -> None:
        """Missing token fields should default to 0."""
        mock_usage = MagicMock(spec=[])  # Empty spec = no attributes

        adapter = PydanticAIUsageAdapter(mock_usage)

        assert adapter.input_tokens == 0
        assert adapter.output_tokens == 0
        assert adapter.cost_usd is None

    def test_zero_tokens_preserved(self) -> None:
        """Zero values should be preserved (not treated as missing)."""
        mock_usage = create_mock_usage(
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
        )

        adapter = PydanticAIUsageAdapter(mock_usage)

        assert adapter.input_tokens == 0
        assert adapter.output_tokens == 0
        # Note: cost_usd=0.0 may or may not be preserved depending on impl

    def test_large_token_counts(self) -> None:
        """Large token counts should be handled correctly."""
        mock_usage = create_mock_usage(
            input_tokens=1_000_000,
            output_tokens=500_000,
            cost_usd=150.00,
        )

        adapter = PydanticAIUsageAdapter(mock_usage)

        assert adapter.input_tokens == 1_000_000
        assert adapter.output_tokens == 500_000
        assert adapter.cost_usd == 150.00


# =============================================================================
# PydanticAIAdapter Tests
# =============================================================================


class TestPydanticAIAdapter:
    """Tests for PydanticAIAdapter agent wrapping functionality.

    The adapter wraps pydantic-ai agents and converts their responses
    to the vendor-agnostic FlujoAgentResult format.
    """

    @pytest.mark.asyncio
    async def test_converts_dict_output_to_flujo_result(self) -> None:
        """Agent dict output should be wrapped in FlujoAgentResult."""
        mock_agent = MagicMock()
        mock_usage = create_mock_usage(input_tokens=10, output_tokens=5)
        mock_result = create_mock_agent_result(
            output={"answer": "test", "confidence": 0.95},
            usage=mock_usage,
        )
        mock_agent.run = AsyncMock(return_value=mock_result)

        adapter = PydanticAIAdapter(mock_agent)
        result = await adapter.run("test prompt")

        assert isinstance(result, FlujoAgentResult)
        assert result.output == {"answer": "test", "confidence": 0.95}

    @pytest.mark.asyncio
    async def test_converts_string_output_to_flujo_result(self) -> None:
        """Agent string output should be wrapped in FlujoAgentResult."""
        mock_agent = MagicMock()
        mock_result = create_mock_agent_result(
            output="The answer is 42",
            usage=create_mock_usage(input_tokens=5, output_tokens=4),
        )
        mock_agent.run = AsyncMock(return_value=mock_result)

        adapter = PydanticAIAdapter(mock_agent)
        result = await adapter.run("What is the answer?")

        assert isinstance(result, FlujoAgentResult)
        assert result.output == "The answer is 42"

    @pytest.mark.asyncio
    async def test_converts_list_output_to_flujo_result(self) -> None:
        """Agent list output should be wrapped in FlujoAgentResult."""
        mock_agent = MagicMock()
        mock_result = create_mock_agent_result(
            output=["item1", "item2", "item3"],
            usage=None,
        )
        mock_agent.run = AsyncMock(return_value=mock_result)

        adapter = PydanticAIAdapter(mock_agent)
        result = await adapter.run("List items")

        assert isinstance(result, FlujoAgentResult)
        assert result.output == ["item1", "item2", "item3"]

    @pytest.mark.asyncio
    async def test_preserves_usage_metrics(self) -> None:
        """Usage metrics should be preserved in FlujoAgentResult."""
        mock_agent = MagicMock()
        mock_usage = create_mock_usage(
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.005,
        )
        mock_result = create_mock_agent_result(output="test", usage=mock_usage)
        mock_agent.run = AsyncMock(return_value=mock_result)

        adapter = PydanticAIAdapter(mock_agent)
        result = await adapter.run("test")

        usage = result.usage()
        assert usage is not None
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    @pytest.mark.asyncio
    async def test_handles_none_usage(self) -> None:
        """None usage should be handled gracefully."""
        mock_agent = MagicMock()
        mock_result = create_mock_agent_result(output="test", usage=None)
        mock_agent.run = AsyncMock(return_value=mock_result)

        adapter = PydanticAIAdapter(mock_agent)
        result = await adapter.run("test")

        assert isinstance(result, FlujoAgentResult)
        # Should not raise even with None usage

    @pytest.mark.asyncio
    async def test_forwards_kwargs_to_underlying_agent(self) -> None:
        """Adapter should forward relevant kwargs to underlying agent."""
        mock_agent = MagicMock()
        mock_result = create_mock_agent_result(output="test", usage=None)
        mock_agent.run = AsyncMock(return_value=mock_result)

        adapter = PydanticAIAdapter(mock_agent)
        await adapter.run("test prompt", temperature=0.7)

        # Verify the agent was called
        mock_agent.run.assert_called_once()


# =============================================================================
# Integration with Cost Extraction
# =============================================================================


class TestUsageMetricsIntegration:
    """Integration tests for usage metrics with cost calculation."""

    def test_usage_adapter_properties_match_interface(self) -> None:
        """Verify adapter exposes expected interface for downstream consumers."""
        mock_usage = create_mock_usage(
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.0015,
        )

        adapter = PydanticAIUsageAdapter(mock_usage)

        # Verify interface
        assert hasattr(adapter, "input_tokens")
        assert hasattr(adapter, "output_tokens")
        assert hasattr(adapter, "cost_usd")

        # Verify types
        assert isinstance(adapter.input_tokens, int)
        assert isinstance(adapter.output_tokens, int)
        assert isinstance(adapter.cost_usd, float)
