"""Flujo-controlled agent result interface that abstracts vendor-specific details.

This module provides vendor-agnostic interfaces for agent results and usage metrics,
allowing Flujo to work with different agent backends (pydantic-ai, LangChain, etc.)
without tight coupling to any specific vendor.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class FlujoAgentUsage(Protocol):
    """Vendor-agnostic usage metrics interface.

    This protocol defines the standard interface for usage metrics that Flujo
    expects from any agent backend. Implementations should adapt their vendor-specific
    usage objects to match this interface.
    """

    input_tokens: int
    """Number of input/prompt tokens used."""

    output_tokens: int
    """Number of output/completion tokens used."""

    cost_usd: Optional[float]
    """Cost in USD, if available. None if not provided by the agent backend."""


class FlujoAgentResult:
    """Flujo-controlled agent result that abstracts vendor specifics.

    This class provides a vendor-agnostic representation of agent execution results.
    It encapsulates the output, usage metrics, and cost information in a way that
    Flujo's orchestration layer can consume without knowing about the underlying
    agent backend implementation.

    Attributes:
        output: The actual output from the agent (any type).
        cost_usd: Optional explicit cost in USD (for cases where cost is set
            directly on the result object rather than in usage).
        token_counts: Optional total token count (for explicit cost reporting
            when detailed usage breakdown is not available).
    """

    def __init__(
        self,
        output: Any,
        usage: Optional[FlujoAgentUsage] = None,
        cost_usd: Optional[float] = None,
        token_counts: Optional[int] = None,
    ) -> None:
        """Initialize FlujoAgentResult.

        Args:
            output: The actual output from the agent.
            usage: Optional usage metrics conforming to FlujoAgentUsage protocol.
            cost_usd: Optional explicit cost in USD.
            token_counts: Optional total token count.
        """
        self.output = output
        self._usage = usage
        self.cost_usd = cost_usd
        self.token_counts = token_counts

    def usage(self) -> Optional[FlujoAgentUsage]:
        """Get usage metrics.

        This method provides backward compatibility with code that expects
        usage to be accessed as a method call (e.g., result.usage()).

        Returns:
            Optional[FlujoAgentUsage]: Usage metrics if available, None otherwise.
        """
        return self._usage


__all__ = [
    "FlujoAgentResult",
    "FlujoAgentUsage",
]
