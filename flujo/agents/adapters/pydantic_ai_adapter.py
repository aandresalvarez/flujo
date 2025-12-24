"""Adapter for pydantic-ai agents to Flujo's vendor-agnostic interface.

This module provides adapters that wrap pydantic-ai agents and convert their
responses to FlujoAgentResult, isolating pydantic-ai-specific details from
Flujo's orchestration layer.
"""

from __future__ import annotations

from typing import Any, Optional

from flujo.domain.agent_result import FlujoAgentResult, FlujoAgentUsage
from flujo.agents.agent_like import AgentLike


class PydanticAIUsageAdapter:
    """Adapts pydantic-ai usage objects to Flujo interface.

    This adapter converts pydantic-ai's usage objects (which may have different
    attribute names across versions) to the FlujoAgentUsage protocol.

    Note: This class satisfies the FlujoAgentUsage protocol by exposing
    input_tokens, output_tokens, and cost_usd as public attributes.
    """

    def __init__(self, pydantic_usage: Any) -> None:
        """Initialize adapter with pydantic-ai usage object.

        Args:
            pydantic_usage: The usage object from pydantic-ai (RunUsage or Usage).
        """
        self._usage = pydantic_usage
        # Pre-compute values for protocol compliance and performance
        self.input_tokens: int = getattr(
            pydantic_usage, "input_tokens", getattr(pydantic_usage, "request_tokens", 0)
        )
        self.output_tokens: int = getattr(
            pydantic_usage,
            "output_tokens",
            getattr(pydantic_usage, "response_tokens", 0),
        )
        self.cost_usd: Optional[float] = getattr(pydantic_usage, "cost_usd", None)


class PydanticAIAdapter:
    """Adapter that wraps pydantic-ai agents and converts responses.

    This adapter wraps a pydantic-ai agent and converts its responses to
    FlujoAgentResult, isolating pydantic-ai-specific details from Flujo's
    orchestration layer.

    The adapter also filters kwargs to only pass those that the underlying
    agent's run method accepts, preventing TypeError for unsupported arguments.
    """

    def __init__(self, pydantic_agent: AgentLike) -> None:
        """Initialize adapter with pydantic-ai agent.

        Args:
            pydantic_agent: The pydantic-ai agent to wrap (must conform to AgentLike protocol).
        """
        self._agent = pydantic_agent

    def _filter_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        """Filter kwargs to only include those accepted by the underlying agent.

        This prevents TypeError when passing vendor-specific kwargs to agents
        that don't support them (e.g., response_format for simple agents).
        """
        from flujo.application.core.context.context_manager import _accepts_param

        filtered: dict[str, Any] = {}

        for key, value in kwargs.items():
            # Context/pipeline_context handling
            if key in ["context", "pipeline_context"]:
                if _accepts_param(self._agent.run, "context"):
                    filtered["context"] = value
                continue

            # Structured options pass-through
            if key == "options" and isinstance(value, dict):
                if _accepts_param(self._agent.run, "options"):
                    filtered["options"] = value
                else:
                    # Expand only keys accepted by the underlying agent
                    for ok, ov in value.items():
                        if _accepts_param(self._agent.run, str(ok)):
                            filtered[str(ok)] = ov
                continue

            # For other kwargs, check if the agent accepts them
            accepts = _accepts_param(self._agent.run, str(key))
            if accepts is not False:
                filtered[str(key)] = value

        return filtered

    async def run(self, *args: Any, **kwargs: Any) -> FlujoAgentResult:
        """Run pydantic-ai agent and convert response to FlujoAgentResult.

        Args:
            *args: Positional arguments to pass to the agent's run method.
            **kwargs: Keyword arguments to pass to the agent's run method.
                These are filtered to only include kwargs that the underlying
                agent's run method accepts.

        Returns:
            FlujoAgentResult: Vendor-agnostic result containing output and usage metrics.
        """
        # Filter kwargs to only include those the agent accepts
        filtered_kwargs = self._filter_kwargs(**kwargs)
        raw_response = await self._agent.run(*args, **filtered_kwargs)

        # Extract usage if available (guard against None from usage())
        usage: Optional[FlujoAgentUsage] = None
        if hasattr(raw_response, "usage"):
            usage_attr = raw_response.usage
            pydantic_usage = usage_attr() if callable(usage_attr) else usage_attr
            if pydantic_usage is not None:
                usage = PydanticAIUsageAdapter(pydantic_usage)

        # Extract output (pydantic-ai responses have .output attribute)
        output = getattr(raw_response, "output", raw_response)

        # Extract explicit cost if present (for cases where cost is set directly)
        cost_usd = getattr(raw_response, "cost_usd", None)
        token_counts = getattr(raw_response, "token_counts", None)

        return FlujoAgentResult(
            output=output,
            usage=usage,
            cost_usd=cost_usd,
            token_counts=token_counts,
        )


__all__ = [
    "PydanticAIAdapter",
    "PydanticAIUsageAdapter",
]
