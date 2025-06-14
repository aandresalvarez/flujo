"""Defines the protocol for agent-like objects in the orchestrator."""

from __future__ import annotations

from typing import Protocol, TypeVar, Any, Optional, runtime_checkable
from ..infra.agents import AsyncAgentProtocol, T_co

T_Input = TypeVar("T_Input", contravariant=True)


@runtime_checkable
class AgentProtocol(AsyncAgentProtocol[T_co], Protocol[T_Input, T_co]):
    """Essential interface for all agent types used by the Orchestrator."""

    async def run(self, input_data: Optional[T_Input] = None, **kwargs: Any) -> T_co:
        """Asynchronously run the agent with the given input and return a result."""
        ...

    async def run_async(self, input_data: Optional[T_Input] = None, **kwargs: Any) -> T_co:
        """Alias for run() to maintain compatibility with AsyncAgentProtocol."""
        return await self.run(input_data, **kwargs)

# Explicit exports
__all__ = ['AgentProtocol', 'T_Input']
