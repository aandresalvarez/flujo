"""Defines the protocol for agent-like objects in the orchestrator."""

from __future__ import annotations

from typing import Protocol, TypeVar, Any, Optional, runtime_checkable

AgentInT = TypeVar("AgentInT", contravariant=True)
AgentOutT = TypeVar("AgentOutT", covariant=True)


@runtime_checkable
class AsyncAgentProtocol(Protocol[AgentInT, AgentOutT]):
    async def run(self, data: AgentInT, **kwargs: Any) -> AgentOutT: ...

    async def run_async(self, data: AgentInT, **kwargs: Any) -> AgentOutT:
        return await self.run(data, **kwargs)


T_Input = TypeVar("T_Input", contravariant=True)


@runtime_checkable
class AgentProtocol(AsyncAgentProtocol[T_Input, AgentOutT], Protocol[T_Input, AgentOutT]):
    """Essential interface for all agent types used by the Orchestrator."""

    async def run(self, data: T_Input, **kwargs: Any) -> AgentOutT:
        """Asynchronously run the agent with the given input and return a result."""
        ...

    async def run_async(self, data: T_Input, **kwargs: Any) -> AgentOutT:
        """Alias for run() to maintain compatibility with AsyncAgentProtocol."""
        return await self.run(data, **kwargs)


# Explicit exports
__all__ = ["AgentProtocol", "AsyncAgentProtocol", "T_Input", "AgentInT", "AgentOutT"]
