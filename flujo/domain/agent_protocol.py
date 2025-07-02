"""Defines the protocol for agent-like objects in the orchestrator."""

from __future__ import annotations

from typing import Protocol, TypeVar, Any, runtime_checkable

from .types import ContextT

AgentInT = TypeVar("AgentInT", contravariant=True)
AgentOutT = TypeVar("AgentOutT", covariant=True)


@runtime_checkable
class AsyncAgentProtocol(Protocol[AgentInT, AgentOutT]):
    """Generic asynchronous agent interface.

    Note: For agents requiring a typed pipeline context, implementing the
    :class:`flujo.domain.ContextAwareAgentProtocol` is now the recommended and
    type-safe approach.
    """

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


@runtime_checkable
class ContextAwareAgentProtocol(Protocol[AgentInT, AgentOutT, ContextT]):
    """A protocol for agents that are aware of a specific pipeline context type."""

    __context_aware__: bool = True

    async def run(
        self,
        data: AgentInT,
        *,
        context: ContextT,
        **kwargs: Any,
    ) -> AgentOutT: ...


# Explicit exports
__all__ = [
    "AgentProtocol",
    "AsyncAgentProtocol",
    "ContextAwareAgentProtocol",
    "T_Input",
    "AgentInT",
    "AgentOutT",
    "ContextT",
]
