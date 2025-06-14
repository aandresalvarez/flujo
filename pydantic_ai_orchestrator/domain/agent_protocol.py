"""Defines the protocol for agent-like objects in the orchestrator."""

from typing import Protocol, TypeVar, Any, Optional, runtime_checkable

T_Input = TypeVar("T_Input", contravariant=True)
T_Output = TypeVar("T_Output", covariant=True)


@runtime_checkable
class AgentProtocol(Protocol[T_Input, T_Output]):
    """Essential interface for all agent types used by the Orchestrator."""

    async def run(self, input_data: Optional[T_Input] = None, **kwargs: Any) -> T_Output:
        """Asynchronously run the agent with the given input and return a result."""
        ...
