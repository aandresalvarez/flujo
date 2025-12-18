from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AgentLike(Protocol):
    """Vendor-agnostic protocol for agent-like objects.

    This protocol defines the minimal interface required by Flujo's agent layer.
    It is intentionally minimal to support different agent backends (pydantic-ai,
    LangChain, custom implementations, etc.) without coupling to any specific vendor.

    The protocol only requires a `run` method that accepts arbitrary arguments.
    Vendor-specific parameters (model settings, usage limits, tool configurations)
    should be passed via **kwargs and handled by the concrete implementation or
    adapter layer.

    Example implementations:
        - pydantic-ai Agent: Wrapped by PydanticAIAdapter
        - LangChain Agent: Would need a LangChainAdapter
        - Custom Agent: Direct implementation

    The AsyncAgentWrapper filters kwargs based on what the underlying agent's
    run method actually accepts, providing compatibility across implementations.
    """

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the agent with the given input.

        This method signature is intentionally minimal and vendor-agnostic.
        All parameters are passed through to allow maximum flexibility.

        Args:
            *args: Positional arguments. Typically the first arg is the user
                prompt/input data.
            **kwargs: Keyword arguments. May include vendor-specific parameters
                such as:
                - pydantic-ai: message_history, deps, model_settings, usage_limits
                - LangChain: callbacks, tags, metadata
                - Custom: Any implementation-specific options

        Returns:
            Any: The agent's response in vendor-specific format. This will be
                adapted to FlujoAgentResult by the appropriate adapter
                (e.g., PydanticAIAdapter).
        """
        ...
