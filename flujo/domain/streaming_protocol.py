from __future__ import annotations

from typing import Protocol, AsyncIterator, Any


class StreamingAgentProtocol(Protocol):
    """Protocol for agents that can stream their output."""

    async def stream(self, data: Any, **kwargs: Any) -> AsyncIterator[Any]:
        """Asynchronously yield output chunks."""
        # This is a protocol method - actual implementations will override this
        raise NotImplementedError("Subclasses must implement stream")
