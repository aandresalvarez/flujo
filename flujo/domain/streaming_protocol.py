"""Defines protocols for streaming agents with explicit data type handling."""

from __future__ import annotations

from typing import Protocol, AsyncIterator, Any, TypeVar, runtime_checkable

try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

# Type variables for streaming data types
StreamInT = TypeVar("StreamInT", contravariant=True)
TextOutT = TypeVar("TextOutT", bound=str, covariant=True)
BinaryOutT = TypeVar("BinaryOutT", bound=bytes, covariant=True)


@runtime_checkable
class TextStreamingAgentProtocol(Protocol[StreamInT, TextOutT]):
    """Protocol for agents that stream text data.

    This protocol explicitly defines that the agent streams text chunks,
    making the data type clear at the interface level.
    """

    async def stream(self, data: StreamInT, **kwargs: Any) -> AsyncIterator[TextOutT]:
        """Asynchronously yield text output chunks."""
        ...


@runtime_checkable
class BinaryStreamingAgentProtocol(Protocol[StreamInT, BinaryOutT]):
    """Protocol for agents that stream binary data.

    This protocol explicitly defines that the agent streams binary chunks,
    making the data type clear at the interface level.
    """

    async def stream(self, data: StreamInT, **kwargs: Any) -> AsyncIterator[BinaryOutT]:
        """Asynchronously yield binary output chunks."""
        ...


# More specific protocols that are mutually exclusive
@runtime_checkable
class TextOnlyStreamingAgentProtocol(Protocol[StreamInT]):
    """Protocol for agents that stream ONLY text data."""

    async def stream(self, data: StreamInT, **kwargs: Any) -> AsyncIterator[str]:
        """Asynchronously yield text output chunks."""
        ...


@runtime_checkable
class BinaryOnlyStreamingAgentProtocol(Protocol[StreamInT]):
    """Protocol for agents that stream ONLY binary data."""

    async def stream(self, data: StreamInT, **kwargs: Any) -> AsyncIterator[bytes]:
        """Asynchronously yield binary output chunks."""
        ...


@runtime_checkable
class StreamingAgentProtocol(Protocol[StreamInT]):
    """Legacy protocol for backward compatibility.

    This protocol allows any data type to be streamed, but doesn't provide
    type safety. New code should use TextStreamingAgentProtocol or
    BinaryStreamingAgentProtocol for explicit type safety.
    """

    async def stream(self, data: StreamInT, **kwargs: Any) -> AsyncIterator[Any]:
        """Asynchronously yield output chunks of any type."""
        ...


# Type-safe streaming protocols for common use cases
TextStreamingAgent: TypeAlias = TextStreamingAgentProtocol[Any, str]
BinaryStreamingAgent: TypeAlias = BinaryStreamingAgentProtocol[Any, bytes]

# Legacy type for backward compatibility
StreamingAgent: TypeAlias = StreamingAgentProtocol[Any]
