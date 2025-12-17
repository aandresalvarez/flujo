from types import TracebackType
from typing import Any, Protocol, runtime_checkable, Optional, Type

from flujo.domain.models import BaseModel


class AppResources(BaseModel):
    """Base class for user-defined resource containers."""


@runtime_checkable
class AsyncResourceContextManager(Protocol):
    """Async context manager for resources that need per-attempt lifecycle handling."""

    async def __aenter__(self) -> Any: ...

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None: ...


@runtime_checkable
class ResourceContextManager(Protocol):
    """Sync context manager for resources that need per-attempt lifecycle handling."""

    def __enter__(self) -> Any: ...

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None: ...
