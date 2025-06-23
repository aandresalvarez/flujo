"""Shared type aliases for the domain layer."""

from typing import Callable, Coroutine, Any, TypeVar, TYPE_CHECKING
from pydantic import BaseModel

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from .events import HookPayload

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from .resources import AppResources  # noqa: F401

# Generic type for pipeline context models
ContextT = TypeVar("ContextT", bound=BaseModel, contravariant=True)

# A hook is an async callable that receives a typed payload object.
HookCallable = Callable[["HookPayload"], Coroutine[Any, Any, None]]
