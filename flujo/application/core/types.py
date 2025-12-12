from typing import (
    TypeVar,
    Any,
    Generic,
    Optional,
    Callable,
    Awaitable,
    TYPE_CHECKING,
)
from dataclasses import dataclass
from pydantic import BaseModel

from ...domain.resources import AppResources
from ...domain.models import UsageLimits, PipelineResult, Quota
from ...domain.interfaces import StepLike

if TYPE_CHECKING:
    pass  # pragma: no cover


TContext_w_Scratch = TypeVar("TContext_w_Scratch", bound=BaseModel)


@dataclass
class ExecutionFrame(Generic[TContext_w_Scratch]):
    """
    Encapsulates all state for a single step execution call.

    This provides a formal, type-safe data contract for internal execution calls,
    eliminating parameter-passing bugs and making recursive logic easier to reason about.
    """

    # Core execution parameters
    step: StepLike
    data: Any
    context: Optional[TContext_w_Scratch]
    resources: Optional[AppResources]
    limits: Optional[UsageLimits]

    # Streaming and callback parameters
    stream: bool
    on_chunk: Optional[Callable[[Any], Awaitable[None]]]
    # Context management
    context_setter: Callable[[PipelineResult[Any], Optional[Any]], None]

    # Optional quota for proactive reservations
    quota: Optional[Quota] = None

    # Optional parameters for backward compatibility and advanced features
    result: Optional[Any] = None  # For backward compatibility
    _fallback_depth: int = 0  # Track fallback recursion depth
