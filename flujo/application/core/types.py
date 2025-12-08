from typing import (
    Protocol,
    TypeVar,
    Any,
    List,
    Generic,
    Optional,
    Callable,
    Awaitable,
    TYPE_CHECKING,
)
from dataclasses import dataclass
from flujo.type_definitions.common import JSONObject

from ...domain.resources import AppResources
from ...domain.models import UsageLimits, PipelineResult, Quota
from ...domain.interfaces import StepLike

if TYPE_CHECKING:
    pass  # pragma: no cover


class ContextWithScratchpad(Protocol):
    """A contract ensuring a context object has a scratchpad attribute."""

    scratchpad: JSONObject
    executed_branches: List[str]


# For now, we'll use BaseModel as the bound and rely on runtime checks for scratchpad
# This maintains backward compatibility while providing some type safety
TContext_w_Scratch = TypeVar("TContext_w_Scratch", bound=ContextWithScratchpad)


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
