from typing import (
    TypeVar,
    Generic,
    Optional,
    Callable,
    Awaitable,
    TYPE_CHECKING,
)
from dataclasses import dataclass
from ....domain.models import BaseModel

from ....domain.models import UsageLimits, PipelineResult, Quota
from ....domain.interfaces import StepLike

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
    data: object
    context: Optional[TContext_w_Scratch]
    resources: object | None
    limits: Optional[UsageLimits]

    # Streaming and callback parameters
    stream: bool
    on_chunk: Optional[Callable[[object], Awaitable[None]]]
    # Context management
    context_setter: Callable[[PipelineResult[TContext_w_Scratch], TContext_w_Scratch | None], None]

    # Optional quota for proactive reservations
    quota: Optional[Quota] = None

    # Optional parameters for backward compatibility and advanced features
    result: object | None = None  # For backward compatibility
    _fallback_depth: int = 0  # Track fallback recursion depth
