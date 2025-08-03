
from typing import Protocol, TypeVar, Dict, Any, List
from ...domain.models import BaseModel

class ContextWithScratchpad(Protocol):
    """A contract ensuring a context object has a scratchpad attribute."""
    scratchpad: Dict[str, Any]
    executed_branches: List[str]

# For now, we'll use BaseModel as the bound and rely on runtime checks for scratchpad
# This maintains backward compatibility while providing some type safety
TContext_w_Scratch = TypeVar("TContext_w_Scratch", bound=ContextWithScratchpad)
