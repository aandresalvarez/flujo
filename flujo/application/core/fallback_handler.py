"""Fallback chain management to prevent infinite loops."""

from __future__ import annotations
import contextvars
from typing import Any, Dict, List, Set, TYPE_CHECKING

from ...exceptions import InfiniteFallbackError

if TYPE_CHECKING:
    from ...domain.dsl.step import Step as _DSLStep

    StepType = _DSLStep[Any, Any]
else:
    StepType = Any  # type: ignore

# Context variables for tracking
_FALLBACK_RELATIONSHIPS: contextvars.ContextVar[Dict[str, str]] = contextvars.ContextVar(
    "fallback_relationships", default={}
)
_FALLBACK_CHAIN: contextvars.ContextVar[List[StepType]] = contextvars.ContextVar(
    "fallback_chain", default=[]
)
_FALLBACK_GRAPH_CACHE: contextvars.ContextVar[Dict[str, bool]] = contextvars.ContextVar(
    "fallback_graph_cache", default={}
)


class FallbackHandler:
    """Manages fallback step execution with loop detection."""

    MAX_CHAIN_LENGTH: int = 10
    MAX_DETECTION_ITERATIONS: int = 100

    def __init__(self) -> None:
        self._visited_steps: Set[str] = set()

    def register_fallback(self, primary_step: "StepType", fallback_step: "StepType") -> None:
        """Register a fallback relationship for loop detection."""
        relationships = _FALLBACK_RELATIONSHIPS.get()
        relationships[primary_step.name] = fallback_step.name
        _FALLBACK_RELATIONSHIPS.set(relationships)

    def push_to_chain(self, step: "StepType") -> None:
        """Add step to the current fallback chain."""
        chain = _FALLBACK_CHAIN.get()
        if len(chain) >= self.MAX_CHAIN_LENGTH:
            chain_names = [s.name for s in chain]
            raise InfiniteFallbackError(
                f"Fallback chain exceeded maximum length ({self.MAX_CHAIN_LENGTH}). "
                f"Chain: {' -> '.join(chain_names)}"
            )
        chain.append(step)
        _FALLBACK_CHAIN.set(chain)

    def pop_from_chain(self) -> None:
        """Remove last step from the fallback chain."""
        chain = _FALLBACK_CHAIN.get()
        if chain:
            chain.pop()
            _FALLBACK_CHAIN.set(chain)

    def check_for_loop(self, step: "StepType") -> bool:
        """Check if adding this step would create a loop."""
        cache = _FALLBACK_GRAPH_CACHE.get()
        step_name = step.name

        if step_name in cache:
            return cache[step_name]

        # Detect cycle using visited set
        chain = _FALLBACK_CHAIN.get()
        chain_names = {s.name for s in chain}

        if step_name in chain_names:
            cache[step_name] = True
            _FALLBACK_GRAPH_CACHE.set(cache)
            return True

        cache[step_name] = False
        _FALLBACK_GRAPH_CACHE.set(cache)
        return False

    def reset(self) -> None:
        """Reset all fallback tracking state."""
        _FALLBACK_RELATIONSHIPS.set({})
        _FALLBACK_CHAIN.set([])
        _FALLBACK_GRAPH_CACHE.set({})

    def get_current_chain_length(self) -> int:
        """Get the current length of the fallback chain."""
        return len(_FALLBACK_CHAIN.get())

    def is_step_in_chain(self, step: "StepType") -> bool:
        """Check if a step is already in the current fallback chain."""
        chain = _FALLBACK_CHAIN.get()
        chain_names = {s.name for s in chain}
        return step.name in chain_names
