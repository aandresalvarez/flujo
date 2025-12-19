"""Fallback chain management to prevent infinite loops."""

from __future__ import annotations
import contextvars
from typing import Protocol

from ....exceptions import InfiniteFallbackError


class StepType(Protocol):
    """Minimal step contract for fallback tracking."""

    name: str


# Context variables for tracking (initialized lazily to avoid shared mutable defaults)
_FALLBACK_RELATIONSHIPS: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "fallback_relationships", default=None
)
_FALLBACK_CHAIN: contextvars.ContextVar[list[StepType] | None] = contextvars.ContextVar(
    "fallback_chain", default=None
)
_FALLBACK_GRAPH_CACHE: contextvars.ContextVar[dict[str, bool] | None] = contextvars.ContextVar(
    "fallback_graph_cache", default=None
)


def _get_relationships() -> dict[str, str]:
    rel = _FALLBACK_RELATIONSHIPS.get()
    if rel is None:
        rel = {}
        _FALLBACK_RELATIONSHIPS.set(rel)
    return rel


def _get_chain() -> list[StepType]:
    chain = _FALLBACK_CHAIN.get()
    if chain is None:
        chain = []
        _FALLBACK_CHAIN.set(chain)
    return chain


def _get_graph_cache() -> dict[str, bool]:
    cache = _FALLBACK_GRAPH_CACHE.get()
    if cache is None:
        cache = {}
        _FALLBACK_GRAPH_CACHE.set(cache)
    return cache


class FallbackHandler:
    """Manages fallback step execution with loop detection."""

    MAX_CHAIN_LENGTH: int = 10
    MAX_DETECTION_ITERATIONS: int = 100

    def __init__(self) -> None:
        self._visited_steps: set[str] = set()

    def register_fallback(self, primary_step: StepType, fallback_step: StepType) -> None:
        """Register a fallback relationship for loop detection."""
        relationships = _get_relationships()
        relationships[primary_step.name] = fallback_step.name
        _FALLBACK_RELATIONSHIPS.set(relationships)

    def push_to_chain(self, step: StepType) -> None:
        """Add step to the current fallback chain."""
        chain = _get_chain()
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
        chain = _get_chain()
        if chain:
            chain.pop()
            _FALLBACK_CHAIN.set(chain)

    def check_for_loop(self, step: StepType) -> bool:
        """Check if adding this step would create a loop."""
        cache = _get_graph_cache()
        step_name = step.name

        if step_name in cache:
            return cache[step_name]

        # Detect cycle using visited set
        chain = _get_chain()
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
        return len(_get_chain())

    def is_step_in_chain(self, step: StepType) -> bool:
        """Check if a step is already in the current fallback chain."""
        chain = _get_chain()
        chain_names = {s.name for s in chain}
        return step.name in chain_names
