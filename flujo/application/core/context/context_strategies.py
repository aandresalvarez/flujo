from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol
from ....domain.models import BaseModel


class ContextIsolationStrategy(Protocol):
    """Strategy interface for context isolation/merge behavior."""

    def isolate(
        self, context: Optional[BaseModel], include_keys: Optional[List[str]] = None
    ) -> Optional[BaseModel]: ...

    def merge(
        self, main_context: Optional[BaseModel], branch_context: Optional[BaseModel]
    ) -> Optional[BaseModel]: ...


@dataclass
class LenientIsolation(ContextIsolationStrategy):
    """Default isolation/merge using best-effort deep copies and safe merges."""

    isolate_impl: Callable[[Optional[BaseModel], Optional[List[str]]], Optional[BaseModel]]
    merge_impl: Callable[[Optional[BaseModel], Optional[BaseModel]], Optional[BaseModel]]

    def isolate(
        self, context: Optional[BaseModel], include_keys: Optional[List[str]] = None
    ) -> Optional[BaseModel]:
        return self.isolate_impl(context, include_keys)

    def merge(
        self, main_context: Optional[BaseModel], branch_context: Optional[BaseModel]
    ) -> Optional[BaseModel]:
        return self.merge_impl(main_context, branch_context)


@dataclass
class StrictIsolation(ContextIsolationStrategy):
    """Strict isolation/merge that raises on failures."""

    isolate_impl: Callable[[Optional[BaseModel], Optional[List[str]]], Optional[BaseModel]]
    merge_impl: Callable[[Optional[BaseModel], Optional[BaseModel]], Optional[BaseModel]]

    def isolate(
        self, context: Optional[BaseModel], include_keys: Optional[List[str]] = None
    ) -> Optional[BaseModel]:
        return self.isolate_impl(context, include_keys)

    def merge(
        self, main_context: Optional[BaseModel], branch_context: Optional[BaseModel]
    ) -> Optional[BaseModel]:
        return self.merge_impl(main_context, branch_context)


@dataclass
class SelectiveIsolation(ContextIsolationStrategy):
    """Isolation/merge that focuses on a subset of context fields when provided."""

    isolate_impl: Callable[[Optional[BaseModel], Optional[List[str]]], Optional[BaseModel]]
    merge_impl: Callable[[Optional[BaseModel], Optional[BaseModel]], Optional[BaseModel]]
    include_keys: Optional[List[str]] = None

    def isolate(
        self, context: Optional[BaseModel], include_keys: Optional[List[str]] = None
    ) -> Optional[BaseModel]:
        keys = include_keys or self.include_keys
        return self.isolate_impl(context, keys)

    def merge(
        self, main_context: Optional[BaseModel], branch_context: Optional[BaseModel]
    ) -> Optional[BaseModel]:
        return self.merge_impl(main_context, branch_context)


__all__ = [
    "ContextIsolationStrategy",
    "LenientIsolation",
    "SelectiveIsolation",
    "StrictIsolation",
]
