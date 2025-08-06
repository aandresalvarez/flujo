from typing import Optional, List
import copy
from pydantic import BaseModel
from ...utils.context import safe_merge_context_updates


class ContextManager:
    """Centralized context isolation and merging."""

    @staticmethod
    def isolate(context: Optional[BaseModel], include_keys: Optional[List[str]] = None) -> Optional[BaseModel]:
        """Return a deep copy of the context for isolation."""
        if context is None:
            return None
        # Selective isolation: include only specified keys if requested
        if include_keys:
            try:
                # Pydantic deep copy with include set
                return context.model_copy(include=set(include_keys), deep=True)  # type: ignore
            except Exception:
                # Fallback to manual key-based copy
                try:
                    data = {k: getattr(context, k) for k in include_keys if hasattr(context, k)}
                    return type(context)(**data)  # type: ignore
                except Exception:
                    pass
        try:
            # Use pydantic's deep copy when available
            return context.model_copy(deep=True)  # type: ignore
        except Exception:
            return copy.deepcopy(context)

    @staticmethod
    def merge(main_context: Optional[BaseModel], branch_context: Optional[BaseModel]) -> Optional[BaseModel]:
        """Merge updates from branch_context into main_context and return the result."""
        if main_context is None:
            return branch_context
        if branch_context is None:
            return main_context
        safe_merge_context_updates(main_context, branch_context)
        return main_context