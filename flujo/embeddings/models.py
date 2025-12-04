"""Models for embedding operations and results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, TYPE_CHECKING, Union, runtime_checkable, Any

if TYPE_CHECKING:
    from pydantic_ai.usage import RunUsage as PydanticRunUsage
else:  # pragma: no cover - runtime fallback
    try:
        from pydantic_ai.usage import RunUsage as PydanticRunUsage
    except Exception:
        try:
            from pydantic_ai.usage import Usage as PydanticRunUsage
        except Exception:
            PydanticRunUsage = object  # Fallback when pydantic-ai is absent


@runtime_checkable
class UsageLike(Protocol):
    """Minimal usage surface used for cost tracking across pydantic-ai versions."""

    input_tokens: int | None
    output_tokens: int | None
    request_tokens: int | None
    response_tokens: int | None
    total_tokens: int | None
    requests: int | None


UsageType = Union[PydanticRunUsage, UsageLike]


def resolve_usage_constructor() -> type[Any]:
    """Return the best-available RunUsage/Usage class for runtime construction."""
    try:
        from pydantic_ai.usage import RunUsage as ctor

        return ctor
    except Exception:
        try:
            from pydantic_ai.usage import Usage as ctor

            return ctor
        except Exception:
            from types import SimpleNamespace

            return SimpleNamespace


@dataclass
class EmbeddingResult:
    """
    Result from an embedding operation.

    This class implements the UsageReportingProtocol by providing a .usage() method
    that returns the usage information from the embedding operation.
    """

    embeddings: List[List[float]]
    """The embedding vectors returned by the embedding model."""

    usage_info: UsageType
    """Usage information from the embedding operation."""

    def usage(self) -> UsageType:
        """
        Return the usage information from this embedding operation.

        This method implements the UsageReportingProtocol, making EmbeddingResult
        compatible with the existing cost tracking system.

        Returns
        -------
        UsageType
            The usage information from the embedding operation.
        """
        return self.usage_info
