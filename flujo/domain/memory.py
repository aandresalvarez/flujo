from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@dataclass
class MemoryRecord:
    """A single vector-backed memory item.

    Note: slots=True removed for Python 3.13+ compatibility with serialization libraries
    that rely on __dict__ access. ScoredMemory retains slots=True as it's internal only.
    """

    vector: Sequence[float]
    payload: Any | None = None
    metadata: Mapping[str, Any] | None = None
    id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class VectorQuery:
    """Query parameters for vector search.

    Note: slots=True removed for Python 3.13+ compatibility with serialization libraries
    that rely on __dict__ access. ScoredMemory retains slots=True as it's internal only.
    """

    vector: Sequence[float]
    limit: int = 5
    filter_metadata: Mapping[str, Any] | None = None


@dataclass(slots=True)
class ScoredMemory:
    """MemoryRecord with similarity score."""

    record: MemoryRecord
    score: float


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Abstract interface for vector-backed long-term memory stores."""

    async def add(self, records: Sequence[MemoryRecord]) -> None:
        """Add or upsert a collection of memory records."""

    async def query(self, query: VectorQuery) -> list[ScoredMemory]:
        """Return the top-k scored memories for the provided vector query."""

    async def delete(self, ids: Sequence[str]) -> None:
        """Delete memories by identifier."""

    async def close(self) -> None:
        """Release any resources held by the store."""


def _cosine_similarity(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    """Compute cosine similarity for two vectors."""

    if len(lhs) != len(rhs):
        raise ValueError("Vector dimensions must match for similarity computation")
    lhs_norm = math.sqrt(sum(component * component for component in lhs))
    rhs_norm = math.sqrt(sum(component * component for component in rhs))
    if lhs_norm == 0.0 or rhs_norm == 0.0:
        return 0.0
    return sum(left * right for left, right in zip(lhs, rhs)) / (lhs_norm * rhs_norm)


def _assign_id(record: MemoryRecord) -> MemoryRecord:
    """Ensure the record has an identifier.

    Notes:
    - If the caller supplies an id, it is preserved and will upsert on stores that
      use ON CONFLICT(id) semantics (e.g., Postgres vector). Supplying your own id
      therefore implies "replace/update this record".
    - If no id is provided, a UUID is assigned to avoid collisions across runs.
    """

    if record.id is None or record.id == "":
        record.id = uuid.uuid4().hex
    return record
