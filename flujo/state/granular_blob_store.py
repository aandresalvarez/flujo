"""Granular Blob Store - Durable storage for large payloads.

Implements PRD v12 §8.1: Blob offloading for payloads exceeding threshold.
Provides offload/hydrate operations with fail-fast semantics on missing refs.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Protocol

    class StateBackend(Protocol):
        """Protocol for state backend."""

        async def save_state(self, key: str, data: object) -> None: ...
        async def load_state(self, key: str) -> object | None: ...


__all__ = ["BlobNotFoundError", "BlobRef", "GranularBlobStore"]


# Blob reference marker format
BLOB_REF_PREFIX = "<<FL_BLOB_REF:"
BLOB_REF_SUFFIX = ">>"


class BlobNotFoundError(Exception):
    """Raised when a blob reference cannot be resolved."""

    def __init__(self, blob_id: str, message: str = "") -> None:
        self.blob_id = blob_id
        super().__init__(message or f"Blob not found: {blob_id}")


class BlobRef:
    """Represents a reference to an offloaded blob."""

    def __init__(self, blob_id: str, size: int) -> None:
        self.blob_id = blob_id
        self.size = size

    def to_marker(self) -> str:
        """Convert to inline marker string."""
        return f"{BLOB_REF_PREFIX}{self.blob_id}:size={self.size}{BLOB_REF_SUFFIX}"

    @classmethod
    def from_marker(cls, marker: str) -> Optional["BlobRef"]:
        """Parse a blob reference marker. Returns None if not a valid marker."""
        if not marker.startswith(BLOB_REF_PREFIX) or not marker.endswith(BLOB_REF_SUFFIX):
            return None
        try:
            inner = marker[len(BLOB_REF_PREFIX) : -len(BLOB_REF_SUFFIX)]
            parts = inner.split(":")
            if len(parts) != 2:
                return None
            blob_id = parts[0]
            size_part = parts[1]
            if not size_part.startswith("size="):
                return None
            size = int(size_part[5:])
            return cls(blob_id, size)
        except (ValueError, IndexError):
            return None

    @staticmethod
    def is_marker(value: Any) -> bool:
        """Check if a value is a blob reference marker."""
        if not isinstance(value, str):
            return False
        return value.startswith(BLOB_REF_PREFIX) and value.endswith(BLOB_REF_SUFFIX)


class GranularBlobStore:
    """Durable blob storage for granular execution payloads.

    Uses the existing state backend for persistence, storing blobs
    in a dedicated namespace to avoid conflicts with run state.

    Attributes:
        backend: State backend for persistence
        threshold_bytes: Size threshold for offloading (default 20KB)
        namespace: Storage namespace for blobs
    """

    BLOB_NAMESPACE = "_granular_blobs"

    def __init__(
        self,
        backend: "StateBackend",
        threshold_bytes: int = 20_000,
    ) -> None:
        self._backend = backend
        self._threshold_bytes = threshold_bytes

    def should_offload(self, payload: Any) -> bool:
        """Check if payload exceeds threshold and should be offloaded."""
        try:
            serialized = json.dumps(payload, ensure_ascii=True)
            return len(serialized.encode("utf-8")) > self._threshold_bytes
        except (TypeError, ValueError):
            # Non-JSON-serializable payloads can't be offloaded
            return False

    def generate_blob_id(
        self, run_id: str, step_name: str, turn_index: int, content_hash: str
    ) -> str:
        """Generate a unique blob ID."""
        composite = f"{run_id}:{step_name}:{turn_index}:{content_hash}:{time.time_ns()}"
        return hashlib.sha256(composite.encode("utf-8")).hexdigest()[:32]

    async def offload(
        self,
        run_id: str,
        step_name: str,
        turn_index: int,
        payload: Any,
    ) -> BlobRef:
        """Offload a large payload to durable storage.

        Args:
            run_id: Run identifier
            step_name: Step name
            turn_index: Current turn index
            payload: Data to offload (must be JSON-serializable)

        Returns:
            BlobRef pointing to stored data

        Raises:
            TypeError: If payload is not JSON-serializable
        """
        # Serialize payload
        try:
            serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Payload is not JSON-serializable: {e}") from e

        size = len(serialized.encode("utf-8"))
        content_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
        blob_id = self.generate_blob_id(run_id, step_name, turn_index, content_hash)

        # Store in backend
        storage_key = f"{self.BLOB_NAMESPACE}:{blob_id}"
        blob_data = {
            "blob_id": blob_id,
            "run_id": run_id,
            "step_name": step_name,
            "turn_index": turn_index,
            "size": size,
            "content_hash": content_hash,
            "created_at": time.time(),
            "payload": serialized,
        }

        await self._backend.save_state(storage_key, blob_data)

        return BlobRef(blob_id, size)

    async def hydrate(self, blob_ref: BlobRef) -> Any:
        """Retrieve and deserialize a blob.

        Args:
            blob_ref: Reference to the blob

        Returns:
            Deserialized payload

        Raises:
            BlobNotFoundError: If blob cannot be found
        """
        storage_key = f"{self.BLOB_NAMESPACE}:{blob_ref.blob_id}"

        try:
            blob_data = await self._backend.load_state(storage_key)
        except Exception as exc:
            import logging

            logging.getLogger(__name__).debug("Failed to load blob %s: %s", blob_ref.blob_id, exc)
            blob_data = None

        if blob_data is None:
            raise BlobNotFoundError(
                blob_ref.blob_id,
                f"Blob {blob_ref.blob_id} not found in store. "
                "This may indicate data loss or store unavailability.",
            )

        # Deserialize payload (blob_data should be a dict from our save_state)
        if not isinstance(blob_data, dict):
            raise BlobNotFoundError(
                blob_ref.blob_id,
                f"Blob {blob_ref.blob_id} has invalid format.",
            )
        serialized = blob_data.get("payload")
        if serialized is None:
            raise BlobNotFoundError(
                blob_ref.blob_id,
                f"Blob {blob_ref.blob_id} exists but has no payload.",
            )

        return json.loads(serialized)

    async def hydrate_marker(self, marker: str) -> Any:
        """Parse a marker string and hydrate the referenced blob.

        Args:
            marker: Blob reference marker string

        Returns:
            Deserialized payload

        Raises:
            BlobNotFoundError: If blob cannot be found
            ValueError: If marker is invalid
        """
        ref = BlobRef.from_marker(marker)
        if ref is None:
            raise ValueError(f"Invalid blob marker: {marker}")
        return await self.hydrate(ref)

    def process_for_storage(
        self,
        data: Dict[str, Any],
    ) -> tuple[Dict[str, Any], list[tuple[str, Any]]]:
        """Identify dict fields that exceed the offload threshold.

        This is a synchronous helper that identifies offload candidates.
        Actual offloading must be done separately with async offload().

        Args:
            data: Dictionary to process

        Returns:
            Tuple of (original_data_copy, list_of_offload_candidates)
            where each candidate is (field_path, payload)
        """
        candidates: list[tuple[str, Any]] = []
        result = dict(data)

        for key, value in data.items():
            if self.should_offload(value):
                candidates.append((key, value))

        return result, candidates

    async def process_history_entry(
        self,
        entry: Dict[str, Any],
        run_id: str,
        step_name: str,
        turn_index: int,
    ) -> Dict[str, Any]:
        """Process a history entry, offloading large values.

        Args:
            entry: History entry dict
            run_id: Run identifier
            step_name: Step name
            turn_index: Turn index

        Returns:
            Entry with large values replaced by blob markers
        """
        result = dict(entry)

        for key in ("input", "output"):
            value = entry.get(key)
            if value is not None and self.should_offload(value):
                ref = await self.offload(run_id, step_name, turn_index, value)
                result[key] = ref.to_marker()

        return result

    async def hydrate_history_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Hydrate blob references in a history entry.

        Args:
            entry: History entry possibly containing blob markers

        Returns:
            Entry with blob markers replaced by actual data

        Raises:
            BlobNotFoundError: If any blob cannot be found
        """
        result = dict(entry)

        for key in ("input", "output"):
            value = entry.get(key)
            if isinstance(value, str) and BlobRef.is_marker(value):
                result[key] = await self.hydrate_marker(value)

        return result

    async def cleanup_blobs(
        self,
        run_id: str,
        *,
        older_than_seconds: Optional[float] = None,
    ) -> int:
        """Clean up blobs for a run (best-effort).

        Args:
            run_id: Run to clean up
            older_than_seconds: Only delete blobs older than this

        Returns:
            Number of blobs deleted (approximate)
        """
        # This is a placeholder - actual implementation depends on backend capabilities
        # Many backends don't support prefix listing, so this may be a no-op
        return 0
