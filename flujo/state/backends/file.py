from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

from .base import StateBackend
from flujo.type_definitions.common import JSONObject
from ...utils.serialization import (
    safe_deserialize,
    _serialize_to_json_internal as serialize_to_json,
)


class FileBackend(StateBackend):
    """Persist workflow state to JSON files."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        """Lazily create the lock on first access."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _resolve_path(self, run_id: str) -> Path:
        """Return an absolute path for the given ``run_id`` within ``self.path``.

        Raises ``ValueError`` if the resolved path would escape the configured
        directory. This guards against path traversal attempts.
        """
        if any(sep in run_id for sep in (os.sep, os.altsep) if sep):
            raise ValueError(f"Invalid run_id: {run_id!r}")
        if ".." in Path(run_id).parts:
            raise ValueError(f"Invalid run_id: {run_id!r}")
        candidate = (self.path / f"{run_id}.json").resolve()
        base = self.path.resolve()
        if not candidate.is_relative_to(base):
            raise ValueError(f"Invalid run_id: {run_id!r}")
        return candidate

    async def save_state(self, run_id: str, state: JSONObject) -> None:
        file_path = self._resolve_path(run_id)
        # Use proper serialization that fails fast on non-serializable objects
        data = serialize_to_json(state)
        async with self._get_lock():
            await asyncio.to_thread(self._atomic_write, file_path, data.encode())

    def _atomic_write(self, file_path: Path, data: bytes) -> None:
        tmp = file_path.with_suffix(file_path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, file_path)

    async def load_state(self, run_id: str) -> Optional[JSONObject]:
        file_path = self._resolve_path(run_id)
        async with self._get_lock():
            if not file_path.exists():
                return None
            return await asyncio.to_thread(self._read_json, file_path)

    def _read_json(self, file_path: Path) -> JSONObject:
        with open(file_path, "rb") as f:
            data = json.loads(f.read().decode())
        # Apply safe_deserialize to restore custom types
        restored = safe_deserialize(data)
        if isinstance(restored, dict):
            return restored
        raise ValueError(
            "Invalid state payload in file backend; expected JSON object, "
            f"got {type(restored).__name__}"
        )

    async def delete_state(self, run_id: str) -> None:
        file_path = self._resolve_path(run_id)
        async with self._get_lock():
            if file_path.exists():
                await asyncio.to_thread(file_path.unlink)

    async def get_trace(self, run_id: str) -> Optional[JSONObject]:
        """Retrieve trace data for a given run_id."""
        # For FileBackend, traces are stored as part of the state
        # We'll return None as FileBackend doesn't implement separate trace storage
        return None

    async def save_trace(self, run_id: str, trace: JSONObject) -> None:
        """Save trace data for a given run_id."""
        # FileBackend doesn't support separate trace storage
        # Traces would need to be integrated into the main state if needed
        raise NotImplementedError("FileBackend doesn't support separate trace storage")

    async def get_spans(
        self, run_id: str, status: Optional[str] = None, name: Optional[str] = None
    ) -> List[JSONObject]:
        """Get individual spans with optional filtering."""
        # FileBackend doesn't support normalized span storage
        return []

    async def get_span_statistics(
        self,
        pipeline_name: Optional[str] = None,
        time_range: Optional[Tuple[float, float]] = None,
    ) -> JSONObject:
        """Get aggregated span statistics."""
        # FileBackend doesn't support span statistics
        return {
            "total_spans": 0,
            "by_name": {},
            "by_status": {},
            "avg_duration_by_name": {},
        }
