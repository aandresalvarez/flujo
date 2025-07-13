from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, cast

from .base import StateBackend
from ...utils.serialization import serialize_to_json, safe_deserialize


class FileBackend(StateBackend):
    """Persist workflow state to JSON files."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

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

    async def save_state(self, run_id: str, state: Dict[str, Any]) -> None:
        file_path = self._resolve_path(run_id)
        # Use proper serialization that fails fast on non-serializable objects
        data = serialize_to_json(state)
        async with self._lock:
            await asyncio.to_thread(self._atomic_write, file_path, data.encode())

    def _atomic_write(self, file_path: Path, data: bytes) -> None:
        tmp = file_path.with_suffix(file_path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, file_path)

    async def load_state(self, run_id: str) -> Optional[Dict[str, Any]]:
        file_path = self._resolve_path(run_id)
        async with self._lock:
            if not file_path.exists():
                return None
            return await asyncio.to_thread(self._read_json, file_path)

    def _read_json(self, file_path: Path) -> Dict[str, Any]:
        with open(file_path, "rb") as f:
            data = json.loads(f.read().decode())
        # Apply safe_deserialize to restore custom types
        return cast(Dict[str, Any], safe_deserialize(data))

    async def delete_state(self, run_id: str) -> None:
        file_path = self._resolve_path(run_id)
        async with self._lock:
            if file_path.exists():
                await asyncio.to_thread(file_path.unlink)
