from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, cast

from .base import StateBackend
from ...utils.serialization import safe_serialize


class FileBackend(StateBackend):
    """Persist workflow state to JSON files."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def save_state(self, run_id: str, state: Dict[str, Any]) -> None:
        file_path = self.path / f"{run_id}.json"
        # Use enhanced serialization for any Pydantic models in the state
        serialized_state = {k: safe_serialize(v) for k, v in state.items()}
        data = json.dumps(serialized_state, default=str)
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
        file_path = self.path / f"{run_id}.json"
        async with self._lock:
            if not file_path.exists():
                return None
            return await asyncio.to_thread(self._read_json, file_path)

    def _read_json(self, file_path: Path) -> Dict[str, Any]:
        with open(file_path, "rb") as f:
            data = json.loads(f.read().decode())
        return cast(Dict[str, Any], data)

    async def delete_state(self, run_id: str) -> None:
        file_path = self.path / f"{run_id}.json"
        async with self._lock:
            if file_path.exists():
                await asyncio.to_thread(file_path.unlink)
