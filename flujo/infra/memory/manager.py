from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable, Sequence

from ...domain.memory import MemoryRecord, VectorQuery, VectorStoreProtocol, ScoredMemory
from ...infra import telemetry


EmbeddingFn = Callable[[list[str]], Awaitable[list[list[float]]]]


def _to_text(payload: Any) -> str | None:
    """Best-effort text extraction from step outputs."""
    if payload is None:
        return None
    if isinstance(payload, str):
        return payload
    if isinstance(payload, (int, float, bool)):
        return str(payload)
    try:
        return json.dumps(payload, ensure_ascii=False)
    except Exception:
        return str(payload)


class MemoryManager:
    """Indexes step outputs into a vector store via an embedding function."""

    def __init__(
        self,
        *,
        store: VectorStoreProtocol,
        embedder: EmbeddingFn | None,
        enabled: bool = False,
        background_task_manager: Any | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._enabled = enabled and embedder is not None
        self._bgm = background_task_manager

    async def index_step_output(
        self, *, step_name: str, result: Any, context: Any | None = None
    ) -> None:
        """Index successful step outputs when enabled."""
        if not self._enabled:
            return
        try:
            success = bool(getattr(result, "success", False))
        except Exception:
            success = False
        if not success:
            return

        text = _to_text(getattr(result, "output", None))
        if text is None or text.strip() == "":
            return

        metadata: dict[str, Any] = {"step": step_name}
        try:
            run_id = getattr(context, "run_id", None) if context is not None else None
            if run_id:
                metadata["run_id"] = run_id
        except Exception:
            pass

        async def _run() -> None:
            try:
                embedder = self._embedder
                if embedder is None:
                    return
                vectors = await embedder([text])
                if not vectors:
                    return
                record = MemoryRecord(
                    vector=vectors[0], payload=getattr(result, "output", None), metadata=metadata
                )
                await self._store.add([record])
            except Exception as exc:  # pragma: no cover - defensive logging
                try:
                    telemetry.logfire.warning("Memory indexing failed", extra={"error": str(exc)})
                except Exception:
                    pass

        if self._bgm is not None:
            task = asyncio.create_task(_run(), name=f"flujo_memory_index_{step_name}")
            try:
                self._bgm.add_task(task)
            except Exception:
                pass
            return
        await _run()


class NullMemoryManager(MemoryManager):
    """No-op manager used when indexing is disabled."""

    def __init__(self) -> None:
        super().__init__(
            store=_NullStore(),
            embedder=None,
            enabled=False,
            background_task_manager=None,
        )

    async def index_step_output(
        self, *, step_name: str, result: Any, context: Any | None = None
    ) -> None:
        return None


class _NullStore(VectorStoreProtocol):
    async def add(self, records: Sequence[MemoryRecord]) -> None:  # pragma: no cover - trivial
        return None

    async def query(self, query: VectorQuery) -> list[ScoredMemory]:  # pragma: no cover - trivial
        return []

    async def delete(self, ids: Sequence[str]) -> None:  # pragma: no cover - trivial
        return None

    async def close(self) -> None:  # pragma: no cover - trivial
        return None
