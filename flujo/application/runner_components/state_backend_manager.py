from __future__ import annotations

import inspect
from pathlib import Path
from typing import Optional

from ...state import StateBackend
from ...state.backends.memory import InMemoryBackend
from ...state.backends.sqlite import SQLiteBackend
from ...utils.config import get_settings


class StateBackendManager:
    """Lifecycle management for runner state backends."""

    def __init__(
        self,
        state_backend: Optional[StateBackend] = None,
        *,
        delete_on_completion: bool = False,
        enable_backend: bool = True,
    ) -> None:
        self._enabled = enable_backend
        self._owns_backend = state_backend is None
        self._delete_on_completion = delete_on_completion
        if not enable_backend:
            self._backend: Optional[StateBackend] = None
            self._owns_backend = False
        elif state_backend is not None:
            self._backend = state_backend
        else:
            self._backend = self._create_default_backend()

    @property
    def backend(self) -> Optional[StateBackend]:
        return self._backend

    async def shutdown(self) -> None:
        """Shutdown the backend if this manager owns it."""
        if not self._enabled or self._backend is None:
            return
        if not self._owns_backend:
            return
        shutdown_fn = getattr(self._backend, "shutdown", None)
        if shutdown_fn is None or not callable(shutdown_fn):
            return
        try:
            result = shutdown_fn()
            if inspect.isawaitable(result):
                await result
        except Exception:
            pass

    async def delete_state(self, run_id: str) -> None:
        """Delete state for a completed run when configured to do so."""
        if not self._delete_on_completion or self._backend is None:
            return
        delete_fn = getattr(self._backend, "delete_state", None)
        if delete_fn is None or not callable(delete_fn):
            return
        try:
            result = delete_fn(run_id)
            if inspect.isawaitable(result):
                await result
        except Exception:
            pass

    def _create_default_backend(self) -> StateBackend:
        """Create the default backend based on settings."""
        try:
            if get_settings().test_mode:
                return InMemoryBackend()
        except Exception:
            pass
        try:
            db_path = Path.cwd() / "flujo_ops.db"
            return SQLiteBackend(db_path)
        except Exception:
            # Final fallback to in-memory to avoid crashing on initialization
            return InMemoryBackend()
