from __future__ import annotations

import os
from typing import Any

from ...domain.types import HookCallable
from ...infra.console_tracer import ConsoleTracer
from ...tracing.manager import TraceManager, set_active_trace_manager

_TRUTHY = {"1", "true", "on", "yes"}


class TracingManager:
    """Manage tracing lifecycle and hook wiring for runner executions."""

    def __init__(
        self,
        *,
        enable_tracing: bool = True,
        local_tracer: Any | None = None,
    ) -> None:
        self._enable_tracing = enable_tracing
        self._local_tracer = local_tracer
        self._trace_manager: TraceManager | None = None
        self._hooks: list[HookCallable] = []

    @property
    def trace_manager(self) -> TraceManager | None:
        return self._trace_manager

    @property
    def hooks(self) -> list[HookCallable]:
        return list(self._hooks)

    def setup(self, hooks: list[HookCallable]) -> list[HookCallable]:
        """Attach tracing and console hooks, returning the updated hook list."""
        self._hooks = list(hooks)
        tracing_enabled = self._enable_tracing and not self._disabled_via_env()

        if tracing_enabled:
            try:
                self._trace_manager = TraceManager()
                try:
                    set_active_trace_manager(self._trace_manager)
                except Exception:
                    pass
                self._hooks.append(self._trace_manager.hook)
            except Exception:
                self._trace_manager = None
        else:
            # Ensure any previously active manager is cleared when tracing is disabled
            try:
                set_active_trace_manager(None)
            except Exception:
                pass

        self._attach_local_tracer()
        return list(self._hooks)

    def disable(self, hooks: list[HookCallable] | None = None) -> list[HookCallable]:
        """Disable tracing and detach the trace hook from the provided list."""
        active_hooks = list(hooks) if hooks is not None else list(self._hooks)
        trace_hook = getattr(self._trace_manager, "hook", None)
        if trace_hook is not None:
            active_hooks = [hook for hook in active_hooks if hook is not trace_hook]
        self._hooks = active_hooks
        self.teardown()
        return list(self._hooks)

    def teardown(self) -> None:
        """Clear active trace manager references."""
        if self._trace_manager is None:
            return
        try:
            set_active_trace_manager(None)
        except Exception:
            pass
        self._trace_manager = None

    def add_event(self, name: str, attributes: dict[str, Any]) -> None:
        """Best-effort event emission to the active trace manager."""
        if self._trace_manager is None:
            return
        try:
            self._trace_manager.add_event(name, attributes)
        except Exception:
            pass

    @property
    def root_span(self) -> Any | None:
        trace_manager = self._trace_manager
        if trace_manager is None:
            return None
        return getattr(trace_manager, "_root_span", None)

    def _attach_local_tracer(self) -> None:
        tracer_instance: ConsoleTracer | None
        tracer_instance = None
        if isinstance(self._local_tracer, ConsoleTracer):
            tracer_instance = self._local_tracer
        elif self._local_tracer == "default":
            tracer_instance = ConsoleTracer()

        if tracer_instance is not None:
            self._hooks.append(tracer_instance.hook)

    def _disabled_via_env(self) -> bool:
        try:
            return str(os.getenv("FLUJO_DISABLE_TRACING", "")).strip().lower() in _TRUTHY
        except Exception:
            return False
