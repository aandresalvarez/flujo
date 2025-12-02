from __future__ import annotations

import os
from typing import Any, Optional


from ..infra.console_tracer import ConsoleTracer

__all__ = ["attach_local_tracer", "setup_tracing"]


def attach_local_tracer(local_tracer: Optional[Any], hooks: list[Any]) -> None:
    """Attach a ConsoleTracer hook based on the provided hint.

    This keeps CLI-facing tracer bootstrapping out of the runner core.
    """
    tracer_instance: Optional[ConsoleTracer] = None
    if isinstance(local_tracer, ConsoleTracer):
        tracer_instance = local_tracer
    elif local_tracer == "default":
        tracer_instance = ConsoleTracer()

    if tracer_instance is not None:
        hooks.append(tracer_instance.hook)


def _bool_env(name: str) -> bool:
    try:
        return str(os.getenv(name, "")).strip().lower() in {"1", "true", "on", "yes"}
    except Exception:
        return False


def setup_tracing(
    *,
    enable_tracing: bool,
    local_tracer: Optional[Any],
    hooks: list[Any],
) -> Optional[Any]:
    """Initialize tracing concerns outside the runner core.

    Returns the TraceManager (or None when disabled) while appending any hook
    handlers to the provided ``hooks`` list.
    """
    # Honor env override to disable tracing regardless of caller preference.
    try:
        if _bool_env("FLUJO_DISABLE_TRACING"):
            enable_tracing = False
    except Exception:
        pass

    trace_manager: Optional[Any] = None
    if enable_tracing:
        try:
            from flujo.tracing.manager import TraceManager, set_active_trace_manager

            trace_manager = TraceManager()
            try:
                set_active_trace_manager(trace_manager)
            except Exception:
                pass
            hooks.insert(0, trace_manager.hook)
        except Exception:
            trace_manager = None

    # Always attach console tracers based on caller hint
    attach_local_tracer(local_tracer, hooks)
    return trace_manager
