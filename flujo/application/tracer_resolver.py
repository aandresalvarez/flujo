from __future__ import annotations

from typing import Any

from ..infra.console_tracer import ConsoleTracer

__all__ = ["attach_local_tracer"]


def attach_local_tracer(local_tracer: Any | None, hooks: list[Any]) -> None:
    """Attach a ConsoleTracer hook based on the provided hint.

    This keeps CLI-facing tracer bootstrapping out of the runner core.
    """
    tracer_instance: ConsoleTracer | None = None
    if isinstance(local_tracer, ConsoleTracer):
        tracer_instance = local_tracer
    elif local_tracer == "default":
        tracer_instance = ConsoleTracer()

    if tracer_instance is not None:
        hooks.append(tracer_instance.hook)
