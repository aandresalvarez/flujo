from __future__ import annotations
from typing import Callable

from flujo.domain.models import StepResult, UsageLimits
# Removed ExecutorCore import to break circular dependency


async def _handle_loop_step(
    self: object,
    step: object,
    data: object,
    context: object | None,
    resources: object | None,
    limits: UsageLimits | None,
    context_setter: Callable[..., None] | None,
    _fallback_depth: int = 0,
) -> StepResult:
    """Delegate to the unified loop helper from first principles."""
    execute_loop = getattr(self, "_execute_loop", None)
    if not callable(execute_loop):
        raise TypeError("Loop executor requires _execute_loop()")
    return await execute_loop(
        step,
        data,
        context,
        resources,
        limits,
        context_setter,
        _fallback_depth,
    )
