from __future__ import annotations
from typing import Any, Optional, Callable
from flujo.domain.models import StepResult, UsageLimits
# Removed ExecutorCore import to break circular dependency


async def _handle_loop_step(
    self,
    step: Any,
    data: Any,
    context: Optional[Any],
    resources: Optional[Any],
    limits: Optional[UsageLimits],
    context_setter: Optional[Callable[[Any, Optional[Any]], None]],
    _fallback_depth: int = 0,
) -> StepResult:
    """Delegate to the unified loop helper from first principles."""
    return await self._execute_loop(
        step,
        data,
        context,
        resources,
        limits,
        context_setter,
        _fallback_depth,
    )
