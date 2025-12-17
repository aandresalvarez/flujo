from __future__ import annotations

from ._shared import (  # noqa: F401
    Awaitable,
    Callable,
    PausedException,
    StepOutcome,
    StepResult,
    UsageLimits,
    to_outcome,
)
from ....domain.models import BaseModel as DomainBaseModel


async def run_agent_execution(
    executor: object,
    core: object,
    step: object,
    data: object,
    context: DomainBaseModel | None,
    resources: object | None,
    limits: UsageLimits | None,
    stream: bool,
    on_chunk: Callable[[object], Awaitable[None]] | None,
    cache_key: str | None,
    _fallback_depth: int = 0,
) -> StepOutcome[StepResult]:
    handler = getattr(core, "_agent_handler", None)
    execute_fn = getattr(handler, "execute", None)
    if not callable(execute_fn):
        raise TypeError("run_agent_execution requires core._agent_handler.execute(...)")
    try:
        outcome = await execute_fn(
            step=step,
            data=data,
            context=context,
            resources=resources,
            limits=limits,
            stream=stream,
            on_chunk=on_chunk,
            cache_key=cache_key,
            fallback_depth=_fallback_depth,
        )
    except PausedException as e:
        raise e
    if isinstance(outcome, StepOutcome):
        return outcome
    if isinstance(outcome, StepResult):
        return to_outcome(outcome)
    return to_outcome(
        StepResult(
            name=str(getattr(step, "name", "<unnamed>")),
            success=False,
            feedback=f"Unsupported agent handler outcome: {type(outcome).__name__}",
        )
    )
