from __future__ import annotations
# mypy: disable-error-code=arg-type

from typing import Any, Callable, Coroutine, Dict, cast

from flujo.architect.states.common import goto, make_transition_guard, trace_next_state
from flujo.domain.base_model import BaseModel as _BaseModel
from flujo.domain.dsl import Pipeline, Step


def build_goal_clarification_state() -> Pipeline[Any, Any]:
    """Bridge from GatheringContext to Planning."""

    async def _goto_plan(
        _data: Dict[str, Any] | None = None, context: _BaseModel | None = None
    ) -> Dict[str, Any]:
        return await goto("Planning", context=context)

    goto_plan = Step.from_callable(
        cast(Callable[[Any], Coroutine[Any, Any, Dict[str, Any]]], _goto_plan),
        name="GotoPlanning",
        updates_context=True,
    )
    guard_plan: Step[Any, Any] = Step.from_callable(
        make_transition_guard("Planning"), name="Guard_Planning", updates_context=True
    )
    trace_plan = Step.from_callable(
        trace_next_state, name="TraceNextState_Planning", updates_context=True
    )
    return Pipeline.from_step(goto_plan) >> guard_plan >> trace_plan
