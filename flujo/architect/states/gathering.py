from __future__ import annotations
# mypy: disable-error-code=arg-type

from typing import Any, Callable, Coroutine, Dict, Union, cast

from flujo.architect.states.common import (
    goto,
    make_transition_guard,
    skill_resolver,
    trace_next_state,
)
from flujo.domain.base_model import BaseModel as _BaseModel
from flujo.domain.dsl import Pipeline, Step


def build_gathering_state() -> Pipeline[Any, Any]:
    """Gather available skills, analyze the project, and move to GoalClarification."""
    reg = skill_resolver()
    # discover_skills is an agent; factory returns an agent instance
    discover: Step[Any, Any]
    try:
        discover_entry = reg.get("flujo.builtins.discover_skills")
        if discover_entry and isinstance(discover_entry, dict):
            _discover_agent = discover_entry["factory"]()
            discover = Step.from_callable(
                _discover_agent, name="DiscoverSkills", updates_context=True
            )
        else:

            async def _discover_fallback(*_a: Any, **_k: Any) -> Dict[str, Any]:
                return {"available_skills": []}

            discover = Step.from_callable(
                _discover_fallback, name="DiscoverSkills", updates_context=True
            )
    except Exception:

        async def _discover_fallback(*_a: Any, **_k: Any) -> Dict[str, Any]:
            return {"available_skills": []}

        discover = Step.from_callable(
            _discover_fallback, name="DiscoverSkills", updates_context=True
        )

    try:
        analyze_entry = reg.get("flujo.builtins.analyze_project")
        if analyze_entry and isinstance(analyze_entry, dict):
            _analyze = analyze_entry["factory"]()
            analyze: Step[Any, Any] = Step.from_callable(
                _analyze, name="AnalyzeProject", updates_context=True
            )
        else:

            async def _analyze_fallback(*_a: Any, **_k: Any) -> Dict[str, Any]:
                return {"project_summary": ""}

            analyze = Step.from_callable(
                _analyze_fallback, name="AnalyzeProject", updates_context=True
            )
    except Exception:

        async def _analyze_fallback(*_a: Any, **_k: Any) -> Dict[str, Any]:
            return {"project_summary": ""}

        analyze = Step.from_callable(_analyze_fallback, name="AnalyzeProject", updates_context=True)

    async def _schema_fallback(*_a: Any, **_k: Any) -> Dict[str, Any]:
        return {"flujo_schema": {}}

    get_schema: Union[Step[Any, Any], Pipeline[Any, Any]] = Step.from_callable(
        _schema_fallback, name="MapFrameworkSchema", updates_context=True
    )

    async def _goto_goal(
        _data: Dict[str, Any] | None = None, context: _BaseModel | None = None
    ) -> Dict[str, Any]:
        return await goto("GoalClarification", context=context)

    goto_goal = Step.from_callable(
        cast(Callable[[Any], Coroutine[Any, Any, Dict[str, Any]]], _goto_goal),
        name="GotoGoalClarification",
        updates_context=True,
    )
    guard_gc: Step[Any, Any] = Step.from_callable(
        make_transition_guard("GoalClarification"),
        name="Guard_GoalClarification",
        updates_context=True,
    )
    trace_gc = Step.from_callable(trace_next_state, name="TraceNextState_GC", updates_context=True)
    return discover >> analyze >> get_schema >> goto_goal >> guard_gc >> trace_gc
