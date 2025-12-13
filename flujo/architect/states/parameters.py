from __future__ import annotations
# mypy: disable-error-code=arg-type

from typing import Any

from flujo.type_definitions.common import JSONObject

from flujo.architect.states.common import (
    goto,
    make_transition_guard,
    skill_resolver,
    trace_next_state,
)
from flujo.domain.base_model import BaseModel as _BaseModel
from flujo.domain.dsl import Pipeline, Step


async def _collect_params(
    _data: JSONObject | None = None, context: _BaseModel | None = None
) -> JSONObject:
    try:
        non_interactive = bool(getattr(context, "non_interactive", False)) if context else True
    except Exception:
        non_interactive = True
    try:
        plan = getattr(context, "execution_plan", None) if context is not None else None
    except Exception:
        plan = None
    if not isinstance(plan, list):
        return await goto("Generation", context=context)
    reg = skill_resolver()
    changed = False
    for step in plan:
        try:
            agent = step.get("agent") if isinstance(step, dict) else None
            if not isinstance(agent, dict):
                continue
            sid = agent.get("id")
            params = agent.get("params")
            if not isinstance(params, dict):
                params = {}
                agent["params"] = params
            if not isinstance(sid, str):
                continue
            entry = reg.get(sid) or {}
            schema = entry.get("input_schema") or {}
            required = schema.get("required") if isinstance(schema, dict) else None
            req_list = list(required) if isinstance(required, list) else []
            missing = [k for k in req_list if k not in params]
            if not missing:
                continue
            if non_interactive:
                continue
            try:
                import typer as _typer

                for key in missing:
                    val = _typer.prompt(
                        f"Enter value for required parameter '{key}' of skill '{sid}':"
                    )
                    params[key] = val
                    changed = True
            except Exception:
                continue
        except Exception:
            continue
    out: JSONObject = {"execution_plan": plan} if changed else {}
    nxt = await goto("Generation", context=context)
    out.update(nxt)
    return out


def build_parameter_collection_state() -> Pipeline[Any, Any]:
    """Prompt for missing required parameters when interactive."""
    guard_generation: Step[Any, Any] = Step.from_callable(
        make_transition_guard("Generation"), name="Guard_Generation", updates_context=True
    )
    trace_generation: Step[Any, Any] = Step.from_callable(
        trace_next_state, name="TraceNextState_Generation", updates_context=True
    )
    return (
        Pipeline.from_step(
            Step.from_callable(
                _collect_params,
                name="CollectParams",
                updates_context=True,
            )
        )
        >> guard_generation
        >> trace_generation
    )
