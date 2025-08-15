from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from pydantic import BaseModel

from flujo.builtins import extract_decomposed_steps


class _DecomposerModel(BaseModel):
    steps: List[Dict[str, Any]]


def test_extract_decomposed_steps_from_model() -> None:
    model = _DecomposerModel(steps=[{"step_name": "a"}, {"step_name": "b"}])
    out = asyncio.run(extract_decomposed_steps(model))
    assert isinstance(out, dict)
    assert "prepared_steps_for_mapping" in out
    assert isinstance(out["prepared_steps_for_mapping"], list)
    assert out["prepared_steps_for_mapping"][0]["step_name"] == "a"


def test_extract_decomposed_steps_from_dict() -> None:
    payload = {"steps": [{"step_name": "x"}]}
    out = asyncio.run(extract_decomposed_steps(payload))
    assert out["prepared_steps_for_mapping"][0]["step_name"] == "x"


async def _aggregate_plan(mapped, *, context=None):
    from flujo.builtins import _register_builtins  # ensure registration

    _register_builtins()
    # Access the registered factory and call the returned coroutine function directly
    from flujo.infra.skill_registry import get_skill_registry

    reg = get_skill_registry()
    factory = reg.get("flujo.builtins.aggregate_plan")["factory"]
    agg = factory() if callable(factory) else factory
    return await agg(mapped, context=context)


def test_aggregate_plan_combines_goal_and_steps() -> None:
    class Ctx(BaseModel):
        initial_prompt: str
        user_goal: str

    ctx = Ctx(initial_prompt="demo", user_goal="make a plan")
    mapped = [{"step_name": "a"}, {"step_name": "b"}]
    out = asyncio.run(_aggregate_plan(mapped, context=ctx))
    assert out["user_goal"] == "make a plan"
    assert [s["step_name"] for s in out["step_plans"]] == ["a", "b"]
