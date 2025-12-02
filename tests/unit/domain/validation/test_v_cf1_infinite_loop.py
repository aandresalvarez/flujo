from __future__ import annotations

from flujo.domain.dsl.step import Step
from flujo.domain.dsl.pipeline import Pipeline
from typing import Optional
from flujo.domain.models import BaseModel


def _const_false(_out, _ctx) -> bool:  # type: ignore[no-untyped-def]
    return False


def test_v_cf1_triggers_on_constant_false_exit_condition() -> None:
    # Build a trivial body pipeline
    async def _noop(x: int, *, context: Optional[BaseModel] = None) -> int:
        return x

    body = Pipeline.from_step(Step.from_callable(_noop, name="noop"))
    loop = Step.loop_until(
        name="loop",
        loop_body_pipeline=body,
        exit_condition_callable=_const_false,
        max_loops=10,
    )
    pipe = Pipeline.from_step(loop)
    report = pipe.validate_graph()
    assert any(e.rule_id == "V-CF1" for e in report.errors)


def test_v_cf1_triggers_on_excessive_max_loops() -> None:
    async def _noop(x: int, *, context: Optional[BaseModel] = None) -> int:
        return x

    body = Pipeline.from_step(Step.from_callable(_noop, name="noop"))
    # Provide a sensible exit to ensure only the max_loops heuristic fires
    loop = Step.loop_until(
        name="loop",
        loop_body_pipeline=body,
        exit_condition_callable=lambda out, ctx: True,  # terminate on first
        max_loops=2000,
    )
    pipe = Pipeline.from_step(loop)
    report = pipe.validate_graph()
    assert any(e.rule_id == "V-CF1" for e in report.errors)
