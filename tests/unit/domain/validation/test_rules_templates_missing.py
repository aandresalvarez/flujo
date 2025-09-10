from __future__ import annotations

from typing import Any, Dict
from pydantic import BaseModel

from flujo.domain.dsl import Pipeline, Step


class Out(BaseModel):
    foo: int


def test_v_t5_prior_model_field_existence() -> None:
    async def a(_: str) -> Out:  # type: ignore[override]
        return Out(foo=1)

    s1 = Step.from_callable(a, name="first")

    async def echo(x: str) -> str:  # type: ignore[override]
        return x

    s2 = Step.from_callable(echo, name="second")
    s2.meta["templated_input"] = "{{ previous_step.bar }}"  # missing field
    print("DEBUG meta:", s2.meta)
    report = (Pipeline.from_step(s1) >> s2).validate_graph()
    report = (Pipeline.from_step(s1) >> s2).validate_graph()
    warns = [w for w in report.warnings if w.rule_id == "V-T5" and w.step_name == "second"]
    assert warns, report.model_dump()


def test_v_t6_json_trap() -> None:
    async def b(x: Dict[str, Any]) -> str:  # type: ignore[override]
        return "ok"

    async def pass_through(x: str) -> str:  # type: ignore[override]
        return x

    s1 = Step.from_callable(pass_through, name="first")
    s1.meta["templated_input"] = "{}"
    s2 = Step.from_callable(b, name="second")
    # Non-JSON literal attempt (single quotes/identifiers) in template
    s2.meta["templated_input"] = "{ not_json: yes }"
    report = (Pipeline.from_step(s1) >> s2).validate_graph()
    warns = [w for w in report.warnings if w.rule_id == "V-T6" and w.step_name == "second"]
    assert warns, report.model_dump()
