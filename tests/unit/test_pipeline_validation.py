import os
from typing import Any

from flujo import Pipeline, step
from flujo.domain.models import BaseModel
from pydantic import BaseModel as PydBaseModel
from flujo.domain.dsl.step import Step
from flujo.domain.blueprint.loader import _finalize_step_types
from flujo.type_definitions.common import JSONObject


class Ctx(BaseModel):
    num: int = 0


@step
async def deprecated_step(x: int, *, context: Ctx) -> int:
    return x


@step
async def modern_step(x: int, *, context: Ctx) -> int:
    return x


def test_no_warning_for_context() -> None:
    pipeline = Pipeline.from_step(deprecated_step)
    report = pipeline.validate_graph()
    assert not any(f.rule_id == "V-A4" for f in report.warnings)


def test_no_warning_for_modern_context() -> None:
    pipeline = Pipeline.from_step(modern_step)
    report = pipeline.validate_graph()
    assert not any(f.rule_id == "V-A4" for f in report.warnings)


class MyOutModel(PydBaseModel):
    value: int


async def _consume_dict(d: JSONObject) -> str:
    return "ok"


async def _consume_str(s: str) -> str:
    return s


def test_pydantic_output_bridges_to_dict_input() -> None:
    # Previous step produces a Pydantic model type
    s1 = Step[Any, Any](name="s1", agent=object())
    s1.__step_output_type__ = MyOutModel

    # Next step expects a dict[str, Any]
    s2 = Step.from_callable(_consume_dict, name="s2")

    prev_strict = os.environ.get("FLUJO_STRICT_DSL")
    os.environ["FLUJO_STRICT_DSL"] = "0"
    try:
        p = Pipeline.model_construct(steps=[s1, s2])
    finally:
        if prev_strict is None:
            os.environ.pop("FLUJO_STRICT_DSL", None)
        else:
            os.environ["FLUJO_STRICT_DSL"] = prev_strict
    report = p.validate_graph()
    # Strict mode: Pydantic -> dict is not allowed without an explicit adapter.
    assert any(e.rule_id == "V-A2-TYPE" for e in report.errors), (
        "Expected V-A2-TYPE error when piping a Pydantic model output into a dict input "
        "without an adapter."
    )


def test_templated_input_tojson_skips_type_mismatch() -> None:
    # Previous step produces a Pydantic model type (not a str)
    s1 = Step[Any, Any](name="a", agent=object())
    s1.__step_output_type__ = MyOutModel

    # Next step expects a string, but uses templated input with | tojson
    s2 = Step.from_callable(_consume_str, name="b")
    s2.meta["templated_input"] = "{{ steps.a.output | tojson }}"

    p = Pipeline.model_construct(steps=[s1, s2])
    report = p.validate_graph()
    assert not any(e.rule_id == "V-A2" for e in report.errors)


def test_finalize_types_uses_wrapper_target_output_type() -> None:
    class FakeWrapper:
        def __init__(self, out_t: Any) -> None:
            self.target_output_type = out_t

    step_obj = Step[Any, Any](name="wrapped", agent=FakeWrapper(MyOutModel))
    # Default type before finalize
    assert step_obj.__step_output_type__ is object

    _finalize_step_types(step_obj)

    assert step_obj.__step_output_type__ is MyOutModel, (
        "Expected finalize to copy target_output_type to step output type"
    )
