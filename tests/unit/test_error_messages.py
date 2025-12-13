import os
from typing import Optional
import pytest

from flujo import Step, Pipeline, step
from flujo.exceptions import (
    ImproperStepInvocationError,
    StepInvocationError,  # Add the new error class
    MissingAgentError,
    TypeMismatchError,
    ConfigurationError,
)
from tests.conftest import create_test_flujo


@step
async def echo(x: str) -> str:
    return x


def test_improper_step_call() -> None:
    with pytest.raises(ImproperStepInvocationError):
        echo("hi")
    # StepInvocationError is raised when a step is invoked improperly via its internal 'run' method, ensuring stricter validation compared to ImproperStepInvocationError
    with pytest.raises(StepInvocationError):
        getattr(echo, "run")("hi")


def test_missing_agent_errors() -> None:
    blank = Step.model_validate({"name": "blank"})
    pipeline = Pipeline.from_step(blank)
    report = pipeline.validate_graph()
    assert not report.is_valid
    assert any(f.rule_id == "V-A1" for f in report.errors)
    with pytest.raises(ConfigurationError):
        pipeline.validate_graph(raise_on_error=True)
    runner = create_test_flujo(blank)
    with pytest.raises(MissingAgentError):
        runner.run(None)


@step
async def make_int(x: str) -> int:
    return len(x)


@step
async def need_str(x: str) -> str:
    return x


def test_type_mismatch_errors() -> None:
    prev_strict = os.environ.get("FLUJO_STRICT_DSL")
    os.environ["FLUJO_STRICT_DSL"] = "0"
    try:
        pipeline = Pipeline.model_construct(steps=[make_int, need_str])
    finally:
        if prev_strict is None:
            os.environ.pop("FLUJO_STRICT_DSL", None)
        else:
            os.environ["FLUJO_STRICT_DSL"] = prev_strict
    report = pipeline.validate_graph()
    assert not report.is_valid
    assert any(f.rule_id == "V-A2" for f in report.errors)
    with pytest.raises(ConfigurationError):
        pipeline.validate_graph(raise_on_error=True)
    runner = create_test_flujo(pipeline)
    with pytest.raises(TypeMismatchError):
        runner.run("abc")


@step
async def maybe_str(x: str) -> Optional[str]:
    return x if x else None


@step
async def expect_optional(x: Optional[str]) -> str:
    return x or ""


def test_union_optional_handling() -> None:
    ok_pipeline = echo >> expect_optional
    expect_optional.__step_input_type__ = Optional[str]
    report_ok = ok_pipeline.validate_graph()
    assert report_ok.is_valid
    runner_ok = create_test_flujo(ok_pipeline)
    result_ok = runner_ok.run("hi")
    assert result_ok.step_history[-1].output == "hi"

    bad_pipeline = maybe_str >> need_str
    report_bad = bad_pipeline.validate_graph()
    assert not report_bad.is_valid
    with pytest.raises(ConfigurationError):
        bad_pipeline.validate_graph(raise_on_error=True)
    runner_bad = create_test_flujo(bad_pipeline)
    with pytest.raises(TypeMismatchError):
        runner_bad.run("")
