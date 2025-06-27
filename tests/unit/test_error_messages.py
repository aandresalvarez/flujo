import pytest

from flujo import Step, Pipeline, Flujo, step
from flujo.exceptions import (
    ImproperStepInvocationError,
    MissingAgentError,
    TypeMismatchError,
)


@step
async def echo(x: str) -> str:
    return x


def test_improper_step_call() -> None:
    with pytest.raises(ImproperStepInvocationError):
        echo("hi")
    with pytest.raises(ImproperStepInvocationError):
        echo.run("hi")  # type: ignore[attr-defined]


def test_missing_agent_errors() -> None:
    blank = Step("blank")
    pipeline = Pipeline.from_step(blank)
    with pytest.raises(MissingAgentError):
        pipeline.validate()
    runner = Flujo(blank)
    with pytest.raises(MissingAgentError):
        runner.run(None)


@step
async def make_int(x: str) -> int:
    return len(x)


@step
async def need_str(x: str) -> str:
    return x


def test_type_mismatch_errors() -> None:
    pipeline = make_int >> need_str
    with pytest.raises(TypeMismatchError):
        pipeline.validate()
    runner = Flujo(pipeline)
    with pytest.raises(TypeMismatchError):
        runner.run("abc")


@step
async def maybe_str(x: str) -> str | None:
    return x if x else None


@step
async def expect_optional(x: str | None) -> str:
    return x or ""


def test_union_optional_handling() -> None:
    ok_pipeline = echo >> expect_optional
    ok_pipeline.validate()  # should not raise
    runner_ok = Flujo(ok_pipeline)
    result_ok = runner_ok.run("hi")
    assert result_ok.step_history[-1].output == "hi"

    bad_pipeline = maybe_str >> need_str
    with pytest.raises(TypeMismatchError):
        bad_pipeline.validate()
    runner_bad = Flujo(bad_pipeline)
    with pytest.raises(TypeMismatchError):
        runner_bad.run("")
