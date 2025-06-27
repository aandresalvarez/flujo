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
