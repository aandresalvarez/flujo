from pydantic_ai_orchestrator.domain import Step, Pipeline
from unittest.mock import AsyncMock
from typing import Any


def test_step_chaining_operator() -> None:
    a = Step("A")
    b = Step("B")
    pipeline = a >> b
    assert isinstance(pipeline, Pipeline)
    assert [s.name for s in pipeline.steps] == ["A", "B"]

    c = Step("C")
    pipeline2 = pipeline >> c
    assert [s.name for s in pipeline2.steps] == ["A", "B", "C"]


def test_role_based_constructor() -> None:
    agent = AsyncMock()
    step = Step.review(agent)
    assert step.name == "review"
    assert step.agent is agent

    vstep = Step.validate(agent)
    assert vstep.name == "validate"


def test_step_configuration() -> None:
    step = Step("A", max_retries=5)
    assert step.config.max_retries == 5


def test_dsl() -> None:
    a: int = 1
    b: int = 2


def test_dsl_with_step() -> None:
    c: int = 3


def test_dsl_with_agent() -> None:
    step: Step[Any, Any]


def test_dsl_with_agent_and_step() -> None:
    step: Step[Any, Any]
