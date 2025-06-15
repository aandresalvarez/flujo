from flujo.domain import Step, Pipeline
from unittest.mock import AsyncMock


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

    vstep = Step.validate_step(agent)
    assert vstep.name == "validate"
    assert vstep.agent is agent


def test_step_configuration() -> None:
    step = Step("A", max_retries=5)
    assert step.config.max_retries == 5


def test_dsl() -> None:
    step = Step("dummy")
    assert step.name == "dummy"


def test_dsl_with_step() -> None:
    step = Step("A")
    pipeline = Pipeline.from_step(step)
    assert pipeline.steps == [step]


def test_dsl_with_agent() -> None:
    agent = AsyncMock()
    step = Step.review(agent)
    assert step.agent is agent


def test_dsl_with_agent_and_step() -> None:
    agent = AsyncMock()
    step = Step.solution(agent)
    pipeline = step >> Step.validate_step(agent)
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0].name == step.name
    assert pipeline.steps[0].agent is step.agent
    assert pipeline.steps[1].name == "validate"
    assert pipeline.steps[1].agent is agent
