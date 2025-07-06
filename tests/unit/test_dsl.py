from flujo.domain import Step, Pipeline, step, AgentProcessors
from flujo.domain.models import BaseModel
from unittest.mock import AsyncMock, MagicMock, Mock
from flujo.domain.plugins import ValidationPlugin
from typing import Any
import pytest
from flujo.domain import StepConfig


def test_step_chaining_operator() -> None:
    a = Step.model_validate({"name": "A"})
    b = Step.model_validate({"name": "B"})
    pipeline = a >> b
    assert isinstance(pipeline, Pipeline)
    assert [s.name for s in pipeline.steps] == ["A", "B"]

    c = Step.model_validate({"name": "C"})
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
    step = Step.model_validate({"name": "A", "config": StepConfig(max_retries=5)})
    assert step.config.max_retries == 5


def test_dsl() -> None:
    step = Step.model_validate({"name": "dummy"})
    assert step.name == "dummy"


def test_dsl_with_step() -> None:
    step = Step.model_validate({"name": "A"})
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


def test_step_class_methods_create_correct_steps() -> None:
    agent = MagicMock()

    review_step = Step.review(agent)
    assert isinstance(review_step, Step)
    assert review_step.name == "review"
    assert review_step.agent is agent

    solution_step = Step.solution(agent, max_retries=5)
    assert solution_step.name == "solution"
    assert solution_step.config.max_retries == 5

    validate_step = Step.validate_step(agent)
    assert validate_step.name == "validate"


def test_step_fluent_builder_methods() -> None:
    agent = MagicMock()
    plugin1 = MagicMock(spec=ValidationPlugin)
    plugin2 = MagicMock(spec=ValidationPlugin)
    handler1 = Mock()
    handler2 = Mock()

    step = Step.model_validate(
        {
            "name": "test_step",
            "agent": agent,
            "plugins": [(plugin1, 0), (plugin2, 10)],
            "failure_handlers": [handler1, handler2],
        }
    )

    assert isinstance(step, Step)
    assert len(step.plugins) == 2
    assert step.plugins[0] == (plugin1, 0)
    assert step.plugins[1] == (plugin2, 10)
    assert len(step.failure_handlers) == 2
    assert step.failure_handlers == [handler1, handler2]


def test_step_init_handles_mixed_plugin_formats() -> None:
    agent = MagicMock()
    plugin1 = MagicMock(spec=ValidationPlugin)
    plugin2 = MagicMock(spec=ValidationPlugin)

    step = Step.model_validate(
        {"name": "test_init", "agent": agent, "plugins": [(plugin1, 0), (plugin2, 5)]}
    )

    assert len(step.plugins) == 2
    assert step.plugins[0] == (plugin1, 0)
    assert step.plugins[1] == (plugin2, 5)


@pytest.mark.asyncio
async def test_step_from_callable_basic() -> None:
    async def echo(x: str) -> int:
        return len(x)

    step = Step.from_callable(echo)
    assert step.name == "echo"
    result = await step.arun("hi")
    assert result == 2


@pytest.mark.asyncio
async def test_step_from_callable_name_and_config() -> None:
    async def do(x: int) -> int:
        return x + 1

    step = Step.from_callable(do, name="increment", timeout_s=5)
    assert step.name == "increment"
    assert step.config.timeout_s == 5
    out = await step.arun(1)
    assert out == 2


class _Service:
    async def process(self, value: str) -> str:
        return value.upper()


@pytest.mark.asyncio
async def test_step_from_callable_bound_method() -> None:
    svc = _Service()
    step = Step.from_callable(svc.process)
    assert step.name == "process"
    assert await step.arun("ok") == "OK"


@pytest.mark.asyncio
async def test_step_from_callable_untyped_defaults_any() -> None:
    async def untyped(x: Any) -> Any:
        return x

    step = Step.from_callable(untyped)
    assert step.name == "untyped"
    assert await step.arun(5) == 5


@pytest.mark.asyncio
async def test_step_from_mapper_basic() -> None:
    async def double(x: int) -> int:
        return x * 2

    step = Step.from_mapper(double)
    assert isinstance(step, Step)
    assert await step.arun(3) == 6


@pytest.mark.asyncio
async def test_step_decorator_basic() -> None:
    @step
    async def echo(x: str) -> int:
        return len(x)

    assert isinstance(echo, Step)
    assert echo.name == "echo"
    result = await echo.arun("hi")
    assert result == 2


@pytest.mark.asyncio
async def test_step_decorator_name_and_config() -> None:
    @step(name="inc", timeout_s=10)
    async def do(x: int) -> int:
        return x + 1

    assert do.name == "inc"
    assert do.config.timeout_s == 10
    assert await do.arun(1) == 2


@pytest.mark.asyncio
async def test_step_arun_basic() -> None:
    @step
    async def echo(x: str) -> int:
        return len(x)

    result = await echo.arun("hi")
    assert result == 2


class DummyCtx(BaseModel):
    num: int = 0


@pytest.mark.asyncio
async def test_step_arun_with_context() -> None:
    @step
    async def increment(x: int, *, context: DummyCtx) -> int:
        context.num += x
        return context.num

    ctx = DummyCtx(num=1)
    out = await increment.arun(2, context=ctx)
    assert out == 3
    assert ctx.num == 3


@pytest.mark.asyncio
async def test_step_arun_no_agent() -> None:
    step_without_agent = Step.model_validate({"name": "blank"})
    with pytest.raises(ValueError):
        await step_without_agent.arun(None)


def test_pipeline_chaining_operator() -> None:
    """Ensure that `Pipeline >> Pipeline` concatenates their steps in order."""
    a1 = Step.model_validate({"name": "A1"})
    a2 = Step.model_validate({"name": "A2"})
    b1 = Step.model_validate({"name": "B1"})
    b2 = Step.model_validate({"name": "B2"})

    pipeline_one = a1 >> a2  # Pipeline with steps [A1, A2]
    pipeline_two = b1 >> b2  # Pipeline with steps [B1, B2]

    chained_pipeline = pipeline_one >> pipeline_two

    assert isinstance(chained_pipeline, Pipeline)
    assert [s.name for s in chained_pipeline.steps] == ["A1", "A2", "B1", "B2"]


@pytest.mark.asyncio
async def test_step_decorator_matches_from_callable() -> None:
    async def add(x: int) -> int:
        return x + 1

    via_decorator = step(add)
    via_method = Step.from_callable(add)

    assert via_decorator.name == via_method.name
    assert via_decorator.updates_context == via_method.updates_context
    assert via_decorator.config.model_dump() == via_method.config.model_dump()
    assert via_decorator.processors.model_dump() == via_method.processors.model_dump()
    assert via_decorator.persist_feedback_to_context == via_method.persist_feedback_to_context
    assert via_decorator.persist_validation_results_to == via_method.persist_validation_results_to
    assert await via_decorator.arun(1) == await via_method.arun(1)


@pytest.mark.asyncio
async def test_step_decorator_matches_kwargs() -> None:
    async def add(x: int) -> int:
        return x + 1

    procs = AgentProcessors()

    via_decorator = step(
        add,
        name="plus_one",
        updates_context=True,
        processors=procs,
        persist_feedback_to_context="fb",
        persist_validation_results_to="vr",
        timeout_s=5,
    )
    via_method = Step.from_callable(
        add,
        name="plus_one",
        updates_context=True,
        processors=procs,
        persist_feedback_to_context="fb",
        persist_validation_results_to="vr",
        timeout_s=5,
    )

    assert via_decorator.name == via_method.name == "plus_one"
    assert via_decorator.updates_context and via_method.updates_context
    assert via_decorator.config.timeout_s == via_method.config.timeout_s == 5
    assert via_decorator.processors.model_dump() == via_method.processors.model_dump()
    assert via_decorator.persist_feedback_to_context == "fb"
    assert via_decorator.persist_validation_results_to == "vr"
    assert await via_decorator.arun(2) == await via_method.arun(2)
