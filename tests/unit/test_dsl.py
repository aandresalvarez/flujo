from flujo.domain import Step, Pipeline, step
from unittest.mock import AsyncMock, MagicMock, Mock
from flujo.domain.plugins import ValidationPlugin
import pytest


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

    step = (
        Step("test_step", agent)
        .add_plugin(plugin1)
        .add_plugin(plugin2, priority=10)
        .on_failure(handler1)
        .on_failure(handler2)
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

    step = Step("test_init", agent, plugins=[plugin1, (plugin2, 5)])

    assert len(step.plugins) == 2
    assert step.plugins[0] == (plugin1, 0)
    assert step.plugins[1] == (plugin2, 5)


@pytest.mark.asyncio
async def test_step_from_callable_basic() -> None:
    async def echo(x: str) -> int:
        return len(x)

    step = Step.from_callable(echo)
    assert step.name == "echo"
    result = await step.agent.run("hi")  # type: ignore[call-arg]
    assert result == 2


@pytest.mark.asyncio
async def test_step_from_callable_name_and_config() -> None:
    async def do(x: int) -> int:
        return x + 1

    step = Step.from_callable(do, name="increment", timeout_s=5)
    assert step.name == "increment"
    assert step.config.timeout_s == 5
    out = await step.agent.run(1)  # type: ignore[call-arg]
    assert out == 2


class _Service:
    async def process(self, value: str) -> str:
        return value.upper()


@pytest.mark.asyncio
async def test_step_from_callable_bound_method() -> None:
    svc = _Service()
    step = Step.from_callable(svc.process)
    assert step.name == "process"
    assert await step.agent.run("ok") == "OK"  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_step_from_callable_untyped_defaults_any() -> None:
    async def untyped(x):  # type: ignore[no-untyped-def]
        return x

    step = Step.from_callable(untyped)
    assert step.name == "untyped"
    assert await step.agent.run(5) == 5  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_step_decorator_basic() -> None:
    @step
    async def echo(x: str) -> int:
        return len(x)

    assert isinstance(echo, Step)
    assert echo.name == "echo"
    result = await echo.agent.run("hi")  # type: ignore[call-arg]
    assert result == 2


@pytest.mark.asyncio
async def test_step_decorator_name_and_config() -> None:
    @step(name="inc", timeout_s=10)
    async def do(x: int) -> int:
        return x + 1

    assert do.name == "inc"
    assert do.config.timeout_s == 10
    assert await do.agent.run(1) == 2  # type: ignore[call-arg]
