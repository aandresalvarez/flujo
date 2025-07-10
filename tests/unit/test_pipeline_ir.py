import pytest
import hashlib

from flujo.domain import step, Pipeline, Step
from flujo.application.runner import Flujo
from flujo.testing.utils import gather_result


@step
async def add_one(x: int) -> int:
    return x + 1


@step
async def double(x: int) -> int:
    return x * 2


def exit_condition(out: int, _ctx: object | None) -> bool:
    return out >= 2


def branch_condition(out: int, _ctx: object | None) -> str:
    return "done"


@pytest.mark.asyncio
async def test_pipeline_ir_roundtrip() -> None:
    pipeline = add_one >> double

    model = pipeline.to_model()
    loaded = Pipeline.from_model(model)

    runner = Flujo(loaded)
    result = await gather_result(runner, 2)
    assert result.step_history[-1].output == 6
    # Ensure step_uids preserved
    assert loaded.steps[0].step_uid == pipeline.steps[0].step_uid
    assert loaded.steps[1].step_uid == pipeline.steps[1].step_uid


@pytest.mark.asyncio
async def test_complex_pipeline_ir_roundtrip() -> None:
    loop_body = Pipeline.from_step(add_one)
    loop_step = Pipeline.from_step(
        Step.loop_until(
            name="loop",
            loop_body_pipeline=loop_body,
            exit_condition_callable=exit_condition,
        )
    )
    cond_step = Step.branch_on(
        name="branch",
        condition_callable=branch_condition,
        branches={"done": loop_step},
    )
    primary = Step.model_validate({"name": "main", "agent": None})
    fallback = Step.model_validate({"name": "fb", "agent": None})
    primary.fallback(fallback)
    pipeline = Pipeline(steps=[primary, cond_step, fallback])

    model = pipeline.to_model()
    loaded = Pipeline.from_model(model)

    assert loaded.steps[0].fallback_step is loaded.steps[2]


@pytest.mark.asyncio
async def test_pipeline_yaml_roundtrip() -> None:
    pipeline = add_one >> double
    yaml_text = pipeline.to_yaml()
    loaded = Pipeline.from_yaml(yaml_text)
    runner = Flujo(loaded)
    result = await gather_result(runner, 2)
    assert result.step_history[-1].output == 6
    assert loaded.spec_sha256 == hashlib.sha256(yaml_text.encode()).hexdigest()


@pytest.mark.asyncio
async def test_complex_pipeline_yaml_roundtrip() -> None:
    loop_body = Pipeline.from_step(add_one)
    loop_step = Pipeline.from_step(
        Step.loop_until(
            name="loop",
            loop_body_pipeline=loop_body,
            exit_condition_callable=exit_condition,
        )
    )
    cond_step = Step.branch_on(
        name="branch",
        condition_callable=branch_condition,
        branches={"done": loop_step},
    )
    primary = Step.model_validate({"name": "main", "agent": None})
    fallback = Step.model_validate({"name": "fb", "agent": None})
    primary.fallback(fallback)
    pipeline = Pipeline(steps=[primary, cond_step, fallback])

    yaml_text = pipeline.to_yaml()
    loaded = Pipeline.from_yaml(yaml_text)
    assert loaded.steps[0].fallback_step is loaded.steps[2]
    assert loaded.spec_sha256 == hashlib.sha256(yaml_text.encode()).hexdigest()
