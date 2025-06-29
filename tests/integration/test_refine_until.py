import pytest
from flujo.application.flujo_engine import Flujo
from flujo.domain import Step, Pipeline, RefinementCheck
from flujo.testing.utils import StubAgent, gather_result
from pydantic import BaseModel


@pytest.mark.asyncio
async def test_refine_until_basic() -> None:
    gen_agent = StubAgent(["draft1", "draft2"])
    gen_pipeline = Pipeline.from_step(Step("gen", gen_agent))

    critic_agent = StubAgent(
        [
            RefinementCheck(is_complete=False, feedback="bad"),
            RefinementCheck(is_complete=True, feedback="good"),
        ]
    )
    critic_pipeline = Pipeline.from_step(Step("crit", critic_agent))

    loop = Step.refine_until(
        name="refine",
        generator_pipeline=gen_pipeline,
        critic_pipeline=critic_pipeline,
        max_refinements=3,
    )

    runner = Flujo(loop)
    result = await gather_result(runner, "goal")
    step_result = result.step_history[-1]
    assert step_result.success is True
    assert step_result.attempts == 2
    assert step_result.output == "draft2"
    assert gen_agent.inputs == [
        {"original_input": "goal", "feedback": None},
        {"original_input": "goal", "feedback": "bad"},
    ]


@pytest.mark.asyncio
async def test_refine_until_with_feedback_mapper() -> None:
    gen_agent = StubAgent(["v1", "v2"])
    gen_pipeline = Pipeline.from_step(Step("gen", gen_agent))

    critic_agent = StubAgent(
        [
            RefinementCheck(is_complete=False, feedback="err"),
            RefinementCheck(is_complete=True, feedback="done"),
        ]
    )
    critic_pipeline = Pipeline.from_step(Step("crit", critic_agent))

    def fmap(original: str | None, check: RefinementCheck) -> dict[str, str | None]:
        return {"original_input": f"{original}-orig", "feedback": f"fix:{check.feedback}"}

    loop = Step.refine_until(
        name="refine_map",
        generator_pipeline=gen_pipeline,
        critic_pipeline=critic_pipeline,
        max_refinements=3,
        feedback_mapper=fmap,
    )

    runner = Flujo(loop)
    result = await gather_result(runner, "goal")
    step_result = result.step_history[-1]
    assert step_result.output == "v2"
    assert gen_agent.inputs[1] == {"original_input": "goal-orig", "feedback": "fix:err"}


class SimpleCtx(BaseModel):
    pass


@pytest.mark.asyncio
async def test_refine_until_with_custom_context() -> None:
    gen_agent = StubAgent(["one", "two"])
    gen_pipeline = Pipeline.from_step(Step("gen", gen_agent))

    critic_agent = StubAgent([RefinementCheck(is_complete=True)])
    critic_pipeline = Pipeline.from_step(Step("crit", critic_agent))

    loop = Step.refine_until(
        name="refine_ctx",
        generator_pipeline=gen_pipeline,
        critic_pipeline=critic_pipeline,
    )

    runner = Flujo(loop, context_model=SimpleCtx)
    result = await gather_result(runner, "start")
    step_result = result.step_history[-1]
    assert step_result.output == "one"
    assert gen_agent.inputs[0] == {"original_input": "start", "feedback": None}
