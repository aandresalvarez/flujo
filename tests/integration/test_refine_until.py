import asyncio
from typing import Any

import pytest
from flujo.domain import Step, Pipeline
from flujo.domain.models import RefinementCheck
from flujo.testing.utils import StubAgent, gather_result
from tests.conftest import create_test_flujo
from flujo.domain.models import BaseModel


@pytest.mark.asyncio
async def test_refine_until_basic() -> None:
    gen_agent = StubAgent(["draft1", "draft2"])
    gen_pipeline = Pipeline.from_step(Step.model_validate({"name": "gen", "agent": gen_agent}))

    critic_agent = StubAgent(
        [
            RefinementCheck(is_complete=False, feedback="bad"),
            RefinementCheck(is_complete=True, feedback="good"),
        ]
    )
    critic_pipeline = Pipeline.from_step(
        Step.model_validate({"name": "crit_conc", "agent": critic_agent})
    )

    loop = Step.refine_until(
        name="refine",
        generator_pipeline=gen_pipeline,
        critic_pipeline=critic_pipeline,
        max_refinements=3,
    )

    runner = create_test_flujo(loop)
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
    gen_pipeline = Pipeline.from_step(Step.model_validate({"name": "gen", "agent": gen_agent}))

    critic_agent = StubAgent(
        [
            RefinementCheck(is_complete=False, feedback="err"),
            RefinementCheck(is_complete=True, feedback="done"),
        ]
    )
    critic_pipeline = Pipeline.from_step(
        Step.model_validate({"name": "crit_conc", "agent": critic_agent})
    )

    def fmap(original: str | None, check: RefinementCheck) -> dict[str, str | None]:
        return {"original_input": f"{original}-orig", "feedback": f"fix:{check.feedback}"}

    loop = Step.refine_until(
        name="refine_map",
        generator_pipeline=gen_pipeline,
        critic_pipeline=critic_pipeline,
        max_refinements=3,
        feedback_mapper=fmap,
    )

    runner = create_test_flujo(loop)
    result = await gather_result(runner, "goal")
    step_result = result.step_history[-1]
    assert step_result.output == "v2"
    assert gen_agent.inputs[1] == {"original_input": "goal-orig", "feedback": "fix:err"}


class SimpleCtx(BaseModel):
    pass


@pytest.mark.asyncio
async def test_refine_until_with_custom_context() -> None:
    gen_agent = StubAgent(["one", "two"])
    gen_pipeline = Pipeline.from_step(Step.model_validate({"name": "gen", "agent": gen_agent}))

    critic_agent = StubAgent([RefinementCheck(is_complete=True)])
    critic_pipeline = Pipeline.from_step(
        Step.model_validate({"name": "crit_conc", "agent": critic_agent})
    )

    loop = Step.refine_until(
        name="refine_ctx",
        generator_pipeline=gen_pipeline,
        critic_pipeline=critic_pipeline,
    )

    runner = create_test_flujo(loop, context_model=SimpleCtx)
    result = await gather_result(runner, "start")
    step_result = result.step_history[-1]
    assert step_result.output == "one"
    assert gen_agent.inputs[0] == {"original_input": "start", "feedback": None}


@pytest.mark.asyncio
async def test_refine_until_concurrent_runs_isolated() -> None:
    class GenAgent:
        def __init__(self) -> None:
            self.inputs: list[dict[str, Any]] = []

        async def run(self, data: dict[str, Any], **_: Any) -> str:
            self.inputs.append(data)
            await asyncio.sleep(0)
            fb = data.get("feedback") or "none"
            return f"{data['original_input']}-{fb}"

    class CriticAgent:
        async def run(self, artifact: str, **_: Any) -> RefinementCheck:
            await asyncio.sleep(0)
            if artifact.endswith("-none"):
                return RefinementCheck(is_complete=False, feedback="fix")
            return RefinementCheck(is_complete=True)

    gen_agent = GenAgent()
    gen_pipeline = Pipeline.from_step(Step.model_validate({"name": "gen_conc", "agent": gen_agent}))
    critic_agent = CriticAgent()
    critic_pipeline = Pipeline.from_step(
        Step.model_validate({"name": "crit_conc", "agent": critic_agent})
    )

    loop = Step.refine_until(
        name="refine_concurrent",
        generator_pipeline=gen_pipeline,
        critic_pipeline=critic_pipeline,
        max_refinements=2,
    )

    runner = create_test_flujo(loop)

    async def run_one(val: str) -> Any:
        return await gather_result(runner, val)

    r1, r2 = await asyncio.gather(run_one("A"), run_one("B"))

    assert r1.step_history[-1].output == "A-fix"
    assert r2.step_history[-1].output == "B-fix"
    assert len(gen_agent.inputs) == 4
    assert {"original_input": "A", "feedback": None} in gen_agent.inputs
    assert {"original_input": "A", "feedback": "fix"} in gen_agent.inputs
    assert {"original_input": "B", "feedback": None} in gen_agent.inputs
    assert {"original_input": "B", "feedback": "fix"} in gen_agent.inputs
