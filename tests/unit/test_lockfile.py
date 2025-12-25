from __future__ import annotations

from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.step import Step
from flujo.domain.models import PipelineResult, StepResult
from flujo.infra.lockfile import build_lockfile_data
from flujo.utils.hash import stable_digest


class _DummyAgent:
    __flujo_skill_id__ = "dummy.skill"
    _original_system_prompt = "system prompt"

    async def run(self, data: object | None = None, **_kwargs: object) -> object:
        return data


def test_build_lockfile_data_hashes_skills_and_prompts() -> None:
    pipeline = Pipeline.from_step(Step(name="dummy", agent=_DummyAgent()))
    result = PipelineResult(step_history=[StepResult(name="dummy", success=True, output="ok")])
    data = build_lockfile_data(
        pipeline=pipeline,
        result=result,
        pipeline_name="pipe",
        pipeline_version="1.0",
        pipeline_id="pid",
        run_id="rid",
    )

    assert data["pipeline"]["name"] == "pipe"
    assert data["skills"][0]["skill_id"] == "dummy.skill"
    assert data["prompts"][0]["hash"] == stable_digest("system prompt")
