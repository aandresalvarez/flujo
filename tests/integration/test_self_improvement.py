import functools
import pytest
from flujo.application.self_improvement import (
    evaluate_and_improve,
    SelfImprovementAgent,
)
from flujo.domain.models import ImprovementReport, ImprovementSuggestion
from flujo.application.eval_adapter import run_pipeline_async
from flujo.domain import Step
from flujo.testing.utils import StubAgent
from tests.conftest import create_test_flujo
from pydantic_evals import Dataset, Case


class DummyAgent:
    async def run(self, prompt: str) -> ImprovementReport:
        return ImprovementReport(
            suggestions=[
                ImprovementSuggestion(
                    target_step_name="solution",
                    suggestion_type="prompt_modification",
                    failure_pattern_summary="error",
                    detailed_explanation="fix it",
                    prompt_modification_details={"modification_instruction": "fix"},
                    example_failing_input_snippets=["c1"],
                )
            ]
        )


@pytest.mark.asyncio
async def test_e2e_self_improvement_with_mocked_llm_suggestions():
    agent = StubAgent(["ok"])
    pipeline = Step.solution(agent)
    runner = create_test_flujo(pipeline)
    dataset = Dataset(cases=[Case(inputs="hi", expected_output="wrong")])
    report = await evaluate_and_improve(
        functools.partial(run_pipeline_async, runner=runner),
        dataset,
        SelfImprovementAgent(DummyAgent()),
    )
    assert report.suggestions[0].target_step_name == "solution"
    assert report.suggestions[0].suggestion_type.value == "prompt_modification"


@pytest.mark.asyncio
async def test_build_context_for_self_improvement_agent():
    from flujo.application.self_improvement import _build_context

    pr = Step.solution(StubAgent(["ok"]))
    runner = create_test_flujo(pr)
    dataset = Dataset(cases=[Case(name="c1", inputs="i", expected_output="o")])
    report = await dataset.evaluate(functools.partial(run_pipeline_async, runner=runner))
    context = _build_context(report.cases, None)
    assert "Case: c1" in context


@pytest.mark.asyncio
async def test_self_improvement_context_includes_config_and_prompts(monkeypatch):
    captured: dict[str, str] = {}

    class CaptureAgent:
        async def run(self, prompt: str) -> ImprovementReport:
            captured["prompt"] = prompt
            return ImprovementReport(suggestions=[])

    agent = StubAgent(["ok"])
    agent.system_prompt = "Test prompt"
    pipeline = Step.solution(agent, max_retries=5, timeout_s=30, temperature=0.5)
    runner = create_test_flujo(pipeline)
    dataset = Dataset(cases=[Case(inputs="i", expected_output="o")])

    await evaluate_and_improve(
        functools.partial(run_pipeline_async, runner=runner),
        dataset,
        SelfImprovementAgent(CaptureAgent()),
        pipeline_definition=pipeline,
    )

    assert "Config(retries=5" in captured["prompt"]
    assert "timeout=30.0s" in captured["prompt"]
    assert "temperature=0.5" in captured["prompt"]
    assert "SystemPromptSummary" in captured["prompt"]
