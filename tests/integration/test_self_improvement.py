import json
import pytest
from pydantic_ai_orchestrator.application.self_improvement import (
    evaluate_and_improve,
    SelfImprovementAgent,
)
from pydantic_ai_orchestrator.application.pipeline_runner import PipelineRunner
from pydantic_ai_orchestrator.application.eval_adapter import run_pipeline_async
from pydantic_ai_orchestrator.domain import Step
from pydantic_ai_orchestrator.testing.utils import StubAgent
from pydantic_evals import Dataset, Case


class DummyAgent:
    async def run(self, prompt: str) -> str:
        return json.dumps(
            {
                "suggestions": [
                    {
                        "target_step_name": "solution",
                        "suggestion_type": "prompt_modification",
                        "failure_pattern_summary": "error",
                        "detailed_explanation": "fix it",
                        "prompt_modification_details": {"modification_instruction": "fix"},
                        "example_failing_input_snippets": ["c1"],
                    }
                ]
            }
        )


@pytest.mark.asyncio
async def test_e2e_self_improvement_with_mocked_llm_suggestions():
    agent = StubAgent(["ok"])
    pipeline = Step.solution(agent)
    runner = PipelineRunner(pipeline)
    dataset = Dataset(cases=[Case(inputs="hi", expected_output="wrong")])
    report = await evaluate_and_improve(
        lambda x: run_pipeline_async(x, runner=runner),
        dataset,
        SelfImprovementAgent(DummyAgent()),
    )
    assert report.suggestions[0].target_step_name == "solution"
    assert report.suggestions[0].suggestion_type.value == "prompt_modification"


@pytest.mark.asyncio
async def test_build_context_for_self_improvement_agent():
    from pydantic_ai_orchestrator.application.self_improvement import _build_context
    pr = Step.solution(StubAgent(["ok"]))
    runner = PipelineRunner(pr)
    dataset = Dataset(cases=[Case(name="c1", inputs="i", expected_output="o")])
    report = await dataset.evaluate(lambda x: run_pipeline_async(x, runner=runner))
    context = _build_context(report.cases, None)
    assert "Case: c1" in context
