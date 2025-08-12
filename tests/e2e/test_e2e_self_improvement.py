import functools
import pytest
from flujo.application.eval_adapter import run_pipeline_async
from flujo.application.self_improvement import (
    evaluate_and_improve,
    SelfImprovementAgent,
)
from flujo.domain.models import ImprovementReport, SuggestionType
from flujo.domain import Step
from flujo.domain.validation import BaseValidator, ValidationResult
from flujo.testing.utils import StubAgent
from tests.conftest import create_test_flujo
from pydantic_evals import Dataset, Case


class ConciseValidator(BaseValidator):
    async def validate(self, output_to_check: str, *, context=None) -> ValidationResult:
        words = len(str(output_to_check).split())
        return ValidationResult(
            is_valid=words <= 5,
            feedback=None if words <= 5 else "too verbose",
            validator_name=self.name,
        )


class IdentityAgent:
    async def run(self, data: str) -> str:
        return data


IMPROVEMENT_JSON = {
    "suggestions": [
        {
            "target_step_name": "solution",
            "suggestion_type": "prompt_modification",
            "failure_pattern_summary": "Solution step output too verbose",
            "detailed_explanation": "Shorten the solution output.",
            "prompt_modification_details": {"modification_instruction": "Be concise."},
            "example_failing_input_snippets": ["Prompt1"],
            "estimated_impact": "HIGH",
            "estimated_effort_to_implement": "LOW",
        }
    ]
}


class EchoAgent:
    async def run(self, prompt: str) -> dict:
        # Deterministic stub: avoid external calls and return the expected JSON payload directly
        return IMPROVEMENT_JSON


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_e2e_self_improvement_workflow() -> None:
    solution_agent = StubAgent(
        [
            "This output is definitely far too long and verbose to be considered concise.",
            "Short answer.",
        ]
    )
    pipeline = Step.solution(solution_agent) >> Step.validate_step(
        IdentityAgent(), validators=[ConciseValidator()]
    )
    runner = create_test_flujo(pipeline)
    dataset = Dataset(
        cases=[
            Case(name="fail_case", inputs="Prompt1", expected_output=None),
            Case(name="success_case", inputs="Prompt2", expected_output=None),
        ]
    )
    report = await evaluate_and_improve(
        functools.partial(run_pipeline_async, runner=runner),
        dataset,
        SelfImprovementAgent(EchoAgent()),
        pipeline_definition=pipeline,
    )
    assert isinstance(report, ImprovementReport)
    assert report.suggestions
    sugg = report.suggestions[0]
    assert sugg.target_step_name == "solution"
    assert sugg.suggestion_type == SuggestionType.PROMPT_MODIFICATION
    assert "verbose" in sugg.failure_pattern_summary.lower()
    assert sugg.prompt_modification_details is not None
