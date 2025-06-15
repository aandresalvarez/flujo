from flujo.application.evaluators import FinalSolutionEvaluator
from flujo.domain.models import PipelineResult, StepResult
from pydantic_evals.evaluators import EvaluatorContext


def test_final_solution_evaluator_matches_expected() -> None:
    pipeline_result = PipelineResult(
        step_history=[
            StepResult(name="sol", output="hi"),
        ]
    )
    ctx = EvaluatorContext(
        name="c1",
        inputs="hello",
        metadata=None,
        expected_output="hi",
        output=pipeline_result,
        duration=0.0,
        _span_tree=None,
        attributes={},
        metrics={},
    )
    ev = FinalSolutionEvaluator()
    assert ev.evaluate_sync(ctx)
