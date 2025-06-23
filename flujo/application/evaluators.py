"""Custom evaluators for pydantic-evals integration."""

from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from typing import Any

from ..domain.models import PipelineResult


class FinalSolutionEvaluator(Evaluator):
    """Extracts the final step output from a PipelineResult."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        result: PipelineResult[Any] = ctx.output
        final_output = None
        if result.step_history:
            final_output = result.step_history[-1].output
        return final_output == ctx.expected_output
