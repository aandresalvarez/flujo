"""Step execution history tracking and aggregation."""

from __future__ import annotations

from pydantic import BaseModel as PydanticBaseModel

from ....domain.models import PipelineResult, StepResult


class StepHistoryTracker:
    """Manages step execution history tracking and aggregation."""

    def __init__(self) -> None:
        self._history: list[StepResult] = []

    def clear_history(self) -> None:
        """Clear the current step history."""
        self._history.clear()

    def add_step_result(self, step_result: StepResult) -> None:
        """Add a step result to the history."""
        self._history.append(step_result)

    def extend_history(self, step_results: list[StepResult]) -> None:
        """Extend history with multiple step results."""
        self._history.extend(step_results)

    def get_history(self) -> list[StepResult]:
        """Get a copy of the current step history."""
        return self._history.copy()

    def get_history_length(self) -> int:
        """Get the number of steps in the history."""
        return len(self._history)

    def get_last_step_result(self) -> StepResult | None:
        """Get the last step result in the history."""
        return self._history[-1] if self._history else None

    def aggregate_metrics(self) -> dict[str, float | int]:
        """Aggregate metrics from all step results in history."""
        total_cost = 0.0
        total_tokens = 0
        total_latency = 0.0

        for step_result in self._history:
            total_cost += getattr(step_result, "cost_usd", 0.0)
            total_tokens += getattr(step_result, "token_counts", 0)
            total_latency += getattr(step_result, "latency_s", 0.0)

        return {
            "total_cost_usd": total_cost,
            "total_tokens": total_tokens,
            "total_latency_s": total_latency,
            "step_count": len(self._history),
        }

    def create_pipeline_result(
        self, final_context: PydanticBaseModel | None = None, **kwargs: object
    ) -> PipelineResult[PydanticBaseModel]:
        """Create a PipelineResult with the current step history."""
        metrics = self.aggregate_metrics()
        return PipelineResult(
            step_history=self._history,
            total_cost_usd=metrics["total_cost_usd"],
            total_tokens=metrics["total_tokens"],
            final_pipeline_context=final_context,
            **kwargs,
        )

    def merge_nested_history(self, nested_result: PipelineResult[PydanticBaseModel]) -> None:
        """Merge step history from a nested pipeline execution."""
        if hasattr(nested_result, "step_history") and nested_result.step_history:
            self.extend_history(nested_result.step_history)
