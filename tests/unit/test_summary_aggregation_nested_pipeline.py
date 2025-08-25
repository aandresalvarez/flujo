from __future__ import annotations

from typing import Any

from flujo.domain.dsl import Pipeline, Step
from flujo.application.runner import Flujo
from flujo.domain.models import PipelineResult, StepResult, PipelineContext


def _make_nested_pipeline_result() -> PipelineResult[Any]:
    # Construct a nested PipelineResult with one inner step carrying usage
    inner = StepResult(name="inner_step", success=True, cost_usd=0.1234, token_counts=42)
    return PipelineResult[Any](step_history=[inner], total_cost_usd=0.1234, total_tokens=42)


def test_nested_pipeline_result_aggregates_usage() -> None:
    # A single step that returns a nested PipelineResult from its agent
    async def _agent(_: Any) -> Any:
        return _make_nested_pipeline_result()

    step = Step.from_callable(_agent, name="wrapper_step")
    pipeline = Pipeline.from_step(step)

    runner = Flujo[Any, Any, PipelineContext](pipeline)
    result = runner.run("input")

    # Expect top-level totals to include nested usage
    assert abs(result.total_cost_usd - 0.1234) < 1e-8
    assert result.total_tokens == 42

    # The single top-level step should have its own usage updated to reflect nested totals
    assert len(result.step_history) == 1
    top = result.step_history[0]
    assert abs(top.cost_usd - 0.1234) < 1e-8
    assert top.token_counts == 42


def test_nested_step_history_attached_for_visibility() -> None:
    async def _agent(_: Any) -> Any:
        return _make_nested_pipeline_result()

    step = Step.from_callable(_agent, name="wrapper_step")
    pipeline = Pipeline.from_step(step)

    runner = Flujo[Any, Any, PipelineContext](pipeline)
    result = runner.run("input")

    # The top-level step should expose nested history for CLI/reporting introspection
    top = result.step_history[0]
    assert isinstance(top.step_history, list)
    assert len(top.step_history) == 1
    assert top.step_history[0].name == "inner_step"
