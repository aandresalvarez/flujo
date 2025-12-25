import pytest

from flujo.application.self_improvement import SelfImprovementAgent
from flujo.application.tree_search_improvement import (
    build_tree_search_tuning_prompt,
    build_tree_search_trace_report,
    distill_tree_search_path,
    tune_tree_search_evaluator,
)
from flujo.domain.models import (
    ImprovementReport,
    ImprovementSuggestion,
    SearchNode,
    SearchState,
    SuggestionType,
)


def _make_state() -> SearchState:
    state = SearchState(status="completed", best_node_id="n2")
    state.nodes = {
        "n0": SearchNode(
            node_id="n0",
            candidate="root",
            output="root",
            depth=0,
            state_hash="h0",
            metadata={"rubric_score": 0.1},
        ),
        "n1": SearchNode(
            node_id="n1",
            parent_id="n0",
            candidate="dead_end",
            output="dead_end",
            depth=1,
            state_hash="h1",
            metadata={"rubric_score": 0.95},
        ),
        "n2": SearchNode(
            node_id="n2",
            parent_id="n0",
            candidate="winner",
            output="winner",
            depth=1,
            state_hash="h2",
            metadata={"rubric_score": 1.0},
        ),
    }
    return state


def test_tree_search_trace_report_flags_dead_end() -> None:
    report = build_tree_search_trace_report(
        _make_state(),
        objective="Solve the task",
        high_score_threshold=0.9,
    )
    assert report.best_node_id == "n2"
    assert report.winner_path[-1].node_id == "n2"
    assert any(issue.node_id == "n1" for issue in report.high_score_dead_ends)


def test_tree_search_tuning_prompt_contains_objective() -> None:
    report = build_tree_search_trace_report(
        _make_state(),
        objective="Find solution",
        high_score_threshold=0.9,
    )
    prompt = build_tree_search_tuning_prompt(report)
    assert "Primary Objective: Find solution" in prompt


@pytest.mark.asyncio
async def test_tree_search_tuning_uses_agent() -> None:
    report = build_tree_search_trace_report(
        _make_state(),
        objective="Find solution",
        high_score_threshold=0.9,
    )

    class _Agent:
        async def run(self, _prompt: str) -> ImprovementReport:
            return ImprovementReport(
                suggestions=[
                    ImprovementSuggestion(
                        suggestion_type=SuggestionType.PROMPT_MODIFICATION,
                        failure_pattern_summary="Over-scored dead end",
                        detailed_explanation="Tighten rubric for evidence quality.",
                    )
                ]
            )

    improvement = await tune_tree_search_evaluator(
        report,
        improvement_agent=SelfImprovementAgent(_Agent()),
    )
    assert improvement.suggestions


@pytest.mark.asyncio
async def test_tree_search_distillation_uses_agent() -> None:
    state = _make_state()

    class _Distiller:
        async def run(self, _prompt: str) -> str:
            return "Distilled prompt"

    result = await distill_tree_search_path(
        state,
        objective="Find solution",
        distillation_agent=_Distiller(),
    )
    assert result == "Distilled prompt"
