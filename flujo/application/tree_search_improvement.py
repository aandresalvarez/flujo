"""Tree search trace analysis, tuning, and distillation helpers."""

from __future__ import annotations

from typing import Iterable

from pydantic import Field

from ..agents import make_agent_async, make_self_improvement_agent
from .conversation.history_manager import HistoryManager
from ..domain.models import BaseModel, ImprovementReport, SearchNode, SearchState
from ..infra.config_manager import get_config_manager
from .self_improvement import SelfImprovementAgent

DEFAULT_DISTILLATION_MODEL = "openai:gpt-4o"


class TreeSearchPathNode(BaseModel):
    node_id: str
    score: float | None = None
    candidate_preview: str = ""


class TreeSearchTraceIssue(BaseModel):
    node_id: str
    score: float
    candidate_preview: str
    reason: str


class TreeSearchTraceReport(BaseModel):
    objective: str
    goal_reached: bool
    total_nodes: int
    best_node_id: str | None = None
    best_score: float | None = None
    winner_path: list[TreeSearchPathNode] = Field(default_factory=list)
    high_score_dead_ends: list[TreeSearchTraceIssue] = Field(default_factory=list)


def _safe_preview(value: object, *, limit: int = 200) -> str:
    try:
        text = str(value)
    except Exception:
        text = repr(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _collect_path(state: SearchState, node_id: str | None) -> list[SearchNode]:
    path: list[SearchNode] = []
    current = node_id
    while current:
        node = state.nodes.get(current)
        if node is None:
            break
        path.append(node)
        current = node.parent_id
    path.reverse()
    return path


def build_tree_search_trace_report(
    state: SearchState,
    *,
    objective: str,
    high_score_threshold: float = 0.8,
) -> TreeSearchTraceReport:
    winner_path_nodes = _collect_path(state, state.best_node_id)
    winner_ids = {node.node_id for node in winner_path_nodes}
    best_score: float | None = None
    if state.best_node_id and state.best_node_id in state.nodes:
        try:
            best_score = float(state.nodes[state.best_node_id].metadata.get("rubric_score", 0.0))
        except Exception:
            best_score = None

    winner_path = [
        TreeSearchPathNode(
            node_id=node.node_id,
            score=float(node.metadata.get("rubric_score", 0.0))
            if node.metadata.get("rubric_score") is not None
            else None,
            candidate_preview=_safe_preview(
                node.output if node.output is not None else node.candidate
            ),
        )
        for node in winner_path_nodes
    ]

    issues: list[TreeSearchTraceIssue] = []
    for node in state.nodes.values():
        try:
            score = float(node.metadata.get("rubric_score", 0.0))
        except Exception:
            score = 0.0
        if score < high_score_threshold:
            continue
        if winner_ids and node.node_id in winner_ids:
            continue
        reason = "high_score_no_goal_reached" if not winner_ids else "high_score_off_winner_path"
        issues.append(
            TreeSearchTraceIssue(
                node_id=node.node_id,
                score=score,
                candidate_preview=_safe_preview(
                    node.output if node.output is not None else node.candidate
                ),
                reason=reason,
            )
        )

    return TreeSearchTraceReport(
        objective=objective,
        goal_reached=state.status == "completed",
        total_nodes=len(state.nodes),
        best_node_id=state.best_node_id,
        best_score=best_score,
        winner_path=winner_path,
        high_score_dead_ends=issues,
    )


def build_tree_search_tuning_prompt(report: TreeSearchTraceReport) -> str:
    lines: list[str] = []
    lines.append(
        "Analyze this TreeSearchStep trace and suggest rubric or prompt updates to "
        "avoid over-scoring dead-end branches."
    )
    lines.append("")
    lines.append(f"Primary Objective: {report.objective}")
    lines.append(f"Goal Reached: {report.goal_reached}")
    if report.best_score is not None:
        lines.append(f"Best Score: {report.best_score:.2f}")
    lines.append("")

    lines.append("Winner Path:")
    if report.winner_path:
        for idx, node in enumerate(report.winner_path, start=1):
            score = f"{node.score:.2f}" if node.score is not None else "n/a"
            lines.append(f"{idx}. score={score} candidate={node.candidate_preview}")
    else:
        lines.append("- None")

    lines.append("")
    lines.append("High-Score Dead Ends:")
    if report.high_score_dead_ends:
        for issue in report.high_score_dead_ends:
            lines.append(
                f"- node={issue.node_id} score={issue.score:.2f} "
                f"reason={issue.reason} candidate={issue.candidate_preview}"
            )
    else:
        lines.append("- None")

    lines.append("")
    lines.append(
        "Return an ImprovementReport JSON with rubric or prompt modifications. "
        "Prefer prompt_modification or config_adjustment suggestions."
    )
    return "\n".join(lines)


def _resolve_tuning_model(model: str | None) -> str | None:
    if model is not None:
        return model
    settings = get_config_manager().get_settings()
    shadow_cfg = getattr(settings, "shadow_eval", None)
    shadow_model = getattr(shadow_cfg, "judge_model", None) if shadow_cfg is not None else None
    default_model = getattr(settings, "default_self_improvement_model", None)
    return shadow_model or default_model


async def tune_tree_search_evaluator(
    report: TreeSearchTraceReport,
    *,
    improvement_agent: SelfImprovementAgent | None = None,
    model: str | None = None,
) -> ImprovementReport:
    if improvement_agent is None:
        agent = make_self_improvement_agent(model=_resolve_tuning_model(model))
        improvement_agent = SelfImprovementAgent(agent)
    prompt = build_tree_search_tuning_prompt(report)
    return await improvement_agent.run(prompt)


def build_tree_search_distillation_prompt(
    *,
    objective: str,
    path: Iterable[TreeSearchPathNode],
) -> str:
    hm = HistoryManager()
    parts: list[str] = []
    for node in path:
        score = f"{node.score:.2f}" if node.score is not None else "n/a"
        parts.append(f"[score={score}] {node.candidate_preview}")
    summary = hm.summarize(parts, max_tokens=800)
    return (
        "You are distilling a successful TreeSearchStep path into a reusable prompt.\n"
        "Create a concise prompt or few-shot example that guides the model to follow the same path.\n"
        f"Primary Objective: {objective}\n"
        "Winning Path:\n"
        f"{summary}\n"
        "Return only the distilled prompt text."
    )


async def distill_tree_search_path(
    state: SearchState,
    *,
    objective: str,
    distillation_agent: object | None = None,
    model: str | None = None,
) -> str:
    report = build_tree_search_trace_report(state, objective=objective)
    if not report.winner_path:
        raise ValueError("TreeSearchStep has no winning path to distill.")
    prompt = build_tree_search_distillation_prompt(
        objective=objective,
        path=report.winner_path,
    )
    agent = distillation_agent
    if agent is None:
        resolved_model = _resolve_tuning_model(model) or DEFAULT_DISTILLATION_MODEL
        agent = make_agent_async(
            resolved_model,
            "You are a prompt distiller. Output only the distilled prompt.",
            str,
        )
    run_fn = getattr(agent, "run", None)
    if callable(run_fn):
        result = await run_fn(prompt)
    elif callable(agent):
        res = agent(prompt)
        result = await res if hasattr(res, "__await__") else res
    else:
        raise TypeError("distillation_agent must be callable or expose run()")
    return str(result)


__all__ = [
    "TreeSearchPathNode",
    "TreeSearchTraceIssue",
    "TreeSearchTraceReport",
    "build_tree_search_trace_report",
    "build_tree_search_tuning_prompt",
    "tune_tree_search_evaluator",
    "build_tree_search_distillation_prompt",
    "distill_tree_search_path",
]
