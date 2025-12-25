#!/usr/bin/env python3
"""
Biomedical A* Research Agent Demo

Demonstrates:
- TreeSearchStep for hypothesis exploration
- Backtracking on contradicted branches
- Selecting the best-supported hypothesis path
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from flujo.application.runner import Flujo
from flujo.domain.dsl.tree_search import TreeSearchStep
from flujo.domain.evaluation import EvaluationReport
from flujo.domain.models import PipelineContext
from flujo.testing.utils import gather_result


EXPANSIONS: dict[str, list[dict[str, Any]]] = {
    "root": [
        {"id": "hyp_inflammation", "summary": "IL-6 driven inflammation"},
        {"id": "hyp_metabolic", "summary": "Metabolic stress response"},
        {"id": "hyp_offtarget", "summary": "Off-target ion channel effect"},
    ],
    "hyp_inflammation": [
        {"id": "evidence_cytokine", "summary": "CRP/IL-6 elevation observed"},
        {"id": "evidence_no_signal", "summary": "Cytokine panel shows no signal"},
    ],
    "hyp_metabolic": [
        {"id": "evidence_atp", "summary": "ATP depletion signature detected"},
    ],
    "hyp_offtarget": [
        {"id": "evidence_panel", "summary": "Ion channel panel negative"},
    ],
}

SCORES: dict[str, float] = {
    "hyp_inflammation": 0.6,
    "hyp_metabolic": 0.5,
    "hyp_offtarget": 0.2,
    "evidence_cytokine": 0.95,
    "evidence_no_signal": 0.1,
    "evidence_atp": 0.7,
    "evidence_panel": 0.3,
}

HARD_FAIL: set[str] = {"evidence_no_signal"}


def _parse_candidate(prompt: str) -> dict[str, Any]:
    marker = "Candidate:"
    if marker not in prompt:
        return {"id": "root"}
    candidate_line = prompt.split(marker, 1)[1].strip().splitlines()[0]
    try:
        return json.loads(candidate_line)
    except Exception:
        return {"id": "root"}


async def _proposer(prompt: str) -> list[dict[str, Any]]:
    candidate = _parse_candidate(prompt)
    key = str(candidate.get("id", "root"))
    return EXPANSIONS.get(key, [])


async def _evaluator(prompt: str) -> EvaluationReport:
    candidate = _parse_candidate(prompt)
    key = str(candidate.get("id", ""))
    score = SCORES.get(key, 0.2)
    hard_fail = key in HARD_FAIL
    return EvaluationReport(score=score, hard_fail=hard_fail, metadata={"node": key})


async def main() -> None:
    step = TreeSearchStep(
        name="biomed_astar",
        proposer=_proposer,
        evaluator=_evaluator,
        branching_factor=3,
        beam_width=3,
        max_depth=2,
        goal_score_threshold=0.9,
        require_goal=True,
        candidate_validator=lambda candidate: isinstance(candidate, dict) and "id" in candidate,
    )
    runner = Flujo(step, context_model=PipelineContext, persist_state=False)
    result = await gather_result(
        runner,
        {"id": "root", "summary": "Investigate mechanism for compound X"},
        initial_context_data={"initial_prompt": "Find the best-supported mechanism for compound X"},
    )
    output = getattr(result, "output", None)
    print("Best path:", output)


if __name__ == "__main__":
    asyncio.run(main())
