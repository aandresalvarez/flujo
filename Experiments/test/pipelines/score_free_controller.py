from __future__ import annotations

from typing import Any, Dict, List

from flujo.agents import make_agent_async
from flujo.domain.dsl import Pipeline, Step

from Experiments.test.app.controllers import OrdinalController, OrdinalControllerConfig
from Experiments.test.app.evidence import copy_badges, update_badges
from Experiments.test.app.models import PlanResponse
from Experiments.test.app.tools import search_local_docs


PLAN_PROMPT = """
You are a planning agent for a decision pipeline.

Given a goal, propose 3 distinct candidate approaches. For each candidate:
- include a short summary (<= 30 words)
- provide 1-3 search queries that could gather evidence
- estimate initial badges with fields:
  - constraints_ok (true/false)
  - support (integer, start at 0 or 1 based on confidence)
  - contradict (integer)
  - quality_avg (0.0-1.0)
  - recency (0.0-1.0)
  - directness (0.0-1.0)
  - verify_hits (integer)
  - cost_tokens (estimated usage)

Return JSON with shape:
{
  "goal": "<echo input goal>",
  "candidates": [
    {
      "id": "plan_a",
      "summary": "...",
      "queries": ["..."],
      "badges": {
        "constraints_ok": true,
        ...
      }
    },
    ...
  ]
}

Only return valid JSON. Do not include markdown code fences.
"""


async def gather_evidence(plan: PlanResponse) -> Dict[str, Any]:
    enriched: List[Dict[str, Any]] = []
    for candidate in plan.candidates:
        badges = copy_badges(candidate.badges or None)
        evidence_bundle: List[Dict[str, Any]] = []
        for query in candidate.queries:
            matches = search_local_docs(query, limit=3)
            evidence_bundle.extend(matches)
            if matches:
                quality = min(1.0, badges.quality_avg + 0.05 * len(matches))
                badges = update_badges(
                    badges,
                    {
                        "support": min(len(matches), 2),
                        "verify_hits": 1,
                        "quality_avg": quality,
                    },
                )
        enriched.append(
            {
                "id": candidate.id,
                "summary": candidate.summary,
                "queries": candidate.queries,
                "badges": badges.to_dict(),
                "evidence": evidence_bundle,
            }
        )
    return {"goal": plan.goal or "", "candidates": enriched}


async def run_ordinal_controller(state: Dict[str, Any]) -> Dict[str, Any]:
    controller = OrdinalController(
        config=OrdinalControllerConfig(
            tournament_k=5,
            max_frontier=len(state.get("candidates", [])) or 1,
            min_support=3,
        )
    )
    result = controller.run(state.get("candidates", []))
    output = {
        "goal": state.get("goal", ""),
        "winner": result.winner,
        "controller_log": result.log,
        "goal_met": result.goal_met,
    }
    return output


async def finalize_output(state: Dict[str, Any]) -> Dict[str, Any]:
    winner = state.get("winner", {})
    return {
        "goal": state.get("goal", ""),
        "selection_summary": winner.get("summary", ""),
        "badges": winner.get("badges", {}),
        "evidence": winner.get("evidence", []),
        "controller_log": state.get("controller_log", []),
        "goal_met": state.get("goal_met", False),
    }


plan_agent = make_agent_async(
    model="openai:gpt-4o-mini",
    system_prompt=PLAN_PROMPT.strip(),
    output_type=PlanResponse,
)

plan_step = Step.model_validate({"name": "Plan", "agent": plan_agent})
gather_step = Step.from_callable(gather_evidence, name="GatherEvidence")
select_step = Step.from_callable(run_ordinal_controller, name="OrdinalController")
finalize_step = Step.from_callable(finalize_output, name="Finalize")

pipeline = Pipeline(steps=[plan_step, gather_step, select_step, finalize_step])

__all__ = ["pipeline"]
