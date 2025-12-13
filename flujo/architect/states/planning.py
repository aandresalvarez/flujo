from __future__ import annotations
# mypy: disable-error-code=arg-type

import os as _os
from typing import Any, List

from flujo.type_definitions.common import JSONObject

from flujo.architect.states.common import (
    goto,
    make_transition_guard,
    skill_available,
    skill_resolver,
    trace_next_state,
)
from flujo.domain.base_model import BaseModel as _BaseModel
from flujo.domain.dsl import Pipeline, Step
from flujo.exceptions import InfiniteRedirectError, PausedException, PipelineAbortSignal


async def make_plan_from_goal(*_: Any, context: _BaseModel | None = None) -> JSONObject:
    goal = ""
    available: List[JSONObject] | None = None
    try:
        if context is not None:
            goal = str(getattr(context, "user_goal", "") or "")
            available = getattr(context, "available_skills", None)
    except Exception:
        pass
    g = goal.lower()
    chosen: JSONObject
    import re as _re

    url = None
    try:
        m = _re.search(r"https?://\S+", goal)
        if m:
            url = m.group(0)
    except Exception:
        url = None

    save_path: str | None = None
    try:
        m = _re.search(r"save\s+(?:it\s+)?to\s+(?:a\s+)?file\s+(?:named\s+)?([\w\.-/\\]+)", g)
        if m:
            save_path = m.group(1)
        else:
            m2 = _re.search(r"save\s+(?:it\s+)?as\s+([\w\.-/\\]+)", g)
            if m2:
                save_path = m2.group(1)
    except Exception:
        save_path = None

    if ("http" in g or url) and skill_available("flujo.builtins.http_get", available=available):
        params = {"url": url} if url else {}
        chosen = {
            "name": "Fetch URL",
            "purpose": "Fetch the content from the specified URL for downstream processing.",
            "agent": {"id": "flujo.builtins.http_get", "params": params},
        }
        summary = "Fetches content from the specified URL."
    elif ("search" in g or "find" in g) and skill_available(
        "flujo.builtins.web_search", available=available
    ):
        chosen = {
            "name": "Web Search",
            "purpose": "Perform a web search to find information related to the user's goal.",
            "agent": {"id": "flujo.builtins.web_search", "params": {"query": goal}},
        }
        summary = "Performs a web search for the goal text."
    else:
        chosen = {
            "name": "Echo Input",
            "purpose": "Safely echo or stringify the input as a baseline step.",
            "agent": {"id": "flujo.builtins.stringify", "params": {}},
        }
        summary = "Returns the input unchanged."

    plan: List[JSONObject] = [chosen]
    if save_path and skill_available("flujo.builtins.fs_write_file", available=available):
        plan.append(
            {
                "name": "Save To File",
                "purpose": "Persist the previous step's output to a file on disk.",
                "agent": {
                    "id": "flujo.builtins.fs_write_file",
                    "params": {"path": save_path},
                },
            }
        )
    return {"execution_plan": plan, "plan_summary": summary}


async def run_planner_agent(_x: Any = None, *, context: _BaseModel | None = None) -> JSONObject:
    """Call Planner Agent when available; fallback to heuristics.

    Expects planner agent to accept a dict with keys:
      - user_goal, available_skills, project_summary, flujo_schema
    And return an ExecutionPlan-like dict with keys:
      - plan_summary: str, steps: List[{'step_name','purpose'}]
    """
    try:
        reg = skill_resolver()
    except Exception:
        reg = None

    goal = ""
    available = []
    proj = ""
    schema = {}
    try:
        if context is not None:
            goal = str(getattr(context, "user_goal", "") or "")
            available = list(getattr(context, "available_skills", []) or [])
            proj = str(getattr(context, "project_summary", "") or "")
            schema = dict(getattr(context, "flujo_schema", {}) or {})
    except Exception:
        pass

    payload = {
        "user_goal": goal,
        "available_skills": available,
        "project_summary": proj,
        "flujo_schema": schema,
    }

    try:
        disable = str(_os.environ.get("FLUJO_ARCHITECT_AGENTIC_PLANNER", "")).strip().lower()
        if disable in {"0", "false", "no", "off"}:
            raise RuntimeError("Agentic planner explicitly disabled by env var")

        entry = reg.get("flujo.architect.planner") if reg else None
        if entry and isinstance(entry, dict) and entry.get("factory"):
            agent_callable = entry["factory"]()
            result = await agent_callable(payload)
            plan_summary = None
            steps_out: List[JSONObject] = []
            if isinstance(result, dict):
                plan_summary = result.get("plan_summary")
                steps = result.get("steps")
                if isinstance(steps, list):
                    for s in steps:
                        if isinstance(s, dict):
                            nm = s.get("step_name") or s.get("name") or "step"
                            purpose = s.get("purpose") or ""
                            steps_out.append({"name": nm, "purpose": purpose})
            out: JSONObject = {}
            if steps_out:
                out["execution_plan"] = steps_out
            if plan_summary:
                out["plan_summary"] = plan_summary
            return out
    except (PausedException, PipelineAbortSignal, InfiniteRedirectError):
        # Preserve orchestration signals
        raise
    except Exception:
        pass

    return await make_plan_from_goal(context=context)


def build_planning_state() -> Pipeline[Any, Any]:
    """Create Planning pipeline (plan -> visualize -> estimate -> approval transition)."""
    reg = skill_resolver()
    try:
        viz_entry = reg.get("flujo.builtins.visualize_plan")
        if viz_entry and isinstance(viz_entry, dict):
            _viz = viz_entry["factory"]()
            visualize: Step[Any, Any] = Step.from_callable(
                _viz, name="VisualizePlan", updates_context=True
            )
        else:

            async def _viz_fallback(plan: Any) -> JSONObject:
                return {"plan_mermaid_graph": "graph TD"}

            visualize = Step.from_callable(
                _viz_fallback, name="VisualizePlan", updates_context=True
            )
    except Exception:

        async def _viz_fallback(plan: Any) -> JSONObject:
            return {"plan_mermaid_graph": "graph TD"}

        visualize = Step.from_callable(_viz_fallback, name="VisualizePlan", updates_context=True)

    try:
        est_entry = reg.get("flujo.builtins.estimate_plan_cost")
        if est_entry and isinstance(est_entry, dict):
            _est = est_entry["factory"]()
            estimate: Step[Any, Any] = Step.from_callable(
                _est, name="EstimateCost", updates_context=True
            )
        else:

            async def _est_fallback(plan: Any) -> JSONObject:
                return {"plan_estimated_cost_usd": 0.0}

            estimate = Step.from_callable(_est_fallback, name="EstimateCost", updates_context=True)
    except Exception:

        async def _est_fallback(plan: Any) -> JSONObject:
            return {"plan_estimated_cost_usd": 0.0}

        estimate = Step.from_callable(_est_fallback, name="EstimateCost", updates_context=True)

    async def _goto_approval(
        _data: JSONObject | None = None, context: _BaseModel | None = None
    ) -> JSONObject:
        return await goto("PlanApproval", context=context)

    goto_approval_step = Step.from_callable(
        _goto_approval,
        name="GotoApproval",
        updates_context=True,
    )

    return (
        Pipeline.from_step(
            Step.from_callable(run_planner_agent, name="MakePlan", updates_context=True)
        )
        >> visualize
        >> estimate
        >> goto_approval_step
        >> Step.from_callable(
            make_transition_guard("PlanApproval"),
            name="Guard_PlanApproval",
            updates_context=True,
        )
        >> Step.from_callable(
            trace_next_state, name="TraceNextState_PlanApproval", updates_context=True
        )
    )
