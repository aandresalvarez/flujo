from __future__ import annotations

from typing import List

from flujo.domain.models import PipelineContext
from flujo.type_definitions.common import JSONObject


async def clarify(goal: str | dict | None, *, context: PipelineContext | None = None) -> JSONObject:
    """Produce a minimal cohort definition from the initial goal.

    In real pipelines this would call an LLM or validator. Here we synthesize a structure.
    """
    if isinstance(goal, dict):
        basis = goal
    else:
        basis = {"goal": str(goal or getattr(context, "initial_prompt", ""))}
    cohort_definition = {"name": basis.get("name") or "demo", "criteria": ["age > 18"]}
    return {"import_artifacts": {"cohort_definition": cohort_definition}}


async def discover_concepts(_data: object, *, context: PipelineContext | None = None) -> JSONObject:
    """Derive concept sets from the cohort definition in context."""
    assert context is not None
    cd = context.import_artifacts.get("cohort_definition") or {}
    name = cd.get("name", "demo") if isinstance(cd, dict) else "demo"
    concept_sets: List[int] = [1, 2, 3] if name else []
    return {"import_artifacts": {"concept_sets": concept_sets}}


async def build_sql(_data: object, *, context: PipelineContext | None = None) -> JSONObject:
    """Build a final SQL string from cohort_definition and concept_sets in context."""
    assert context is not None
    cd = context.import_artifacts.get("cohort_definition")
    cs = context.import_artifacts.get("concept_sets") or []
    final_sql = f"-- cohorts: {str(cd)}; concepts: {len(cs)}"
    return {"import_artifacts": {"final_sql": final_sql}}


async def accept_review(_data: object, *, context: PipelineContext | None = None) -> JSONObject:
    """Map the last human response from HITL into import_artifacts.cohort_definition.

    If the user pasted JSON, try to parse `{ "name": ..., "criteria": [...] }`.
    Otherwise, store a simple structured wrapper.
    """
    import json

    assert context is not None
    # Prefer the last explicit human response
    human = None
    try:
        if context.hitl_history:
            human = context.hitl_history[-1].human_response
    except Exception:
        human = None
    # Fallback to compact steps map value
    if human is None:
        try:
            steps_map = context.step_outputs
            human = steps_map.get("review")
        except Exception:
            human = None

    cd: JSONObject
    if isinstance(human, (dict, list)):
        cd = {"name": "user", "criteria": ["from_json"]}
    else:
        text = str(human or "")
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                cd = obj  # user provided a structured definition
            else:
                cd = {"name": "user", "criteria": [text]}
        except Exception:
            cd = {"name": "user", "criteria": [text]}
    return {"import_artifacts": {"cohort_definition": cd}}
