from __future__ import annotations

import inspect
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Tuple, TypeVar, cast

from ..domain.dsl.conditional import ConditionalStep
from ..domain.dsl.dynamic_router import DynamicParallelRouterStep
from ..domain.dsl.loop import LoopStep
from ..domain.dsl.parallel import ParallelStep
from ..domain.dsl.pipeline import Pipeline
from ..domain.dsl.tree_search import TreeSearchStep
from ..domain.models import BaseModel, PipelineResult
from ..type_definitions.common import JSONObject
from ..utils.hash import stable_digest


def _iter_steps(obj: object) -> Iterable[object]:
    if isinstance(obj, Pipeline):
        for step in obj.steps:
            yield from _iter_steps(step)
        return
    if isinstance(obj, ParallelStep):
        yield obj
        for branch in obj.branches.values():
            yield from _iter_steps(branch)
        return
    if isinstance(obj, ConditionalStep):
        yield obj
        for branch in obj.branches.values():
            yield from _iter_steps(branch)
        if obj.default_branch_pipeline is not None:
            yield from _iter_steps(obj.default_branch_pipeline)
        return
    if isinstance(obj, DynamicParallelRouterStep):
        yield obj
        for branch in obj.branches.values():
            yield from _iter_steps(branch)
        return
    if isinstance(obj, LoopStep):
        yield obj
        body = getattr(obj, "loop_body_pipeline", None)
        if body is not None:
            yield from _iter_steps(body)
        return
    yield obj


PipelineInT = TypeVar("PipelineInT")
PipelineOutT = TypeVar("PipelineOutT")
ContextT = TypeVar("ContextT", bound=BaseModel)


def _iter_agents(pipeline: Pipeline[PipelineInT, PipelineOutT]) -> Iterable[Tuple[str, object]]:
    for step in _iter_steps(pipeline):
        if isinstance(step, TreeSearchStep):
            yield (f"{step.name}.proposer", step.proposer)
            yield (f"{step.name}.evaluator", step.evaluator)
            continue
        if isinstance(step, DynamicParallelRouterStep):
            yield (f"{step.name}.router", step.router_agent)
        agent = getattr(step, "agent", None)
        if agent is not None:
            yield (getattr(step, "name", "<unnamed>"), agent)


def _unwrap_agent(agent: object) -> object:
    return getattr(agent, "_agent", agent)


def _extract_skill_id(agent: object) -> str | None:
    if isinstance(agent, dict):
        skill_id = agent.get("id") or agent.get("path")
        return str(skill_id) if skill_id else None
    if isinstance(agent, str):
        return agent
    skill_id = getattr(agent, "__flujo_skill_id__", None)
    if isinstance(skill_id, str) and skill_id:
        return skill_id
    unwrapped = _unwrap_agent(agent)
    skill_id = getattr(unwrapped, "__flujo_skill_id__", None)
    if isinstance(skill_id, str) and skill_id:
        return skill_id
    return None


def _extract_prompt(agent: object) -> str | None:
    for attr in ("_original_system_prompt", "system_prompt_template", "system_prompt"):
        val = getattr(agent, attr, None)
        if isinstance(val, str) and val:
            return val
    unwrapped = _unwrap_agent(agent)
    for attr in ("_original_system_prompt", "system_prompt_template", "system_prompt"):
        val = getattr(unwrapped, attr, None)
        if isinstance(val, str) and val:
            return val
    return None


def _source_fingerprint(agent: object) -> str | None:
    target = agent
    if hasattr(agent, "__call__") and not inspect.isfunction(agent):
        target = agent.__call__
    try:
        return inspect.getsource(cast(Any, target))
    except Exception:
        return None


def _hash_skill(agent: object, skill_id: str) -> str:
    payload = {
        "id": skill_id,
        "module": getattr(agent, "__module__", None),
        "qualname": getattr(agent, "__qualname__", None),
        "source": _source_fingerprint(agent),
    }
    return stable_digest(payload)


def _normalize_payload(value: object) -> object:
    if isinstance(value, BaseModel):
        try:
            return value.model_dump(mode="json")
        except Exception:
            return value.model_dump()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, (list, tuple)):
        return [_normalize_payload(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalize_payload(v) for k, v in value.items()}
    return value


def build_lockfile_data(
    *,
    pipeline: Pipeline[PipelineInT, PipelineOutT],
    result: PipelineResult[ContextT],
    pipeline_name: str | None,
    pipeline_version: str,
    pipeline_id: str,
    run_id: str | None,
) -> JSONObject:
    skills: list[JSONObject] = []
    prompts: list[JSONObject] = []

    for step_name, agent in _iter_agents(pipeline):
        skill_id = _extract_skill_id(agent)
        if skill_id:
            skills.append(
                {
                    "step": step_name,
                    "skill_id": skill_id,
                    "hash": _hash_skill(_unwrap_agent(agent), skill_id),
                }
            )
        prompt = _extract_prompt(agent)
        if prompt:
            prompts.append(
                {
                    "step": step_name,
                    "hash": stable_digest(prompt),
                }
            )

    data: JSONObject = {
        "schema_version": 1,
        "pipeline": {
            "name": pipeline_name,
            "version": pipeline_version,
            "id": pipeline_id,
        },
        "run": {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": getattr(result, "status", None),
        },
        "skills": skills,
        "prompts": prompts,
        "result": _normalize_payload(
            {
                "success": getattr(result, "success", False),
                "final_output": getattr(result, "output", None),
                "steps": [
                    {
                        "name": sr.name,
                        "success": sr.success,
                        "feedback": sr.feedback,
                    }
                    for sr in getattr(result, "step_history", []) or []
                ],
            }
        ),
    }
    return data


def write_lockfile(
    *,
    path: str | Path,
    pipeline: Pipeline[PipelineInT, PipelineOutT],
    result: PipelineResult[ContextT],
    pipeline_name: str | None,
    pipeline_version: str,
    pipeline_id: str,
    run_id: str | None,
) -> Path:
    data = build_lockfile_data(
        pipeline=pipeline,
        result=result,
        pipeline_name=pipeline_name,
        pipeline_version=pipeline_version,
        pipeline_id=pipeline_id,
        run_id=run_id,
    )
    target = Path(path)
    target.write_text(
        json.dumps(data, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target
