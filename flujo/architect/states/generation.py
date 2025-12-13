from __future__ import annotations

import os as _os
from typing import Any, List

from flujo.architect.states.common import normalize_name_from_goal, skill_resolver, telemetry
from flujo.domain.base_model import BaseModel as _BaseModel
from flujo.domain.dsl import MapStep, Pipeline, Step
from flujo.type_definitions.common import JSONObject


async def emit_minimal_yaml(goal: str) -> dict[str, Any]:
    """Return a minimal, valid Flujo YAML blueprint derived from the goal."""
    safe_name = normalize_name_from_goal(goal)
    yaml_text = (
        'version: "0.1"\n'
        f"name: {safe_name}\n"
        "steps:\n"
        "- kind: step\n"
        "  name: Echo Input\n"
        "  agent:\n"
        "    id: flujo.builtins.stringify\n"
        "    params: {}\n"
    )
    return {
        "generated_yaml": yaml_text,
        "yaml_text": yaml_text,
        "next_state": "Finalization",
    }


async def generate_yaml_from_plan(
    _x: Any = None, *, context: _BaseModel | None = None
) -> JSONObject:
    try:
        goal = getattr(context, "user_goal", None) if context is not None else None
    except Exception:
        goal = None
    name = normalize_name_from_goal(goal)

    steps_yaml: List[str] = []
    try:
        plan = getattr(context, "execution_plan", None) if context is not None else None
        if not isinstance(plan, list) or not plan:
            plan = [
                {
                    "name": "Echo Input",
                    "purpose": "Safely echo or stringify the input as a baseline step.",
                    "agent": {"id": "flujo.builtins.stringify", "params": {}},
                }
            ]
        import yaml as _yaml

        for s in plan:
            if not isinstance(s, dict):
                continue
            sname = s.get("name") or "step"
            agent = s.get("agent")
            if not isinstance(agent, dict):
                agent = {}
            sid = agent.get("id") or "flujo.builtins.stringify"
            params = agent.get("params") or {}
            step_dict = {
                "kind": "step",
                "name": sname,
                "agent": {"id": sid, "params": params},
            }
            steps_yaml.append(_yaml.safe_dump(step_dict, sort_keys=False).strip())

        steps_block = "\n".join(
            [
                "- " + line if i == 0 else "  " + line
                for block in steps_yaml
                for i, line in enumerate(block.splitlines())
            ]
        )
        yaml_text = f'version: "0.1"\nname: {name}\nsteps:\n{steps_block}\n'
    except Exception:
        yaml_text = f'version: "0.1"\nname: {name}\nsteps: []\n'

    try:
        if context is not None and hasattr(context, "next_state"):
            context.next_state = "Validation"
    except Exception as e:
        telemetry().error(f"[ArchitectSM] Failed to update context next_state: {e}")

    try:
        if context is not None:
            if hasattr(context, "yaml_text"):
                setattr(context, "yaml_text", yaml_text)
            if hasattr(context, "generated_yaml"):
                setattr(context, "generated_yaml", yaml_text)
    except Exception as e:
        telemetry().error(f"[ArchitectSM] GenerateYAML failed to set context fields: {e}")

    result = {
        "generated_yaml": yaml_text,
        "yaml_text": yaml_text,
    }
    try:
        telemetry().info(f"[ArchitectSM] GenerateYAML returning: {result}")
    except Exception:
        pass
    return result


async def prepare_for_map(_x: Any = None, *, context: _BaseModel | None = None) -> JSONObject:
    """Prepare `prepared_steps_for_mapping` from `execution_plan`."""
    items: List[JSONObject] = []
    try:
        plan = getattr(context, "execution_plan", None) if context is not None else None
        if isinstance(plan, list):
            for s in plan:
                if not isinstance(s, dict):
                    continue
                if "purpose" in s and "name" in s and "agent" not in s:
                    items.append({"step_name": s.get("name"), "purpose": s.get("purpose")})
                elif "agent" in s:
                    nm = s.get("name") or "step"
                    agent = s.get("agent") or {}
                    items.append(
                        {
                            "step_name": nm,
                            "purpose": s.get("purpose", ""),
                            "preselected_agent": agent,
                        }
                    )
    except Exception:
        items = []
    if not items:
        items = [
            {
                "step_name": "Echo Input",
                "purpose": "Safely echo or stringify the input as a baseline step.",
            }
        ]
    return {"prepared_steps_for_mapping": items}


async def match_one_tool(step_item: JSONObject, *, context: _BaseModel | None = None) -> JSONObject:
    """Run ToolMatcher agent for a single planned step; resilient with safe fallback."""
    from flujo.exceptions import InfiniteRedirectError, PausedException, PipelineAbortSignal

    try:
        step_name = step_item.get("step_name") if isinstance(step_item, dict) else None
        purpose = step_item.get("purpose") if isinstance(step_item, dict) else None
        preselected = step_item.get("preselected_agent") if isinstance(step_item, dict) else None
        available: List[JSONObject] = []
        try:
            if context is not None:
                available = list(getattr(context, "available_skills", []) or [])
        except Exception:
            available = []

        if isinstance(preselected, dict) and preselected.get("id"):
            return {
                "step_name": step_name or "step",
                "chosen_agent_id": preselected.get("id"),
                "agent_params": preselected.get("params") or {},
            }

        try:
            disable = (
                str(_os.environ.get("FLUJO_ARCHITECT_AGENTIC_TOOLMATCHER", "")).strip().lower()
            )
            if disable in {"0", "false", "no", "off"}:
                raise RuntimeError("Agentic tool matcher explicitly disabled by env var")

            reg = skill_resolver()
            entry = reg.get("flujo.architect.tool_matcher") if reg else None
            if entry and isinstance(entry, dict) and entry.get("factory"):
                agent_callable = entry["factory"]()
                payload = {
                    "step_name": step_name,
                    "purpose": purpose,
                    "available_skills": available,
                }
                res = await agent_callable(payload)
                if isinstance(res, dict) and res.get("chosen_agent_id"):
                    return {
                        "step_name": res.get("step_name") or step_name or "step",
                        "chosen_agent_id": res.get("chosen_agent_id"),
                        "agent_params": res.get("agent_params") or {},
                    }
        except (PausedException, PipelineAbortSignal, InfiniteRedirectError):
            # Preserve orchestration control-flow signals
            raise
        except Exception as e:
            try:
                message = (
                    f"[ArchitectSM] ToolMatcher agent failed for step '{step_name}': {e}. "
                    "Falling back."
                )
                telemetry().warning(message)
            except Exception:
                pass

        return {
            "step_name": step_name or "step",
            "chosen_agent_id": "flujo.builtins.stringify",
            "agent_params": {},
        }
    except (PausedException, PipelineAbortSignal, InfiniteRedirectError):
        # Allow orchestrator control-flow to bubble up
        raise
    except Exception as e:
        try:
            telemetry().warning(
                f"[ArchitectSM] ToolMatcher unexpected error for step: {e}. Using safe default."
            )
        except Exception:
            pass
        return {
            "step_name": (step_item.get("step_name") if isinstance(step_item, dict) else None)
            or "step",
            "chosen_agent_id": "flujo.builtins.stringify",
            "agent_params": {},
        }


async def collect_tool_selections(
    result_list: Any, *, context: _BaseModel | None = None
) -> JSONObject:
    results = result_list if isinstance(result_list, list) else []
    return {"tool_selections": results}


async def generate_yaml_from_tool_selections(
    _x: Any = None, *, context: _BaseModel | None = None
) -> JSONObject:
    """YAML writer using agent when available, with robust fallback."""
    from flujo.exceptions import InfiniteRedirectError, PausedException, PipelineAbortSignal

    goal = None
    flujo_schema: JSONObject = {}
    selections: List[JSONObject] = []
    try:
        if context is not None:
            goal = getattr(context, "user_goal", None)
            flujo_schema = getattr(context, "flujo_schema", {}) or {}
            selections = list(getattr(context, "tool_selections", []) or [])
    except Exception:
        pass

    try:
        disable = str(_os.environ.get("FLUJO_ARCHITECT_AGENTIC_YAMLWRITER", "")).strip().lower()
        if disable in {"0", "false", "no", "off"}:
            raise RuntimeError("Agentic YAML writer explicitly disabled by env var")

        reg = skill_resolver()
        entry = reg.get("flujo.architect.yaml_writer") if reg else None
        if entry and isinstance(entry, dict) and entry.get("factory"):
            agent_callable = entry["factory"]()
            payload = {
                "user_goal": goal,
                "tool_selections": selections,
                "flujo_schema": flujo_schema,
            }
            res = await agent_callable(payload)
            if isinstance(res, dict) and isinstance(res.get("generated_yaml"), str):
                yaml_text = res["generated_yaml"]
                try:
                    if context is not None and hasattr(context, "next_state"):
                        context.next_state = "Validation"
                except Exception:
                    pass
                try:
                    if context is not None and hasattr(context, "yaml_text"):
                        setattr(context, "yaml_text", yaml_text)
                    if context is not None and hasattr(context, "generated_yaml"):
                        setattr(context, "generated_yaml", yaml_text)
                except Exception:
                    pass
                return {"generated_yaml": yaml_text, "yaml_text": yaml_text}
    except Exception:
        pass

    if selections:
        try:
            name = normalize_name_from_goal(str(goal) if goal is not None else None)
            import yaml as _yaml

            steps_yaml: List[str] = []
            for sel in selections:
                if hasattr(sel, "chosen_agent_id"):
                    sid = getattr(sel, "chosen_agent_id", None)
                    params = getattr(sel, "agent_params", None)
                    sname = getattr(sel, "step_name", None)
                else:
                    sid = sel.get("chosen_agent_id") if isinstance(sel, dict) else None
                    params = sel.get("agent_params") if isinstance(sel, dict) else None
                    sname = sel.get("step_name") if isinstance(sel, dict) else None
                if not sid:
                    sid = "flujo.builtins.stringify"
                if not isinstance(params, dict):
                    params = {}
                if not sname:
                    sname = "Step"
                step_dict = {
                    "kind": "step",
                    "name": sname,
                    "agent": {"id": sid, "params": params},
                }
                steps_yaml.append(_yaml.safe_dump(step_dict, sort_keys=False).strip())

            steps_block = "\n".join(
                [
                    "- " + line if i == 0 else "  " + line
                    for block in steps_yaml
                    for i, line in enumerate(block.splitlines())
                ]
            )
            yaml_text = f'\nversion: "0.1"\nname: {name}\nsteps:\n{steps_block}\n'

            try:
                if context is not None and hasattr(context, "next_state"):
                    context.next_state = "Validation"
            except Exception:
                pass
            try:
                if context is not None and hasattr(context, "yaml_text"):
                    setattr(context, "yaml_text", yaml_text)
                if context is not None and hasattr(context, "generated_yaml"):
                    setattr(context, "generated_yaml", yaml_text)
            except Exception:
                pass
            return {"generated_yaml": yaml_text, "yaml_text": yaml_text}
        except (PausedException, PipelineAbortSignal, InfiniteRedirectError):
            # Preserve orchestration signals raised while building YAML
            raise
        except Exception as e:
            telemetry().info(f"[Architect] Exception in tool selections processing: {e}")

    telemetry().info("[Architect] No tool selections, falling back to plan-based generator")
    return await generate_yaml_from_plan(None, context=context)


def build_generation_state() -> Pipeline[Any, Any]:
    """Tool matching + YAML writer pipeline."""
    tool_match_body = Pipeline.from_step(
        Step.from_callable(match_one_tool, name="ToolMatcher", updates_context=False)
    )
    map_tools: MapStep[Any] = MapStep(
        name="MapToolMatcher",
        iterable_input="prepared_steps_for_mapping",
        pipeline_to_run=tool_match_body,
    )
    return (
        Pipeline.from_step(
            Step.from_callable(prepare_for_map, name="PrepareForMap", updates_context=True)
        )
        >> map_tools
        >> Step.from_callable(
            collect_tool_selections, name="CollectToolSelections", updates_context=True
        )
        >> Step.from_callable(
            generate_yaml_from_tool_selections, name="GenerateYAML", updates_context=True
        )
    )
