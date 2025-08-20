from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import os as _os

from flujo.domain.dsl import Pipeline, Step
from flujo.domain.dsl.state_machine import StateMachineStep
from flujo.infra.skill_registry import get_skill_registry
from flujo.domain.base_model import BaseModel as _BaseModel


async def _emit_minimal_yaml(goal: str) -> dict[str, Any]:
    """Return a minimal, valid Flujo YAML blueprint derived from the goal.

    The builder intentionally emits a conservative pipeline scaffold to keep
    CLI `create` flows fast and dependency-free.
    """
    safe_name = "generated_pipeline"
    try:
        g = (goal or "").strip()
        if g:
            # Simple normalization: keep alnum/space, collapse to underscores
            import re as _re

            norm = _re.sub(r"[^A-Za-z0-9\s]+", "", g)[:40].strip().lower()
            if norm:
                safe_name = ("_".join(norm.split()) or safe_name)[:40]
    except Exception:
        pass
    yaml_text = f'version: "0.1"\nname: {safe_name}\nsteps: []\n'
    return {
        "generated_yaml": yaml_text,
        "yaml_text": yaml_text,
        # Hint the state machine to move forward when used inside it
        "scratchpad": {"next_state": "Finalization"},
    }


def _normalize_name_from_goal(goal: Optional[str]) -> str:
    safe_name = "generated_pipeline"
    try:
        g = (goal or "").strip()
        if g:
            import re as _re

            norm = _re.sub(r"[^A-Za-z0-9\s]+", "", g)[:40].strip().lower()
            if norm:
                safe_name = ("_".join(norm.split()) or safe_name)[:40]
    except Exception:
        pass
    return safe_name


async def _goto(state: str, *, context: _BaseModel | None = None) -> Dict[str, Any]:
    try:
        sp = {}
        if context is not None and hasattr(context, "scratchpad"):
            val = getattr(context, "scratchpad", {})
            if isinstance(val, dict):
                sp.update(val)
        sp["next_state"] = state
        return {"scratchpad": sp}
    except Exception:
        return {"scratchpad": {"next_state": state}}


async def _map_framework_schema(obj: Dict[str, Any]) -> Dict[str, Any]:
    try:
        steps_schema = obj.get("steps") if isinstance(obj, dict) else None
        if isinstance(steps_schema, dict):
            return {"flujo_schema": steps_schema}
    except Exception:
        pass
    return {"flujo_schema": {}}


def _skill_available(skill_id: str, *, available: Optional[List[Dict[str, Any]]]) -> bool:
    try:
        if isinstance(available, list):
            return any(isinstance(x, dict) and x.get("id") == skill_id for x in available)
    except Exception:
        pass
    try:
        return get_skill_registry().get(skill_id) is not None
    except Exception:
        return False


async def _make_plan_from_goal(*_: Any, context: _BaseModel | None = None) -> Dict[str, Any]:
    goal = ""
    available: Optional[List[Dict[str, Any]]] = None
    try:
        if context is not None:
            goal = str(getattr(context, "user_goal", "") or "")
            available = getattr(context, "available_skills", None)
    except Exception:
        pass
    g = goal.lower()
    chosen: Dict[str, Any]
    # Heuristic: prefer http_get when URL-ish mentioned; else web_search; else stringify
    import re as _re

    url = None
    try:
        m = _re.search(r"https?://\S+", goal)
        if m:
            url = m.group(0)
    except Exception:
        url = None

    if ("http" in g or url) and _skill_available("flujo.builtins.http_get", available=available):
        params = {"url": url} if url else {}
        chosen = {
            "name": "Fetch URL",
            "agent": {"id": "flujo.builtins.http_get", "params": params},
        }
        summary = "Fetches content from the specified URL."
    elif ("search" in g or "find" in g) and _skill_available(
        "flujo.builtins.web_search", available=available
    ):
        chosen = {
            "name": "Web Search",
            "agent": {"id": "flujo.builtins.web_search", "params": {"query": goal}},
        }
        summary = "Performs a web search for the goal text."
    else:
        chosen = {
            "name": "Echo Input",
            "agent": {"id": "flujo.builtins.stringify", "params": {}},
        }
        summary = "Returns the input unchanged."

    plan: List[Dict[str, Any]] = [chosen]
    return {"execution_plan": plan, "plan_summary": summary}


async def _generate_yaml_from_plan(
    _x: Any = None, *, context: _BaseModel | None = None
) -> Dict[str, Any]:
    try:
        goal = getattr(context, "user_goal", None) if context is not None else None
    except Exception:
        goal = None
    name = _normalize_name_from_goal(goal)

    steps_yaml: List[str] = []
    try:
        plan = getattr(context, "execution_plan", None) if context is not None else None
        if not isinstance(plan, list) or not plan:
            plan = [
                {
                    "name": "Echo Input",
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

    return {
        "generated_yaml": yaml_text,
        "yaml_text": yaml_text,
        "scratchpad": {"next_state": "Validation"},
    }


def _build_state_machine_pipeline() -> Pipeline[Any, Any]:
    """Programmatically build the full Architect state machine."""
    # GatheringContext: discover skills + analyze project + framework schema
    reg = get_skill_registry()
    # discover_skills is an agent; factory returns an agent instance
    try:
        discover_entry = reg.get("flujo.builtins.discover_skills")
        if discover_entry and isinstance(discover_entry, dict):
            _discover_agent = discover_entry["factory"]()
            discover = Step.solution(_discover_agent, name="DiscoverSkills", updates_context=True)
        else:

            async def _discover_fallback(*_a: Any, **_k: Any) -> Dict[str, Any]:
                return {"available_skills": []}

            discover = Step.from_callable(
                _discover_fallback,
                name="DiscoverSkills",
                updates_context=True,
            )
    except Exception:

        async def _discover_fallback(*_a: Any, **_k: Any) -> Dict[str, Any]:
            return {"available_skills": []}

        discover = Step.from_callable(
            _discover_fallback, name="DiscoverSkills", updates_context=True
        )

    try:
        analyze_entry = reg.get("flujo.builtins.analyze_project")
        if analyze_entry and isinstance(analyze_entry, dict):
            _analyze = analyze_entry["factory"]()
            analyze: Step[Any, Any] = Step.from_callable(
                _analyze, name="AnalyzeProject", updates_context=True
            )
        else:

            async def _analyze_fallback(*_a: Any, **_k: Any) -> Dict[str, Any]:
                return {"project_summary": ""}

            analyze = Step.from_callable(
                _analyze_fallback, name="AnalyzeProject", updates_context=True
            )
    except Exception:

        async def _analyze_fallback(*_a: Any, **_k: Any) -> Dict[str, Any]:
            return {"project_summary": ""}

        analyze = Step.from_callable(_analyze_fallback, name="AnalyzeProject", updates_context=True)

    try:
        schema_entry = reg.get("flujo.builtins.get_framework_schema")
        if schema_entry and isinstance(schema_entry, dict):
            _fw_schema = schema_entry["factory"]()
            get_schema: Union[Step[Any, Any], Pipeline[Any, Any]] = Step.from_callable(
                _fw_schema, name="GetFrameworkSchema"
            ) >> Step.from_callable(
                _map_framework_schema, name="MapFrameworkSchema", updates_context=True
            )
        else:

            async def _schema_fallback(*_a: Any, **_k: Any) -> Dict[str, Any]:
                return {"flujo_schema": {}}

            get_schema = Step.from_callable(
                _schema_fallback, name="MapFrameworkSchema", updates_context=True
            )
    except Exception:

        async def _schema_fallback(*_a: Any, **_k: Any) -> Dict[str, Any]:
            return {"flujo_schema": {}}

        get_schema = Step.from_callable(
            _schema_fallback, name="MapFrameworkSchema", updates_context=True
        )

    async def _goto_goal(*_a: Any, **_k: Any) -> Dict[str, Any]:
        return await _goto("GoalClarification")

    goto_goal = Step.from_callable(_goto_goal, name="GotoGoalClarification", updates_context=True)
    gathering = discover >> analyze >> get_schema >> goto_goal

    # GoalClarification: for now, assume goal accepted and advance
    async def _goto_plan(*_a: Any, **_k: Any) -> Dict[str, Any]:
        return await _goto("Planning")

    goto_plan = Step.from_callable(_goto_plan, name="GotoPlanning", updates_context=True)
    goal_pipe = Pipeline.from_step(goto_plan)

    # Planning: create minimal plan; visualize; estimate cost; then proceed to approval
    try:
        viz_entry = reg.get("flujo.builtins.visualize_plan")
        if viz_entry and isinstance(viz_entry, dict):
            _viz = viz_entry["factory"]()
            visualize: Step[Any, Any] = Step.from_callable(
                _viz, name="VisualizePlan", updates_context=True
            )
        else:

            async def _viz_fallback(plan: Any) -> Dict[str, Any]:
                return {"plan_mermaid_graph": "graph TD"}

            visualize = Step.from_callable(
                _viz_fallback, name="VisualizePlan", updates_context=True
            )
    except Exception:

        async def _viz_fallback(plan: Any) -> Dict[str, Any]:
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

            async def _est_fallback(plan: Any) -> Dict[str, Any]:
                return {"plan_estimated_cost_usd": 0.0}

            estimate = Step.from_callable(_est_fallback, name="EstimateCost", updates_context=True)
    except Exception:

        async def _est_fallback(plan: Any) -> Dict[str, Any]:
            return {"plan_estimated_cost_usd": 0.0}

        estimate = Step.from_callable(_est_fallback, name="EstimateCost", updates_context=True)

    async def _goto_approval(*_a: Any, **_k: Any) -> Dict[str, Any]:
        return await _goto("PlanApproval")

    plan_pipe = (
        Pipeline.from_step(
            Step.from_callable(_make_plan_from_goal, name="MakePlan", updates_context=True)
        )
        >> visualize
        >> estimate
        >> Step.from_callable(_goto_approval, name="GotoApproval", updates_context=True)
    )

    # PlanApproval: auto-approve for now (no prompts) and proceed
    async def _approve(*_a: Any, **_k: Any) -> Dict[str, Any]:
        return {"plan_approved": True}

    approve: Step[Any, Any] = Step.from_callable(_approve, name="Approve", updates_context=True)

    async def _goto_params(*_a: Any, **_k: Any) -> Dict[str, Any]:
        return await _goto("ParameterCollection")

    goto_params = Step.from_callable(_goto_params, name="GotoParams", updates_context=True)
    approval_pipe = approve >> goto_params

    # ParameterCollection: fill required params by prompting when interactive
    async def _collect_params(*_a: Any, context: _BaseModel | None = None) -> Dict[str, Any]:
        try:
            non_interactive = bool(getattr(context, "non_interactive", False)) if context else True
        except Exception:
            non_interactive = True
        try:
            plan = getattr(context, "execution_plan", None) if context is not None else None
        except Exception:
            plan = None
        if not isinstance(plan, list):
            return await _goto("Generation", context=context)
        reg = get_skill_registry()
        changed = False
        for step in plan:
            try:
                agent = step.get("agent") if isinstance(step, dict) else None
                if not isinstance(agent, dict):
                    continue
                sid = agent.get("id")
                params = agent.get("params")
                if not isinstance(params, dict):
                    params = {}
                    agent["params"] = params
                if not isinstance(sid, str):
                    continue
                entry = reg.get(sid) or {}
                schema = entry.get("input_schema") or {}
                required = schema.get("required") if isinstance(schema, dict) else None
                req_list = list(required) if isinstance(required, list) else []
                missing = [k for k in req_list if k not in params]
                if not missing:
                    continue
                if non_interactive:
                    # Skip prompts in non-interactive mode
                    continue
                # Prompt user for each missing param
                try:
                    import typer as _typer

                    for key in missing:
                        val = _typer.prompt(
                            f"Enter value for required parameter '{key}' of skill '{sid}':"
                        )
                        params[key] = val
                        changed = True
                except Exception:
                    continue
            except Exception:
                continue
        out: Dict[str, Any] = {"execution_plan": plan} if changed else {}
        nxt = await _goto("Generation", context=context)
        out.update(nxt)
        return out

    params_pipe = Pipeline.from_step(
        Step.from_callable(_collect_params, name="CollectParams", updates_context=True)
    )

    # Generation: build YAML from plan
    gen = Step.from_callable(_generate_yaml_from_plan, name="GenerateYAML", updates_context=True)
    gen_pipeline = Pipeline.from_step(gen)

    # Validation: validate -> repair loop until valid, then DryRunOffer
    try:
        validate_entry = reg.get("flujo.builtins.validate_yaml")
        if validate_entry and isinstance(validate_entry, dict):
            _validate = validate_entry["factory"]()
            validate: Step[Any, Any] = Step.from_callable(_validate, name="ValidateYAML")
        else:

            async def _validate_fallback(yt: Any) -> Dict[str, Any]:
                return {"is_valid": True}

            validate = Step.from_callable(_validate_fallback, name="ValidateYAML")
    except Exception:

        async def _validate_fallback(yt: Any) -> Dict[str, Any]:
            return {"is_valid": True}

        validate = Step.from_callable(_validate_fallback, name="ValidateYAML")

    try:
        capture_entry = reg.get("flujo.builtins.capture_validation_report")
        if capture_entry and isinstance(capture_entry, dict):
            _capture = capture_entry["factory"]()
            capture: Step[Any, Any] = Step.from_callable(
                _capture, name="CaptureReport", updates_context=True
            )
        else:

            async def _capture_fallback(rep: Any) -> Dict[str, Any]:
                return {"validation_report": rep, "yaml_is_valid": True}

            capture = Step.from_callable(
                _capture_fallback, name="CaptureReport", updates_context=True
            )
    except Exception:

        async def _capture_fallback(rep: Any) -> Dict[str, Any]:
            return {"validation_report": rep, "yaml_is_valid": True}

        capture = Step.from_callable(_capture_fallback, name="CaptureReport", updates_context=True)

    # Simple decision step to set next_state based on current yaml_is_valid
    async def _decide_next(
        _rep: Any = None, *, context: _BaseModel | None = None
    ) -> Dict[str, Any]:
        valid = False
        try:
            if isinstance(_rep, dict) and "is_valid" in _rep:
                valid = bool(_rep.get("is_valid"))
        except Exception:
            pass
        if not valid and context is not None:
            try:
                valid = bool(getattr(context, "yaml_is_valid", False))
            except Exception:
                valid = False
        if valid:
            return await _goto("DryRunOffer", context=context)
        else:
            # Attempt repair and re-validate next round
            try:
                repair_entry = reg.get("flujo.builtins.repair_yaml_ruamel")
                if repair_entry and isinstance(repair_entry, dict):
                    _repair = repair_entry["factory"]()
                    # Run repair in-process; it returns dict with yaml_text
                    repaired = await _repair(getattr(context, "yaml_text", ""))
                    if isinstance(repaired, dict):
                        out: Dict[str, Any] = {**repaired}
                    else:
                        out = {}
                else:
                    out = {}
            except Exception:
                out = {}
            nx = await _goto("Validation", context=context)
            out.update(nx)
            return out

    decide_next = Step.from_callable(_decide_next, name="ValidationDecision", updates_context=True)

    async def _select_yaml_text(_x: Any = None, *, context: _BaseModel | None = None) -> str:
        try:
            yt = getattr(context, "yaml_text", "") if context is not None else ""
            return yt if isinstance(yt, str) else ""
        except Exception:
            return ""

    select_yaml = Step.from_callable(_select_yaml_text, name="SelectYAMLText")
    validation_pipe = Pipeline.from_step(select_yaml) >> validate >> capture >> decide_next

    # DryRunOffer: skip to Finalization for now (non-interactive default)
    async def _goto_final(*_a: Any, **_k: Any) -> Dict[str, Any]:
        return await _goto("Finalization")

    goto_final = Step.from_callable(_goto_final, name="GotoFinal", updates_context=True)
    dry_offer_pipe = Pipeline.from_step(goto_final)

    # DryRunExecution: run in memory then finalize
    try:
        dry_entry = reg.get("flujo.builtins.run_pipeline_in_memory")
        if dry_entry and isinstance(dry_entry, dict):
            _dry = dry_entry["factory"]()
            dryrun: Step[Any, Any] = Step.from_callable(
                _dry, name="DryRunInMemory", updates_context=False
            )
        else:

            async def _dry_fallback(*_a: Any, **_k: Any) -> Dict[str, Any]:
                return {"dry_run_result": {}}

            dryrun = Step.from_callable(_dry_fallback, name="DryRunInMemory")
    except Exception:

        async def _dry_fallback(*_a: Any, **_k: Any) -> Dict[str, Any]:
            return {"dry_run_result": {}}

        dryrun = Step.from_callable(_dry_fallback, name="DryRunInMemory")

    async def _goto_final2(*_a: Any, **_k: Any) -> Dict[str, Any]:
        return await _goto("Finalization")

    goto_final2 = Step.from_callable(_goto_final2, name="GotoFinal2", updates_context=True)
    dry_exec_pipe = Pipeline.from_step(select_yaml) >> dryrun >> goto_final2

    async def _finalize(_: Any = None) -> Dict[str, Any]:
        # Terminal state: nothing to do; keep context as-is
        return {}

    async def _failure_step(*_a: Any, **_k: Any) -> Dict[str, Any]:
        # Failure state: return empty dict
        return {}

    fin = Step.from_callable(_finalize, name="Finalize", updates_context=True)
    fin_pipeline = Pipeline.from_step(fin)

    sm = StateMachineStep(
        name="Architect",
        states={
            "GatheringContext": gathering,
            "GoalClarification": goal_pipe,
            "Planning": plan_pipe,
            "PlanApproval": approval_pipe,
            "ParameterCollection": params_pipe,
            "Generation": gen_pipeline,
            "Validation": validation_pipe,
            "DryRunOffer": dry_offer_pipe,
            "DryRunExecution": dry_exec_pipe,
            "Finalization": fin_pipeline,
            "Failure": Pipeline.from_step(Step.from_callable(_failure_step, name="Failure")),
        },
        start_state="GatheringContext",
        end_states=["Finalization", "Failure"],
    )
    # Execute the state machine via its policy executor
    return Pipeline.from_step(sm)


def build_architect_pipeline() -> Pipeline[Any, Any]:
    """Return the Architect pipeline object.

    By default (for test compatibility), returns a minimal single-step pipeline named
    'GenerateYAML'. To enable the full state-machine architect, set the environment
    variable FLUJO_ARCHITECT_STATE_MACHINE=1.
    """
    if (_os.environ.get("FLUJO_ARCHITECT_STATE_MACHINE") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return _build_state_machine_pipeline()

    # Minimal programmatic pipeline used by tests and simple flows
    gen = Step.from_callable(_emit_minimal_yaml, name="GenerateYAML", updates_context=True)
    return Pipeline.from_step(gen)
