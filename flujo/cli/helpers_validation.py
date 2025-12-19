from __future__ import annotations

import os
import re
from typing import Any, Optional

import yaml

from flujo.domain.dsl import Pipeline
from flujo.domain.pipeline_validation import ValidationReport
from flujo.infra.skill_registry import get_skill_registry
from flujo.infra.skills_catalog import load_skills_catalog, load_skills_entry_points
from flujo.type_definitions.common import JSONObject

from .helpers_io import load_pipeline_from_file


def load_mermaid_code(file: str, object_name: str, detail_level: str) -> str:
    """Load a pipeline and return its Mermaid diagram code string."""
    pipeline, _ = load_pipeline_from_file(file, object_name)
    from flujo.visualization.visualize import visualize_with_detail_level

    return visualize_with_detail_level(pipeline, detail_level)


def get_pipeline_step_names(path: str) -> list[str]:
    """Return the ordered step names for a pipeline file."""
    if path.endswith((".yaml", ".yml")):
        with open(path, "r") as f:
            yaml_text = f.read()
        from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml

        base_dir = os.path.dirname(os.path.abspath(path))
        pipeline = load_pipeline_blueprint_from_yaml(yaml_text, base_dir=base_dir, source_file=path)
    else:
        pipeline, _ = load_pipeline_from_file(path)
    return [step.name for step in pipeline.steps]


def get_pipeline_explanation(path: str) -> list[str]:
    """Return explanations for each step in a pipeline file."""
    return get_pipeline_step_names(path)


def validate_pipeline_file(path: str, *, include_imports: bool = True) -> ValidationReport:
    """Return the validation report for a pipeline file."""
    # Reset rule override cache so each validation respects current env/profile settings.
    try:
        import flujo.validation.linters_base as _lb

        _lb._OVERRIDE_CACHE = None
    except Exception:
        pass
    if path.endswith((".yaml", ".yml")):
        with open(path, "r") as f:
            yaml_text = f.read()
        from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml
        from flujo.domain.pipeline_validation import ValidationFinding, ValidationReport

        base_dir = os.path.dirname(os.path.abspath(path))
        try:
            prev_strict = os.environ.get("FLUJO_STRICT_DSL")
            os.environ["FLUJO_STRICT_DSL"] = "0"
            try:
                pipeline: Pipeline[Any, Any] = load_pipeline_blueprint_from_yaml(
                    yaml_text, base_dir=base_dir, source_file=path
                )
            finally:
                if prev_strict is None:
                    os.environ.pop("FLUJO_STRICT_DSL", None)
                else:
                    os.environ["FLUJO_STRICT_DSL"] = prev_strict
        except Exception as e:
            msg = str(e)
            m = re.search(r"Failed to compile import '([^']+)' from '([^']+)': (.+)", msg)
            if m:
                alias, rel_path, reason = m.group(1), m.group(2), m.group(3)
                return ValidationReport(
                    errors=[
                        ValidationFinding(
                            rule_id="V-I1",
                            severity="error",
                            message=(
                                f"Imported blueprint '{alias}' could not be loaded from '{rel_path}': {reason}"
                            ),
                            step_name=f"imports.{alias}",
                            location_path=f"imports.{alias}",
                            file=path,
                        )
                    ],
                    warnings=[],
                )
            raise
    else:
        pipeline, _ = load_pipeline_from_file(path, lenient_dsl=True)

    return pipeline.validate_graph(include_imports=include_imports)


def validate_yaml_text(
    yaml_text: str, base_dir: Optional[str] = None, *, include_imports: bool = False
) -> ValidationReport:
    """Validate a YAML blueprint string and return its ValidationReport."""
    from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml

    prev_strict = os.environ.get("FLUJO_STRICT_DSL")
    os.environ["FLUJO_STRICT_DSL"] = "0"
    try:
        pipeline = load_pipeline_blueprint_from_yaml(yaml_text, base_dir=base_dir)
    finally:
        if prev_strict is None:
            os.environ.pop("FLUJO_STRICT_DSL", None)
        else:
            os.environ["FLUJO_STRICT_DSL"] = prev_strict
    return pipeline.validate_graph(include_imports=include_imports)


def sanitize_blueprint_yaml(yaml_text: str) -> str:
    """Repair common issues in Architect-generated YAML to improve validity."""
    if not yaml_text or not yaml_text.strip():
        return yaml_text

    try:
        data = yaml.safe_load(yaml_text)
    except Exception:
        return yaml_text

    if not isinstance(data, dict):
        return yaml_text

    steps = data.get("steps")
    if not isinstance(steps, list):
        return yaml_text

    name_counter: dict[str, int] = {}
    changed: bool = False

    def _gen_name(prefix: str) -> str:
        idx = name_counter.get(prefix, 0) + 1
        name_counter[prefix] = idx
        return f"{prefix}_{idx}"

    def _derive_prefix(node: JSONObject) -> str:
        agent = node.get("agent")
        if isinstance(agent, dict):
            sid = agent.get("id")
            if isinstance(sid, str) and sid.strip():
                tail = sid.rsplit(".", 1)[-1]
                return tail.replace(" ", "_")
        uses = node.get("uses")
        if isinstance(uses, str) and uses.strip():
            tail = uses.rsplit(".", 1)[-1]
            return tail.replace(" ", "_")
        kind = node.get("kind")
        if isinstance(kind, str) and kind.strip():
            return kind.replace(" ", "_")
        return "step"

    def _repair_node(node: Any) -> None:
        nonlocal changed
        if isinstance(node, dict):
            if isinstance(node.get("id"), str) and "agent" not in node and "uses" not in node:
                try:
                    agent_id = str(node.pop("id"))
                    params_val = node.pop("params", {})
                    if not isinstance(params_val, dict):
                        params_val = {}
                    node["agent"] = {"id": agent_id, "params": params_val}
                    changed = True
                except Exception:
                    pass
            if isinstance(node.get("step"), dict):
                embedded_raw = node.get("step")
                embedded = embedded_raw if isinstance(embedded_raw, dict) else None
                if isinstance(embedded, dict):
                    try:
                        del node["step"]
                    except Exception:
                        pass
                    for k, v in embedded.items():
                        if k not in node:
                            node[k] = v

            if "conditional" in node and "kind" not in node:
                cond = node.get("conditional")
                if isinstance(cond, str) and cond.strip():
                    node["kind"] = "conditional"
                    node["condition"] = cond
                    try:
                        del node["conditional"]
                    except Exception:
                        pass
                    changed = True

            if (
                "branches" in node
                or "agent" in node
                or "uses" in node
                or node.get("kind")
                in {
                    "step",
                    "parallel",
                    "conditional",
                    "loop",
                    "map",
                    "dynamic_router",
                }
            ):
                if not isinstance(node.get("name"), str) or not node.get("name", "").strip():
                    legacy = node.get("step")
                    if isinstance(legacy, str) and legacy.strip():
                        node["name"] = legacy.strip()
                    else:
                        node["name"] = _gen_name(_derive_prefix(node))
                    changed = True
                if not isinstance(node.get("kind"), str) or not node.get("kind", "").strip():
                    node["kind"] = "step"
                    changed = True
                agent_spec = node.get("agent")
                if isinstance(agent_spec, dict):
                    params = agent_spec.get("params")
                    if params is None:
                        agent_spec["params"] = {}
                        changed = True
                    sid = agent_spec.get("id")
                    if (
                        isinstance(sid, str)
                        and sid.endswith("extract_from_text")
                        and isinstance(agent_spec.get("params"), dict)
                    ):
                        p = agent_spec["params"]
                        if "text" not in p:
                            inp = node.get("input")
                            if isinstance(inp, str) and inp.strip():
                                p["text"] = inp
                                try:
                                    del node["input"]
                                except Exception:
                                    pass
                                changed = True
                    if (
                        isinstance(sid, str)
                        and sid.endswith("fs_write_file")
                        and isinstance(agent_spec.get("params"), dict)
                    ):
                        p = agent_spec["params"]
                        if "content" not in p:
                            inp = node.get("input")
                            if isinstance(inp, str) and inp.strip():
                                p["content"] = inp
                                try:
                                    del node["input"]
                                except Exception:
                                    pass
                                changed = True

            branches = node.get("branches")
            if isinstance(branches, list):
                node["branches"] = {"default": branches}
                branches = node["branches"]
                changed = True
            if isinstance(branches, dict):
                for _, lst in branches.items():
                    if isinstance(lst, list):
                        for child in lst:
                            _repair_node(child)

            loop_spec = node.get("loop")
            if isinstance(loop_spec, dict):
                body = loop_spec.get("body")
                if isinstance(body, list):
                    for child in body:
                        _repair_node(child)

            map_spec = node.get("map")
            if isinstance(map_spec, dict):
                if "input" in map_spec and "iterable_input" not in map_spec:
                    try:
                        map_spec["iterable_input"] = map_spec.pop("input")
                        changed = True
                    except Exception:
                        pass
                if "steps" in map_spec and "body" not in map_spec:
                    try:
                        steps_val = map_spec.pop("steps")
                        if isinstance(steps_val, list):
                            map_spec["body"] = steps_val
                        elif isinstance(steps_val, dict):
                            map_spec["body"] = [steps_val]
                        else:
                            map_spec["body"] = []
                        changed = True
                    except Exception:
                        pass
                body = map_spec.get("body")
                if isinstance(body, list):
                    for child in body:
                        _repair_node(child)

            if "parallel" in node and (
                not isinstance(node.get("kind"), str) or node.get("kind") == "step"
            ):
                par = node.get("parallel")
                try:
                    del node["parallel"]
                except Exception:
                    pass
                node["kind"] = "parallel"
                branches_map: JSONObject = {}
                if isinstance(par, list):
                    for idx, child in enumerate(par, start=1):
                        branch_name = f"branch_{idx}"
                        if isinstance(child, dict):
                            branches_map[branch_name] = [child]
                        else:
                            branches_map[branch_name] = []
                elif isinstance(par, dict):
                    branches_map["branch_1"] = [par]
                else:
                    branches_map["branch_1"] = []
                node["branches"] = branches_map
                changed = True
                for lst in branches_map.values():
                    for child in lst:
                        _repair_node(child)

        elif isinstance(node, list):
            for it in node:
                _repair_node(it)

    _repair_node(steps)

    if not changed:
        return yaml_text
    try:
        return str(yaml.safe_dump(data, sort_keys=False))
    except Exception:
        return yaml_text


def find_side_effect_skills_in_yaml(yaml_text: str, *, base_dir: Optional[str] = None) -> list[str]:
    """Return a list of skill IDs in YAML that are marked side_effects=True in registry."""
    if not yaml_text or not yaml_text.strip():
        return []

    if not any(
        indicator in yaml_text for indicator in ["version:", "steps:", "pipeline:", "workflow:"]
    ):
        return []

    try:
        data = yaml.safe_load(yaml_text)
    except Exception:
        return []

    if base_dir:
        load_skills_catalog(base_dir)
        load_skills_entry_points()

    skill_registry = get_skill_registry()
    side_effect_skills: list[str] = []
    visited: set[int] = set()

    def _walk(node: Any) -> None:
        if id(node) in visited:
            return
        visited.add(id(node))

        if isinstance(node, dict):
            if "agent" in node and isinstance(node["agent"], dict):
                agent_info = node["agent"]
                skill_id = agent_info.get("id")
                if isinstance(skill_id, str):
                    skill = skill_registry.get(skill_id)
                    is_side_effect = False
                    if isinstance(skill, dict):
                        is_side_effect = bool(skill.get("side_effects"))
                    elif skill is not None:
                        is_side_effect = bool(getattr(skill, "side_effects", False))
                    if is_side_effect:
                        side_effect_skills.append(skill_id)
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(data)
    return side_effect_skills


def enrich_yaml_with_required_params(
    yaml_text: str, *, base_dir: Optional[str] = None, non_interactive: Optional[bool] = None
) -> str:
    """Enrich YAML with required params for each agent based on the skill registry."""
    try:
        data = yaml.safe_load(yaml_text)
    except Exception:
        return yaml_text

    if not isinstance(data, dict):
        return yaml_text

    if base_dir:
        load_skills_catalog(base_dir)
        load_skills_entry_points()

    skill_registry = get_skill_registry()
    inserted_params: dict[str, JSONObject] = {}

    def _walk(node: Any, path: str = "") -> None:
        if isinstance(node, dict):
            if "agent" in node and isinstance(node["agent"], dict):
                agent_info = node["agent"]
                skill_id = agent_info.get("id")
                if isinstance(skill_id, str):
                    skill = skill_registry.get(skill_id)
                    if isinstance(skill, dict):
                        required_params = skill.get("required_params")
                    else:
                        required_params = getattr(skill, "required_params", None) if skill else None
                    if required_params:
                        params = agent_info.get("params")
                        if not isinstance(params, dict):
                            params = {}
                        added: JSONObject = {}
                        for key, default in required_params.items():
                            if key not in params:
                                params[key] = default
                                added[key] = default
                        if added:
                            agent_info["params"] = params
                            inserted_params[path or "root"] = added
            for key, value in node.items():
                new_path = f"{path}.{key}" if path else str(key)
                _walk(value, new_path)
        elif isinstance(node, list):
            for idx, item in enumerate(node):
                new_path = f"{path}[{idx}]"
                _walk(item, new_path)

    _walk(data)

    return str(yaml.safe_dump(data, sort_keys=False))
