"""Blueprint loader for parsing YAML pipeline definitions.

This module handles loading and parsing YAML blueprints into executable Pipeline objects.
It supports declarative agents, imports, and various step types.

Import Strategy (per FLUJO_TEAM_GUIDE Section 12):
- Top-level imports: Standard library and core dependencies (os, sys, yaml, etc.)
- TYPE_CHECKING imports: Type-only imports to avoid circular dependencies
- Runtime imports: Used only for:
  1. Optional dependencies (ruamel.yaml, flujo.builtins) - loaded conditionally
  2. Circular dependency resolution (compiler module) - loaded when needed
  3. Optional step types (DynamicParallelRouterStep, MapStep, etc.) - loaded on demand

This approach balances static analysis support with runtime flexibility for optional features.
"""

from __future__ import annotations

import os
import sys
from typing import TypeVar

import yaml
from pydantic import ValidationError

from ..dsl import ParallelStep, Pipeline
from ..interfaces import get_skills_discovery
from .loader_models import BlueprintError, BlueprintPipelineModel
from .loader_resolution import _pop_skills_base_dir, _push_skills_base_dir
from .loader_steps import build_pipeline_from_blueprint
from flujo.type_definitions.common import JSONObject
from flujo.exceptions import ControlFlowError


PipeInT = TypeVar("PipeInT")
PipeOutT = TypeVar("PipeOutT")


def dump_pipeline_blueprint_to_yaml(pipeline: Pipeline[PipeInT, PipeOutT]) -> str:
    """Serialize a Pipeline to a minimal YAML blueprint (v0)."""

    def step_to_yaml(step: object) -> JSONObject:
        if isinstance(step, ParallelStep):
            branches: JSONObject = {}
            for k, p in step.branches.items():
                branches[str(k)] = [step_to_yaml(s) for s in p.steps]
            return {
                "kind": "parallel",
                "name": step.name,
                "branches": branches,
                "merge_strategy": getattr(step.merge_strategy, "name", None),
            }
        try:
            from ..dsl.dynamic_router import DynamicParallelRouterStep

            if isinstance(step, DynamicParallelRouterStep):
                branches = {
                    str(k): [step_to_yaml(s) for s in p.steps] for k, p in step.branches.items()
                }
                router_data: JSONObject = {"branches": branches}
                agent = getattr(step, "router_agent", None)
                try:
                    if (
                        agent is not None
                        and hasattr(agent, "__module__")
                        and hasattr(agent, "__name__")
                    ):
                        router_data["router_agent"] = f"{agent.__module__}:{agent.__name__}"
                except Exception:
                    pass
                router_step_data: JSONObject = {
                    "kind": "dynamic_router",
                    "name": step.name,
                    "router": router_data,
                }
                merge_strategy = getattr(step, "merge_strategy", None)
                merge_strategy_name = getattr(merge_strategy, "name", None)
                if merge_strategy_name is not None:
                    router_step_data["merge_strategy"] = merge_strategy_name
                return router_step_data
        except Exception:
            pass
        try:
            from ..dsl.loop import MapStep

            if isinstance(step, MapStep):
                body = getattr(step, "original_body_pipeline", None) or getattr(
                    step, "pipeline_to_run", None
                )
                body_steps: list[JSONObject] = []
                if body is not None:
                    body_steps = [step_to_yaml(s) for s in body.steps]
                return {
                    "kind": "map",
                    "name": step.name,
                    "map": {
                        "iterable_input": getattr(step, "iterable_input", None),
                        "body": body_steps,
                    },
                }
        except Exception:
            pass
        try:
            from ..dsl.conditional import ConditionalStep

            if isinstance(step, ConditionalStep):
                branches = {
                    str(k): [step_to_yaml(s) for s in p.steps] for k, p in step.branches.items()
                }
                data: JSONObject = {
                    "kind": "conditional",
                    "name": step.name,
                    "branches": branches,
                }
                if isinstance(step.default_branch_pipeline, Pipeline):
                    data["default_branch"] = [
                        step_to_yaml(s) for s in step.default_branch_pipeline.steps
                    ]
                return data
        except Exception:
            pass
        try:
            from ..dsl.loop import LoopStep

            if isinstance(step, LoopStep):
                body_pipe = getattr(step, "loop_body_pipeline", None)
                loop_body_steps = getattr(body_pipe, "steps", None)
                if not isinstance(loop_body_steps, (list, tuple)):
                    loop_body_steps = []
                loop_data = {
                    "body": [step_to_yaml(s) for s in loop_body_steps],
                    "max_loops": step.max_retries,
                }
                return {
                    "kind": "loop",
                    "name": step.name,
                    "loop": loop_data,
                }
        except Exception:
            pass
        try:
            from ..dsl.step import HumanInTheLoopStep

            if isinstance(step, HumanInTheLoopStep):
                hitl_data: JSONObject = {
                    "kind": "hitl",
                    "name": step.name,
                }
                try:
                    if getattr(step, "message_for_user", None):
                        hitl_data["message"] = getattr(step, "message_for_user")
                except Exception:
                    pass
                try:
                    schema = getattr(step, "input_schema", None)
                    if schema is not None:
                        if hasattr(schema, "model_json_schema") and callable(
                            getattr(schema, "model_json_schema")
                        ):
                            hitl_data["input_schema"] = schema.model_json_schema()
                        elif isinstance(schema, dict):
                            hitl_data["input_schema"] = schema
                except Exception:
                    pass
                try:
                    if getattr(step, "sink_to", None):
                        hitl_data["sink_to"] = getattr(step, "sink_to")
                except Exception:
                    pass
                return hitl_data
        except Exception:
            pass
        try:
            from flujo.domain.dsl.cache_step import CacheStep as _CacheStep

            if isinstance(step, _CacheStep):
                wrapped = getattr(step, "wrapped_step", None)
                return {
                    "kind": "cache",
                    "name": getattr(step, "name", "cache"),
                    "wrapped_step": step_to_yaml(wrapped)
                    if wrapped is not None
                    else {"kind": "step", "name": "step"},
                }
        except Exception:
            pass
        return {"kind": "step", "name": getattr(step, "name", "step")}

    data: JSONObject = {
        "version": "0.1",
        "steps": [step_to_yaml(s) for s in pipeline.steps],
    }
    return str(yaml.safe_dump(data, sort_keys=False))


def load_pipeline_blueprint_from_yaml(
    yaml_text: str,
    base_dir: str | None = None,
    source_file: str | None = None,
    _visited: list[str] | None = None,
) -> Pipeline[object, object]:
    """
    Load a pipeline from a YAML blueprint with support for imports and agents.

    This function handles sys.path manipulation for skills discovery, which requires
    runtime imports of os and sys. The imports are at module level to improve
    static analysis and follow FLUJO_TEAM_GUIDE Section 12 (Type Safety).

    Args:
        yaml_text: YAML string containing the pipeline blueprint
        base_dir: Optional base directory for resolving relative imports
        source_file: Optional source file path for cycle detection
        _visited: Internal parameter for tracking visited files during import resolution

    Returns:
        Compiled Pipeline object ready for execution

    Raises:
        BlueprintError: If the blueprint is invalid, has cyclic imports, or compilation fails
    """
    _pushed_sys_path: bool = False
    base_dir_abs: str | None = None
    _push_skills_base_dir(base_dir)
    if base_dir:
        try:
            base_dir_abs = os.path.abspath(base_dir)
            if base_dir_abs not in sys.path:
                sys.path.insert(0, base_dir_abs)
                _pushed_sys_path = True
        except Exception:
            _pushed_sys_path = False

    # Cycle detection - stack-based approach
    # Track the current recursion path (stack) rather than all files ever seen
    # This correctly distinguishes cycles (A -> B -> A) from re-imports (A -> B, then C -> B)
    _file_pushed = False
    real_source: str | None = None
    if source_file:
        if _visited is None:
            _visited = []
        real_source = os.path.realpath(source_file)
        # Check if this file is already in the current call stack (cycle)
        if real_source in _visited:
            # Show the cycle path for better error messages
            cycle_path = " -> ".join(_visited[-3:] + [real_source])
            raise BlueprintError(f"Cyclic import detected: {real_source} (path: {cycle_path})")
        # Push onto stack before processing
        _visited.append(real_source)
        _file_pushed = True
    try:
        loc_index: dict[str, tuple[int, int]] = {}
        sup_index: dict[str, list[str]] = {}
        try:
            from ruamel.yaml import YAML as _RYAML
            from ruamel.yaml.comments import CommentedMap as _CMap, CommentedSeq as _CSeq

            def _build_index(
                txt: str,
            ) -> tuple[dict[str, tuple[int, int]], dict[str, list[str]]]:
                yaml_rt = _RYAML(typ="rt")
                root = yaml_rt.load(txt)
                idx: dict[str, tuple[int, int]] = {}
                sup: dict[str, list[str]] = {}

                def _extract_ignores(comment_obj: object) -> list[str]:
                    pats: list[str] = []
                    try:
                        import re as _re

                        texts: list[str] = []
                        if comment_obj is None:
                            return pats
                        if isinstance(comment_obj, (list, tuple)):
                            for c in comment_obj:
                                try:
                                    val = getattr(c, "value", None)
                                    texts.append(str(val if val is not None else c))
                                except Exception:
                                    continue
                        else:
                            val = getattr(comment_obj, "value", None)
                            texts.append(str(val if val is not None else comment_obj))
                        for t in texts:
                            m = _re.search(r"flujo:\s*ignore\s+([^\n#]+)", t, _re.IGNORECASE)
                            if m:
                                body = m.group(1)
                                for part in body.replace("\t", " ").split(","):
                                    tok = part.strip()
                                    if tok:
                                        for sub in tok.split():
                                            if sub and sub not in pats:
                                                pats.append(sub)
                    except Exception:
                        return pats
                    return pats

                def _recurse(node: object, path: str) -> None:
                    try:
                        if isinstance(node, _CMap):
                            for k, v in node.items():
                                key_path = f"{path}.{k}" if path else str(k)
                                try:
                                    if hasattr(node, "lc") and hasattr(node.lc, "key"):
                                        pos = node.lc.key(k)
                                        if isinstance(pos, tuple) and len(pos) >= 2:
                                            idx[key_path] = (int(pos[0]) + 1, int(pos[1]) + 1)
                                    if hasattr(node, "ca") and hasattr(node.ca, "items"):
                                        ent = node.ca.items.get(k)
                                        if ent:
                                            pats = _extract_ignores(ent)
                                            if pats:
                                                sup.setdefault(key_path, []).extend(pats)
                                                try:
                                                    import re as _re

                                                    m = _re.search(r"^(.*)\[(\d+)\]$", path)
                                                    if m:
                                                        base = m.group(1)
                                                        idx_s = m.group(2)
                                                        if base.endswith("steps"):
                                                            step_key = f"{base}[{idx_s}]"
                                                        else:
                                                            step_key = f"{base}.steps[{idx_s}]"
                                                        sup.setdefault(step_key, []).extend(pats)
                                                except Exception:
                                                    pass
                                except Exception:
                                    pass
                                if k == "steps" and isinstance(v, _CSeq):
                                    try:
                                        for i in range(len(v)):
                                            lc = v.lc.data.get(i)
                                            if lc and len(lc) >= 2:
                                                idx[f"{key_path}[{i}]"] = (
                                                    int(lc[0]) + 1,
                                                    int(lc[1]) + 1,
                                                )
                                    except Exception:
                                        pass
                                _recurse(v, key_path)
                        elif isinstance(node, _CSeq):
                            for i, item in enumerate(node):
                                try:
                                    lc = getattr(node.lc, "data", {}).get(i)
                                    if lc and len(lc) >= 2:
                                        line_col = (int(lc[0]) + 1, int(lc[1]) + 1)
                                        idx[f"{path}[{i}]"] = line_col
                                        idx[f"{path}.steps[{i}]"] = line_col
                                    pats_item = []
                                    try:
                                        if hasattr(node, "ca") and hasattr(node.ca, "items"):
                                            ent = node.ca.items.get(i)
                                            if ent:
                                                pats_item.extend(_extract_ignores(ent))
                                        if hasattr(item, "ca") and hasattr(item.ca, "comment"):
                                            pats_item.extend(_extract_ignores(item.ca.comment))
                                    except Exception:
                                        pass
                                    if pats_item:
                                        sup.setdefault(f"{path}.steps[{i}]", []).extend(pats_item)
                                except Exception:
                                    pass
                                _recurse(item, f"{path}[{i}]")
                    except Exception:
                        pass

                _recurse(root, "")
                return idx, sup

            loc_index, sup_index = _build_index(yaml_text)
        except Exception:
            loc_index = {}
            sup_index = {}
        if base_dir:
            discovery = get_skills_discovery()
            try:
                discovery.load_catalog(base_dir)
            except Exception:
                pass
            try:
                discovery.load_entry_points()
            except Exception:
                pass
    except Exception:
        pass
    try:
        import flujo.builtins  # noqa: F401
    except Exception:
        pass
    try:
        data = yaml.safe_load(yaml_text)
        if not isinstance(data, dict) or "steps" not in data:
            raise BlueprintError("YAML blueprint must be a mapping with a 'steps' key")
        bp = BlueprintPipelineModel.model_validate(data)
        # Runtime import required to avoid circular dependency with compiler module
        # The TYPE_CHECKING import above is for type hints only
        from .compiler import DeclarativeBlueprintCompiler

        if bp.agents or getattr(bp, "imports", None):
            try:
                compiler = DeclarativeBlueprintCompiler(bp, base_dir=base_dir, _visited=_visited)
                return compiler.compile_to_pipeline()
            except Exception as e:
                raise BlueprintError(
                    f"Failed to compile declarative blueprint (agents/imports): {e}"
                ) from e
        try:
            p = build_pipeline_from_blueprint(bp)
        except BlueprintError:
            raise
        except Exception as e:
            if isinstance(e, ControlFlowError):
                raise
            raise BlueprintError(f"Failed to build pipeline from blueprint: {e}") from e
        try:
            name_val = getattr(bp, "name", None)
            if isinstance(name_val, str):
                name_val_stripped = name_val.strip()
                if name_val_stripped:
                    try:
                        setattr(p, "name", name_val_stripped)
                    except Exception:
                        try:
                            object.__setattr__(p, "name", name_val_stripped)
                        except Exception:
                            pass
        except Exception:
            pass
        try:
            if source_file:
                try:
                    setattr(p, "_source_file", str(source_file))
                except Exception:
                    object.__setattr__(p, "_source_file", str(source_file))
        except Exception:
            pass
        try:
            from ..dsl import Pipeline as _DPipe, Step as _DStep

            def _attach_step_loc(st: object) -> None:
                try:
                    meta = getattr(st, "meta", None)
                    if isinstance(meta, dict) and "yaml_path" in meta:
                        ypath = str(meta.get("yaml_path"))
                        if ypath in loc_index:
                            ln, col = loc_index[ypath]
                            info = {"path": ypath, "line": int(ln), "column": int(col)}
                            if source_file:
                                info["file"] = str(source_file)
                            meta["_yaml_loc"] = info
                        pats = sup_index.get(ypath) or []
                        if pats:
                            try:
                                lst = meta.setdefault("suppress_rules", [])
                                for ptn in pats:
                                    if ptn not in lst:
                                        lst.append(ptn)
                            except Exception:
                                pass
                except Exception:
                    pass
                try:
                    fb = getattr(st, "fallback_step", None)
                    if isinstance(fb, _DStep):
                        _attach_step_loc(fb)
                except Exception:
                    pass
                try:
                    branches = getattr(st, "branches", None)
                    if isinstance(branches, dict):
                        for _bn, _bp in branches.items():
                            if isinstance(_bp, _DPipe):
                                _attach_pipe_loc(_bp)
                except Exception:
                    pass
                try:
                    states = getattr(st, "states", None)
                    if isinstance(states, dict):
                        for _sn, _sp in states.items():
                            if isinstance(_sp, _DPipe):
                                _attach_pipe_loc(_sp)
                except Exception:
                    pass
                for attr in (
                    "default_branch_pipeline",
                    "loop_body_pipeline",
                    "original_body_pipeline",
                    "pipeline_to_run",
                ):
                    try:
                        bpv = getattr(st, attr, None)
                        if isinstance(bpv, _DPipe):
                            _attach_pipe_loc(bpv)
                    except Exception:
                        continue
                try:
                    ws = getattr(st, "wrapped_step", None)
                    if isinstance(ws, _DStep):
                        _attach_step_loc(ws)
                except Exception:
                    pass

            def _attach_pipe_loc(pipe: object) -> None:
                try:
                    for _st in getattr(pipe, "steps", []) or []:
                        _attach_step_loc(_st)
                except Exception:
                    pass

            _attach_pipe_loc(p)
        except Exception:
            pass
        return p
    except ValidationError as ve:
        try:
            errs = ve.errors()
            messages = [
                f"{e.get('msg')} at {'.'.join(str(p) for p in e.get('loc', []))}" for e in errs
            ]
            raise BlueprintError("; ".join(messages)) from ve
        except Exception:
            raise BlueprintError(str(ve)) from ve
    except yaml.YAMLError as ye:
        mark = getattr(ye, "problem_mark", None)
        if mark is not None:
            msg = f"Invalid YAML at line {getattr(mark, 'line', -1) + 1}, column {getattr(mark, 'column', -1) + 1}: {getattr(ye, 'problem', ye)}"
        else:
            msg = f"Invalid YAML: {ye}"
        raise BlueprintError(msg) from ye
    finally:
        # Pop from stack after processing (critical for correct cycle detection)
        if _file_pushed and _visited and real_source is not None:
            if _visited and _visited[-1] == real_source:
                _visited.pop()
        try:
            if _pushed_sys_path and base_dir_abs:
                if base_dir_abs in sys.path:
                    sys.path.remove(base_dir_abs)
        except Exception:
            pass
        _pop_skills_base_dir()


__all__ = [
    "build_pipeline_from_blueprint",
    "dump_pipeline_blueprint_to_yaml",
    "load_pipeline_blueprint_from_yaml",
]
