from __future__ import annotations

from typing import Callable, Optional, TypeAlias

from flujo.type_definitions.common import JSONObject

from ..dsl import Pipeline, Step, StepConfig
from .loader_models import BlueprintError, BlueprintStepModel
from .loader_resolution import _import_object
from .loader_templates import _render_template_value, _resolve_context_target

AnyPipeline: TypeAlias = Pipeline[object, object]
AnyStep: TypeAlias = Step[object, object]
BuildBranch = Callable[..., AnyPipeline]


def build_loop_step(
    model: BlueprintStepModel,
    step_config: StepConfig,
    *,
    yaml_path: Optional[str],
    compiled_agents: Optional[JSONObject],
    compiled_imports: Optional[JSONObject],
    build_branch: BuildBranch,
) -> AnyStep:
    from typing import Callable as _Callable, Optional as _Optional
    from ..dsl.loop import LoopStep
    from ..models import BaseModel as _BaseModel

    if not model.loop or "body" not in model.loop:
        raise BlueprintError("loop step requires loop.body")
    body = build_branch(
        model.loop.get("body"),
        base_path=f"{yaml_path}.loop.body" if yaml_path else None,
        compiled_agents=compiled_agents,
        compiled_imports=compiled_imports,
    )
    max_loops = model.loop.get("max_loops")

    _initial_mapper = None
    if model.loop.get("initial_input_mapper"):
        _initial_mapper = _import_object(model.loop["initial_input_mapper"])

    _iter_mapper = None
    if model.loop.get("iteration_input_mapper"):
        _iter_mapper = _import_object(model.loop["iteration_input_mapper"])

    _output_mapper = None
    if model.loop.get("loop_output_mapper"):
        _output_mapper = _import_object(model.loop["loop_output_mapper"])

    if model.loop.get("exit_condition"):
        _exit_condition = _import_object(model.loop["exit_condition"])
        import asyncio

        if asyncio.iscoroutinefunction(_exit_condition):
            raise BlueprintError(
                f"exit_condition '{model.loop['exit_condition']}' must be synchronous.\n"
                f"Loop exit conditions are called synchronously and cannot be async functions.\n"
                f"\n"
                f"Change your function from:\n"
                f"  async def my_condition(output, context) -> bool:\n"
                f"      ...\n"
                f"\n"
                f"To:\n"
                f"  def my_condition(output, context) -> bool:\n"
                f"      ...\n"
                f"\n"
                f"Remove 'async' and any 'await' calls in your exit_condition function.\n"
                f"See: https://flujo.dev/docs/loops#exit-conditions"
            )
    elif model.loop.get("exit_expression"):
        try:
            from ...utils.expressions import compile_expression_to_callable as _compile_expr

            _expr_fn2 = _compile_expr(str(model.loop["exit_expression"]))

            def _exit_condition(
                _output: object, _ctx: object | None, *, _state: dict[str, int] | None = None
            ) -> bool:
                return bool(_expr_fn2(_output, _ctx))

        except Exception as e:
            raise BlueprintError(f"Invalid exit_expression: {e}") from e
    else:

        def _exit_condition(
            _output: object, _ctx: object | None, *, _state: dict[str, int] | None = None
        ) -> bool:
            if _state is None:
                _state = {"count": 0}
            _state["count"] += 1
            if isinstance(max_loops, int) and max_loops > 0:
                return _state["count"] >= max_loops
            return _state["count"] >= 1

    _initial_mapper_override: _Optional[_Callable[[object, object | None], object]] = None
    _iteration_mapper_override: _Optional[_Callable[[object, object | None, int], object]] = None
    _output_mapper_override: _Optional[_Callable[[object, object | None], object]] = None

    _state_apply_fn: _Optional[_Callable[[object, object], None]] = None
    _compiled_init_ops: _Optional[_Callable[[object, object], None]] = None
    _compiled_iter_prop: _Optional[_Callable[[object, object | None, int], object]] = None

    try:
        state_spec = model.loop.get("state") if isinstance(model.loop, dict) else None
    except Exception:
        state_spec = None

    if isinstance(state_spec, dict) and any(k in state_spec for k in ("append", "set", "merge")):
        try:
            import json as _json

            ops_append = state_spec.get("append") or []
            ops_set = state_spec.get("set") or []
            ops_merge = state_spec.get("merge") or []

            def _render_value(output: object, ctx: object, tpl: str) -> str:
                return str(_render_template_value(output, ctx, tpl))

            def _apply_state_ops(output: object, ctx: object) -> None:
                for spec in ops_append:
                    try:
                        target = str(spec.get("target"))
                        parent, key = _resolve_context_target(ctx, target)
                        if parent is None or key is None:
                            continue
                        val = _render_value(output, ctx, spec.get("value", ""))
                        if isinstance(parent, dict):
                            seq = parent.get(key)
                        else:
                            seq = getattr(parent, key, None)
                        if not isinstance(seq, list):
                            seq = []
                            if isinstance(parent, dict):
                                parent[key] = seq
                            else:
                                try:
                                    setattr(parent, key, seq)
                                except Exception:
                                    continue
                        seq.append(val)
                    except Exception:
                        continue
                for spec in ops_set:
                    try:
                        target = str(spec.get("target"))
                        parent, key = _resolve_context_target(ctx, target)
                        if parent is None or key is None:
                            continue
                        val = _render_value(output, ctx, spec.get("value", ""))
                        if isinstance(parent, dict):
                            parent[key] = val
                        else:
                            try:
                                setattr(parent, key, val)
                            except Exception:
                                continue
                    except Exception:
                        continue
                for spec in ops_merge:
                    try:
                        target = str(spec.get("target"))
                        parent, key = _resolve_context_target(ctx, target)
                        if parent is None or key is None:
                            continue
                        val_raw = _render_value(output, ctx, spec.get("value", "{}"))
                        try:
                            val_obj = _json.loads(val_raw)
                        except Exception:
                            continue
                        if not isinstance(val_obj, dict):
                            continue
                        try:
                            cur = (
                                parent.get(key)
                                if isinstance(parent, dict)
                                else getattr(parent, key, None)
                            )
                        except Exception:
                            cur = None
                        if not isinstance(cur, dict):
                            cur = {}
                        cur.update(val_obj)
                        if isinstance(parent, dict):
                            parent[key] = cur
                        else:
                            try:
                                setattr(parent, key, cur)
                            except Exception:
                                continue
                    except Exception:
                        continue

            _state_apply_fn = _apply_state_ops

            def _initial_mapper_override(input_data: object, ctx: object | None) -> object:
                return input_data

            def _iteration_mapper_override(
                output: object, ctx: object | None, _iteration: int
            ) -> object:
                if ctx is not None:
                    _apply_state_ops(output, ctx)
                return output

            def _output_mapper_override(output: object, ctx: object | None) -> object:
                return output

        except Exception:
            pass

    try:
        init_spec = model.loop.get("init") if isinstance(model.loop, dict) else None
    except Exception:
        init_spec = None
    if init_spec is not None:
        try:
            import json as _json2

            ops_init_append: list[dict[str, object]] = []
            ops_init_set: list[dict[str, object]] = []
            ops_init_merge: list[dict[str, object]] = []

            if isinstance(init_spec, dict):
                ops_init_append = list(init_spec.get("append") or [])
                ops_init_set = list(init_spec.get("set") or [])
                ops_init_merge = list(init_spec.get("merge") or [])
            elif isinstance(init_spec, list):
                for op in init_spec:
                    try:
                        if not isinstance(op, dict):
                            continue
                        if "set" in op:
                            ops_init_set.append(
                                {
                                    "target": op.get("set"),
                                    "value": op.get("value"),
                                }
                            )
                        elif "append" in op:
                            _tmp_a = op.get("append")
                            if isinstance(_tmp_a, str):
                                ops_init_append.append(
                                    {
                                        "target": _tmp_a,
                                        "value": op.get("value"),
                                    }
                                )
                            elif isinstance(_tmp_a, dict):
                                d = _tmp_a
                                ops_init_append.append(
                                    {
                                        "target": d.get("target"),
                                        "value": d.get("value"),
                                    }
                                )
                        elif "merge" in op:
                            _tmp_m = op.get("merge")
                            if isinstance(_tmp_m, str):
                                ops_init_merge.append(
                                    {
                                        "target": _tmp_m,
                                        "value": op.get("value") if op.get("value") else "{}",
                                    }
                                )
                            elif isinstance(_tmp_m, dict):
                                d = _tmp_m
                                ops_init_merge.append(
                                    {
                                        "target": d.get("target"),
                                        "value": d.get("value") if d.get("value") else "{}",
                                    }
                                )
                    except Exception:
                        continue

            def _init_ops(prev_output: object, ctx: object) -> None:
                for spec in ops_init_append:
                    try:
                        parent, key = _resolve_context_target(ctx, str(spec.get("target")))
                        if parent is None or key is None:
                            continue
                        val = _render_template_value(prev_output, ctx, spec.get("value", ""))
                        if isinstance(parent, dict):
                            seq = parent.get(key)
                        else:
                            seq = getattr(parent, key, None)
                        if not isinstance(seq, list):
                            seq = []
                            if isinstance(parent, dict):
                                parent[key] = seq
                            else:
                                try:
                                    setattr(parent, key, seq)
                                except Exception:
                                    continue
                        seq.append(val)
                    except Exception:
                        continue
                for spec in ops_init_set:
                    try:
                        parent, key = _resolve_context_target(ctx, str(spec.get("target")))
                        if parent is None or key is None:
                            continue
                        val = _render_template_value(prev_output, ctx, spec.get("value", ""))
                        if isinstance(parent, dict):
                            parent[key] = val
                        else:
                            try:
                                setattr(parent, key, val)
                            except Exception:
                                continue
                    except Exception:
                        continue
                for spec in ops_init_merge:
                    try:
                        parent, key = _resolve_context_target(ctx, str(spec.get("target")))
                        if parent is None or key is None:
                            continue
                        val_raw = _render_template_value(prev_output, ctx, spec.get("value", "{}"))
                        try:
                            val_text = (
                                val_raw
                                if isinstance(val_raw, (str, bytes, bytearray))
                                else str(val_raw)
                            )
                            val_obj = _json2.loads(val_text)
                        except Exception:
                            continue
                        if not isinstance(val_obj, dict):
                            continue
                        try:
                            cur = (
                                parent.get(key)
                                if isinstance(parent, dict)
                                else getattr(parent, key, None)
                            )
                        except Exception:
                            cur = None
                        if not isinstance(cur, dict):
                            cur = {}
                        cur.update(val_obj)
                        if isinstance(parent, dict):
                            parent[key] = cur
                        else:
                            try:
                                setattr(parent, key, cur)
                            except Exception:
                                continue
                    except Exception:
                        continue

            _compiled_init_ops = _init_ops
        except Exception:
            pass

    try:
        propagate_spec = None
        if isinstance(model.loop, dict):
            propagate_spec = model.loop.get("propagation") or model.loop.get("propagate")
    except Exception:
        propagate_spec = None
    if propagate_spec:
        try:
            if not isinstance(propagate_spec, dict):
                propagate_spec = {"field": propagate_spec}
            next_input_val = propagate_spec.get("next_input")
            if isinstance(next_input_val, str) and next_input_val.strip().lower() == "context":

                def _propagate_ctx(
                    _output: object, _ctx: object | None, _iteration: int
                ) -> object | None:
                    return _ctx

                _iteration_mapper_override = _propagate_ctx
            if any(k in propagate_spec for k in ("field", "path", "prefix")):
                _field = str(propagate_spec.get("field", "context"))
                _prefix = str(propagate_spec.get("prefix", "loop_"))
                _path = str(propagate_spec.get("path", "context"))

                def _iter_prop(output: object, ctx: object | None, _iteration: int) -> object:
                    if ctx is None:
                        return output
                    target, key = None, None
                    if _path == "context":
                        target, key = ctx, _field
                    else:
                        target, key = _resolve_context_target(ctx, f"{_path}.{_field}")
                    if target is None:
                        return output
                    try:
                        if isinstance(target, dict):
                            target[f"{_prefix}{_iteration}"] = output
                        else:
                            setattr(target, f"{_prefix}{_iteration}", output)
                    except Exception:
                        pass
                    return output

                _compiled_iter_prop = _iter_prop
        except Exception:
            pass

    max_retries_val = (
        max_loops
        if isinstance(max_loops, int) and max_loops > 0
        else LoopStep.model_fields["max_retries"].default
    )

    st_loop: LoopStep[_BaseModel] = LoopStep(
        name=model.name,
        loop_body_pipeline=body,
        max_retries=max_retries_val,
        exit_condition_callable=_exit_condition,
        iteration_input_mapper=(
            _compiled_iter_prop
            if _compiled_iter_prop is not None
            else _iteration_mapper_override
            if _iteration_mapper_override is not None
            else _iter_mapper
        ),
        initial_input_to_loop_body_mapper=(
            _initial_mapper_override if _initial_mapper_override is not None else _initial_mapper
        ),
        loop_output_mapper=(
            _output_mapper_override if _output_mapper_override is not None else _output_mapper
        ),
    )
    try:
        if _compiled_init_ops is not None:
            st_loop.meta["compiled_init_ops"] = _compiled_init_ops
    except Exception:
        pass
    try:
        if isinstance(model.loop, dict) and model.loop.get("exit_expression"):
            st_loop.meta["exit_expression"] = str(model.loop.get("exit_expression"))
    except Exception:
        pass
    try:
        if isinstance(model.loop, dict):
            if "conversation" in model.loop:
                st_loop.meta["conversation"] = bool(model.loop.get("conversation"))
            if "history_management" in model.loop and isinstance(
                model.loop.get("history_management"), dict
            ):
                st_loop.meta["history_management"] = dict(
                    model.loop.get("history_management") or {}
                )
            if "history_template" in model.loop and isinstance(
                model.loop.get("history_template"), str
            ):
                st_loop.meta["history_template"] = str(model.loop.get("history_template"))
            if "ai_turn_source" in model.loop:
                st_loop.meta["ai_turn_source"] = str(model.loop.get("ai_turn_source"))
            if "user_turn_sources" in model.loop:
                uts = model.loop.get("user_turn_sources")
                if isinstance(uts, list):
                    st_loop.meta["user_turn_sources"] = list(uts)
                else:
                    st_loop.meta["user_turn_sources"] = [uts]
            if "named_steps" in model.loop:
                ns = model.loop.get("named_steps")
                if isinstance(ns, list):
                    st_loop.meta["named_steps"] = [str(x) for x in ns]
    except Exception:
        pass
    return st_loop


def build_map_step(
    model: BlueprintStepModel,
    step_config: StepConfig,
    *,
    yaml_path: Optional[str],
    compiled_agents: Optional[JSONObject],
    compiled_imports: Optional[JSONObject],
    build_branch: BuildBranch,
) -> AnyStep:
    from ..dsl.loop import MapStep
    from ..models import BaseModel as _BaseModel

    if not model.map or "iterable_input" not in model.map or "body" not in model.map:
        raise BlueprintError("map step requires map.iterable_input and map.body")
    body = build_branch(
        model.map.get("body"),
        base_path=f"{yaml_path}.map.body" if yaml_path else None,
        compiled_agents=compiled_agents,
        compiled_imports=compiled_imports,
    )
    iterable_input = model.map.get("iterable_input")
    st_map: MapStep[_BaseModel] = MapStep.from_pipeline(
        name=model.name, pipeline=body, iterable_input=str(iterable_input)
    )

    try:
        map_init = model.map.get("init") if isinstance(model.map, dict) else None
    except Exception:
        map_init = None
    if map_init is not None:
        try:
            import json as _jsonm

            m_ops_append: list[dict[str, object]] = []
            m_ops_set: list[dict[str, object]] = []
            m_ops_merge: list[dict[str, object]] = []
            if isinstance(map_init, dict):
                m_ops_append = list(map_init.get("append") or [])
                m_ops_set = list(map_init.get("set") or [])
                m_ops_merge = list(map_init.get("merge") or [])
            elif isinstance(map_init, list):
                for op in map_init:
                    try:
                        if not isinstance(op, dict):
                            continue
                        if "set" in op:
                            m_ops_set.append({"target": op.get("set"), "value": op.get("value")})
                        elif "append" in op:
                            _spec_a = op.get("append")
                            if isinstance(_spec_a, str):
                                m_ops_append.append({"target": _spec_a, "value": op.get("value")})
                            elif isinstance(_spec_a, dict):
                                d = _spec_a
                                m_ops_append.append(
                                    {"target": d.get("target"), "value": d.get("value")}
                                )
                        elif "merge" in op:
                            _spec_m = op.get("merge")
                            if isinstance(_spec_m, str):
                                m_ops_merge.append(
                                    {"target": _spec_m, "value": op.get("value") or "{}"}
                                )
                            elif isinstance(_spec_m, dict):
                                m_ops_merge.append(
                                    {
                                        "target": _spec_m.get("target"),
                                        "value": _spec_m.get("value") or "{}",
                                    }
                                )
                    except Exception:
                        continue

            def _map_init_ops(prev_output: object, ctx: object) -> None:
                for spec in m_ops_append:
                    try:
                        parent, key = _resolve_context_target(ctx, str(spec.get("target")))
                        if parent is None or key is None:
                            continue
                        val = _render_template_value(prev_output, ctx, spec.get("value", ""))
                        if isinstance(parent, dict):
                            seq = parent.get(key)
                        else:
                            seq = getattr(parent, key, None)
                        if not isinstance(seq, list):
                            seq = []
                            if isinstance(parent, dict):
                                parent[key] = seq
                            else:
                                try:
                                    setattr(parent, key, seq)
                                except Exception:
                                    continue
                        seq.append(val)
                    except Exception:
                        continue
                for spec in m_ops_set:
                    try:
                        parent, key = _resolve_context_target(ctx, str(spec.get("target")))
                        if parent is None or key is None:
                            continue
                        val = _render_template_value(prev_output, ctx, spec.get("value", ""))
                        if isinstance(parent, dict):
                            parent[key] = val
                        else:
                            try:
                                setattr(parent, key, val)
                            except Exception:
                                continue
                    except Exception:
                        continue
                for spec in m_ops_merge:
                    try:
                        parent, key = _resolve_context_target(ctx, str(spec.get("target")))
                        if parent is None or key is None:
                            continue
                        val_raw = _render_template_value(prev_output, ctx, spec.get("value", "{}"))
                        try:
                            val_text = (
                                val_raw
                                if isinstance(val_raw, (str, bytes, bytearray))
                                else str(val_raw)
                            )
                            val_obj = _jsonm.loads(val_text)
                        except Exception:
                            continue
                        if not isinstance(val_obj, dict):
                            continue
                        try:
                            cur = (
                                parent.get(key)
                                if isinstance(parent, dict)
                                else getattr(parent, key, None)
                            )
                        except Exception:
                            cur = None
                        if not isinstance(cur, dict):
                            cur = {}
                        cur.update(val_obj)
                        if isinstance(parent, dict):
                            parent[key] = cur
                        else:
                            try:
                                setattr(parent, key, cur)
                            except Exception:
                                continue
                    except Exception:
                        continue

            try:
                st_map.meta["compiled_init_ops"] = _map_init_ops
            except Exception:
                pass
        except Exception:
            pass

    try:
        map_finalize = model.map.get("finalize") if isinstance(model.map, dict) else None
    except Exception:
        map_finalize = None
    if isinstance(map_finalize, dict):
        try:
            output_template_spec = map_finalize.get("output_template")
            output_mapping_spec = map_finalize.get("output")
            finalize_mapper = None
            if isinstance(output_template_spec, str) and output_template_spec.strip():
                tpl = output_template_spec.strip()

                def _finalize_mapper(prev_output: object, ctx: object | None) -> object:
                    if ctx is None:
                        return prev_output
                    return _render_template_value(prev_output, ctx, tpl)

                finalize_mapper = _finalize_mapper
            elif isinstance(output_mapping_spec, dict) and output_mapping_spec:
                items = [(str(k), str(v)) for k, v in output_mapping_spec.items()]

                def _finalize_map(prev_output: object, ctx: object | None) -> object:
                    if ctx is None:
                        return {k: None for k, _ in items}
                    out: JSONObject = {}
                    for mk, mtpl in items:
                        try:
                            out[mk] = _render_template_value(prev_output, ctx, mtpl)
                        except Exception:
                            out[mk] = None
                    return out

                finalize_mapper = _finalize_map
            if finalize_mapper is not None:
                try:
                    st_map.meta["map_finalize_mapper"] = finalize_mapper
                except Exception:
                    pass
        except Exception:
            pass

    return st_map


__all__ = ["BuildBranch", "build_loop_step", "build_map_step"]
