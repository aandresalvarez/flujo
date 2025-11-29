from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from ..dsl import Pipeline, Step, StepConfig
from ..dsl.import_step import ImportStep, OutputMapping
from ..models import UsageLimits
from .loader_models import BlueprintError, BlueprintStepModel, ProcessingConfigModel
from .loader_resolution import (
    _PassthroughAgent,
    _import_object,
    _is_async_callable,
    _resolve_agent_entry,
    _resolve_plugins,
    _resolve_validators,
)
from .loader_steps_common import _finalize_step_types

BuildStep = Callable[..., Step[Any, Any]]


def build_hitl_step(model: BlueprintStepModel, step_config: StepConfig) -> Any:
    from ..dsl.step import HumanInTheLoopStep
    from .model_generator import generate_model_from_schema

    schema_model = None
    try:
        if isinstance(model.input_schema, dict):
            schema_model = generate_model_from_schema(f"{model.name}Input", model.input_schema)
    except Exception:
        schema_model = None

    return HumanInTheLoopStep(
        name=model.name,
        message_for_user=model.message,
        input_schema=schema_model,
        sink_to=model.sink_to,
        config=step_config,
    )


def build_cache_step(
    model: BlueprintStepModel,
    *,
    yaml_path: Optional[str],
    compiled_agents: Optional[Dict[str, Any]],
    compiled_imports: Optional[Dict[str, Any]],
    make_step_fn: BuildStep,
) -> Any:
    from flujo.steps.cache_step import CacheStep as _CacheStep

    if not model.wrapped_step:
        raise BlueprintError("cache step requires 'wrapped_step'")
    inner_spec = BlueprintStepModel.model_validate(model.wrapped_step)
    inner_step = make_step_fn(
        inner_spec,
        yaml_path=f"{yaml_path}.wrapped_step" if yaml_path else None,
        compiled_agents=compiled_agents,
        compiled_imports=compiled_imports,
    )
    return _CacheStep.cached(inner_step)


def build_agentic_loop_step(
    model: BlueprintStepModel,
    step_config: StepConfig,
) -> Any:
    try:
        from ...recipes.factories import make_agentic_loop_pipeline as _make_agentic
    except Exception as e:
        raise BlueprintError(f"Agentic loop factory is unavailable: {e}")

    if not model.planner:
        raise BlueprintError("agentic_loop requires 'planner'")
    planner_agent = _resolve_agent_entry(model.planner)

    reg_obj: Dict[str, Any] = {}
    if isinstance(model.registry, dict):
        reg_obj = dict(model.registry)
    elif isinstance(model.registry, str):
        try:
            obj = _import_object(model.registry)
            if isinstance(obj, dict):
                reg_obj = obj
            else:
                raise BlueprintError(
                    f"registry must resolve to a dict[str, Agent], got {type(obj)}"
                )
        except Exception as e:
            raise BlueprintError(f"Failed to resolve registry: {e}")
    else:
        raise BlueprintError("agentic_loop requires 'registry' (dict or import path)")

    try:
        p = _make_agentic(planner_agent=planner_agent, agent_registry=reg_obj)
    except Exception as e:
        raise BlueprintError(f"Failed to create agentic loop pipeline: {e}")

    if isinstance(model.output_template, str) and model.output_template.strip():
        try:
            step0 = p.steps[0]
            fmt_tpl = str(model.output_template)
            orig_mapper = getattr(step0, "loop_output_mapper", None)

            def _wrapped_output_mapper(output: Any, ctx: Optional[Any]) -> Any:
                base = output
                try:
                    if callable(orig_mapper):
                        base = orig_mapper(output, ctx)
                except Exception:
                    base = output
                try:
                    from ...utils.template_vars import (
                        TemplateContextProxy as _TCP,
                        get_steps_map_from_context as _get_steps,
                        StepValueProxy as _SVP,
                    )
                    from ...utils.prompting import AdvancedPromptFormatter as _Fmt

                    steps_map0 = _get_steps(ctx)
                    steps_wrapped = {
                        k: v if isinstance(v, _SVP) else _SVP(v) for k, v in steps_map0.items()
                    }
                    fmt_ctx = {
                        "context": _TCP(ctx, steps=steps_wrapped),
                        "previous_step": base,
                        "steps": steps_wrapped,
                    }
                    return _Fmt(fmt_tpl).format(**fmt_ctx)
                except Exception:
                    return base

            try:
                setattr(step0, "loop_output_mapper", _wrapped_output_mapper)
            except Exception:
                pass
        except Exception:
            pass

    try:
        return p.as_step(name=model.name)
    except Exception as e:
        raise BlueprintError(f"Failed to wrap agentic loop pipeline as step: {e}")


def build_basic_step(
    model: BlueprintStepModel,
    step_config: StepConfig,
    *,
    yaml_path: Optional[str],
    compiled_agents: Optional[Dict[str, Any]],
    compiled_imports: Optional[Dict[str, Any]],
    make_step_fn: BuildStep,
) -> Step[Any, Any]:
    _use_history_extra = None
    try:
        _use_history_extra = getattr(model, "_use_history_extra", None)
    except Exception:
        _use_history_extra = None
    agent_obj: Any = _PassthroughAgent()
    st: Optional[Step[Any, Any]] = None
    if model.uses:
        uses_spec = model.uses.strip()
        if uses_spec.startswith("agents."):
            if not compiled_agents:
                raise BlueprintError(f"No compiled agents available but step uses '{uses_spec}'")
            key = uses_spec.split(".", 1)[1]
            if key not in compiled_agents:
                raise BlueprintError(f"Unknown declarative agent referenced: {uses_spec}")
            agent_obj = compiled_agents[key]
            if _is_async_callable(agent_obj):
                st = Step.from_callable(
                    agent_obj,
                    name=model.name,
                    updates_context=model.updates_context,
                    validate_fields=model.validate_fields,
                    sink_to=model.sink_to,
                    **(step_config.model_dump() if hasattr(step_config, "model_dump") else {}),
                )
        elif uses_spec.startswith("imports."):
            if not compiled_imports:
                raise BlueprintError(f"No compiled imports available but step uses '{uses_spec}'")
            key = uses_spec.split(".", 1)[1]
            if key not in compiled_imports:
                raise BlueprintError(f"Unknown imported pipeline referenced: {uses_spec}")
            pipeline: Pipeline[Any, Any] = compiled_imports[key]
            import_cfg: Dict[str, Any] = dict(model.config or {})
            inherit_context = bool(import_cfg.get("inherit_context", False))
            input_to_val = str(import_cfg.get("input_to", "initial_prompt"))
            if input_to_val not in {"initial_prompt", "scratchpad", "both"}:
                input_to_val = "initial_prompt"
            input_scratchpad_key = import_cfg.get("input_scratchpad_key", "initial_input")
            outputs_raw = import_cfg.get("outputs", None)
            outputs: Optional[list[OutputMapping]]
            if outputs_raw is None:
                outputs = None
            elif isinstance(outputs_raw, dict):
                outputs = [
                    OutputMapping(child=str(k), parent=str(v)) for k, v in outputs_raw.items()
                ]
            elif isinstance(outputs_raw, list):
                outputs = []
                for item in outputs_raw:
                    if isinstance(item, dict):
                        child = item.get("child") or item.get("from")
                        parent = item.get("parent") or item.get("to")
                        if child is not None and parent is not None:
                            outputs.append(OutputMapping(child=str(child), parent=str(parent)))
                    else:
                        try:
                            # Support pairs like ["child.path", "parent.path"]
                            if isinstance(item, (list, tuple)) and len(item) == 2:
                                outputs.append(
                                    OutputMapping(child=str(item[0]), parent=str(item[1]))
                                )
                        except Exception:
                            continue
            else:
                outputs = None
            on_failure_val = str(import_cfg.get("on_failure", "abort"))
            if on_failure_val not in {"abort", "skip", "continue_with_default"}:
                on_failure_val = "abort"
            propagate_hitl = bool(import_cfg.get("propagate_hitl", True))
            inherit_conversation = bool(import_cfg.get("inherit_conversation", True))
            st = ImportStep(
                name=model.name,
                pipeline=pipeline,
                config=step_config,
                updates_context=model.updates_context,
                validate_fields=model.validate_fields,
                sink_to=model.sink_to,
                inherit_context=inherit_context,
                input_to=input_to_val,  # Literal enforced above
                input_scratchpad_key=input_scratchpad_key,
                outputs=outputs,
                inherit_conversation=inherit_conversation,
                propagate_hitl=propagate_hitl,
                on_failure=on_failure_val,
            )
            try:
                st.meta["import_alias"] = key
            except Exception:
                pass
        else:
            agent_obj = _resolve_agent_entry(uses_spec)
    elif model.agent is not None:
        try:
            if isinstance(model.agent, str):
                agent_obj = _resolve_agent_entry(model.agent)
            elif isinstance(model.agent, dict):
                agent_obj = _resolve_agent_entry(model.agent)
        except Exception:
            pass

    if st is None and callable(agent_obj):
        st = _build_callable_step(
            model=model,
            step_config=step_config,
            agent_obj=agent_obj,
        )
    if st is None:
        st = Step[Any, Any](
            name=model.name,
            agent=agent_obj,
            config=step_config,
            updates_context=model.updates_context,
            validate_fields=model.validate_fields,
            sink_to=model.sink_to,
        )

    _attach_processing_meta(st, model)
    try:
        if model.input is not None:
            st.meta["templated_input"] = model.input
    except Exception:
        pass
    try:
        if _use_history_extra is not None:
            st.meta["use_history"] = bool(_use_history_extra)
    except Exception:
        pass
    _finalize_step_types(st)
    if model.usage_limits is not None:
        try:
            st.usage_limits = UsageLimits(**model.usage_limits)
        except Exception:
            pass
    for plugin, priority in _resolve_plugins(model.plugins or []):
        try:
            st.plugins.append((plugin, priority))
        except Exception:
            pass
    for validator in _resolve_validators(model.validators or []):
        try:
            st.validators.append(validator)
        except Exception:
            pass
    if model.fallback is not None:
        try:
            st.fallback_step = make_step_fn(
                model.fallback
                if isinstance(model.fallback, dict)
                else BlueprintStepModel.model_validate(model.fallback),
                yaml_path=f"{yaml_path}.fallback" if yaml_path else None,
                compiled_agents=compiled_agents,
                compiled_imports=compiled_imports,
            )
        except Exception:
            pass
    if yaml_path:
        try:
            st.meta["yaml_path"] = yaml_path
        except Exception:
            pass
    return st


def _build_callable_step(
    *,
    model: BlueprintStepModel,
    step_config: StepConfig,
    agent_obj: Any,
) -> Optional[Step[Any, Any]]:
    st: Optional[Step[Any, Any]] = None
    callable_obj: Any = agent_obj
    _params_for_callable: Dict[str, Any] = {}
    skill_id_for_attr = None
    try:
        if isinstance(model.agent, dict):
            _params_for_callable = dict(model.agent.get("params") or {})
            skill_id_for_attr = model.agent.get("id", "")
    except Exception:
        _params_for_callable = {}
    try:
        import inspect as __inspect

        is_builtin = False
        try:
            if isinstance(model.agent, dict):
                skill_id_for_attr = model.agent.get("id", "")
                is_builtin = isinstance(skill_id_for_attr, str) and skill_id_for_attr.startswith(
                    "flujo.builtins."
                )
        except (AttributeError, KeyError, TypeError):
            is_builtin = False

        def _with_params(func: Any) -> Any:
            _params_for_callable_local = dict(_params_for_callable)

            def _runner(data: Any, *args: Any, **kwargs: Any) -> Any:
                try:
                    call_kwargs = dict(_params_for_callable_local)
                    call_kwargs.update(kwargs)
                    if isinstance(func, _PassthroughAgent):
                        return data
                    sig = __inspect.signature(func)
                    if any(
                        p.kind == __inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                    ):
                        if model.input is not None:
                            call_kwargs.update(data if isinstance(data, dict) else {})
                        return func(**call_kwargs)
                    if "context" in sig.parameters:
                        call_kwargs.pop("pipeline_context", None)
                        call_kwargs.pop("previous_step", None)
                    if "pipeline_context" not in sig.parameters:
                        call_kwargs.pop("pipeline_context", None)
                    bound = sig.bind_partial(**call_kwargs)
                    if "data" in sig.parameters:
                        bound.arguments.setdefault("data", data)
                    if "input" in sig.parameters:
                        bound.arguments.setdefault("input", data)
                    if "value" in sig.parameters:
                        bound.arguments.setdefault("value", data)
                    if "payload" in sig.parameters:
                        bound.arguments.setdefault("payload", data)
                    if "previous_step" in sig.parameters and "previous_step" not in bound.arguments:
                        bound.arguments["previous_step"] = kwargs.get("previous_step")
                    if "context" in sig.parameters and "context" not in bound.arguments:
                        bound.arguments["context"] = kwargs.get("context")
                    result = func(*bound.args, **bound.kwargs)
                    if __inspect.isawaitable(result):
                        return result
                    return result
                except TypeError as e:
                    if is_builtin:
                        raise BlueprintError(
                            f"Builtin skill {getattr(func, '__name__', 'unknown')} failed: {e}"
                        ) from e
                    result = func(data, **dict(_params_for_callable))
                    if __inspect.isawaitable(result):
                        return result
                    return result

            try:
                if skill_id_for_attr:
                    _runner.__name__ = skill_id_for_attr
                elif hasattr(func, "__name__"):
                    _runner.__name__ = func.__name__
            except (AttributeError, TypeError):
                pass
            return _runner

        should_wrap = callable(agent_obj) and (_params_for_callable or is_builtin)
        if should_wrap:
            callable_obj = _with_params(agent_obj)
            if skill_id_for_attr and not hasattr(callable_obj, "__name__"):
                try:
                    callable_obj.__name__ = skill_id_for_attr
                except (AttributeError, TypeError):
                    pass
    except Exception:
        callable_obj = agent_obj

    if _is_async_callable(callable_obj):
        st = Step.from_callable(
            callable_obj,
            name=model.name,
            updates_context=model.updates_context,
            validate_fields=model.validate_fields,
            sink_to=model.sink_to,
            **(step_config.model_dump() if hasattr(step_config, "model_dump") else {}),
        )
        if skill_id_for_attr and st is not None and st.agent is not None:
            try:
                st.agent.__name__ = skill_id_for_attr
            except (AttributeError, TypeError):
                pass
        return st
    return None


def _attach_processing_meta(st: Step[Any, Any], model: BlueprintStepModel) -> None:
    try:
        if isinstance(model.processing, dict) and model.processing:
            try:
                pc = ProcessingConfigModel.model_validate(model.processing)
                proc_dict = pc.model_dump(exclude_none=True, by_alias=True)
            except Exception as e:
                raise BlueprintError(f"Invalid processing configuration: {e}") from e
            st.meta.setdefault("processing", {})
            st.meta["processing"].update(proc_dict)
    except Exception:
        pass


__all__ = [
    "BuildStep",
    "build_agentic_loop_step",
    "build_basic_step",
    "build_cache_step",
    "build_hitl_step",
]
