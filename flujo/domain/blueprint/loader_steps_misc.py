from __future__ import annotations

from collections.abc import Callable
from typing import Optional, TypeAlias, TypeGuard

from ..dsl import Pipeline, Step, StepConfig
from ..dsl.step import InvariantRule
from ..dsl.import_step import ImportStep, OutputMapping
from ..models import BaseModel, UsageLimits
from ..resources import AppResources
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
from .model_generator import generate_model_from_schema, _python_type_for_json_schema
from flujo.type_definitions.common import JSONObject

AnyStep: TypeAlias = Step[object, object]
AnyPipeline: TypeAlias = Pipeline[object, object]
CompiledAgents: TypeAlias = dict[str, object]
CompiledImports: TypeAlias = dict[str, AnyPipeline]
BuildStep = Callable[..., AnyStep]
BuildBranch = Callable[..., AnyPipeline]

_CallableObj: TypeAlias = Callable[..., object]


def _is_callable_obj(obj: object) -> TypeGuard[_CallableObj]:
    return callable(obj)


def _is_invariant_callable(obj: object) -> TypeGuard[Callable[..., bool]]:
    return callable(obj)


def _is_invariant_rule(obj: object) -> TypeGuard[InvariantRule]:
    return isinstance(obj, str) or _is_invariant_callable(obj)


def _resolve_compiled_reference(
    value: object,
    *,
    label: str,
    compiled_agents: CompiledAgents | None,
    compiled_imports: CompiledImports | None,
) -> object | None:
    if not isinstance(value, str):
        return None
    if value.startswith("agents."):
        if not compiled_agents:
            raise BlueprintError(f"No compiled agents available for '{label}'")
        key = value.split(".", 1)[1]
        if key not in compiled_agents:
            raise BlueprintError(f"Unknown agent reference '{value}' for '{label}'")
        return compiled_agents[key]
    if value.startswith("imports."):
        if not compiled_imports:
            raise BlueprintError(f"No compiled imports available for '{label}'")
        key = value.split(".", 1)[1]
        if key not in compiled_imports:
            raise BlueprintError(f"Unknown import reference '{value}' for '{label}'")
        return compiled_imports[key]
    return None


def _resolve_spec_value(
    spec: object,
    *,
    label: str,
    compiled_agents: CompiledAgents | None = None,
    compiled_imports: CompiledImports | None = None,
) -> object:
    resolved = _resolve_compiled_reference(
        spec,
        label=label,
        compiled_agents=compiled_agents,
        compiled_imports=compiled_imports,
    )
    if resolved is not None:
        return resolved
    if isinstance(spec, (str, dict)):
        try:
            return _resolve_agent_entry(spec)
        except Exception:
            if isinstance(spec, dict):
                path = spec.get("path")
                if isinstance(path, str):
                    return _import_object(path)
                raise
            if isinstance(spec, str):
                return _import_object(spec)
            raise
    return spec


def build_hitl_step(model: BlueprintStepModel, step_config: StepConfig) -> AnyStep:
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
    compiled_agents: CompiledAgents | None,
    compiled_imports: CompiledImports | None,
    make_step_fn: BuildStep,
) -> AnyStep:
    from flujo.domain.dsl.cache_step import CacheStep as _CacheStep

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
) -> AnyStep:
    try:
        from ...recipes.factories import make_agentic_loop_pipeline as _make_agentic
    except Exception as e:
        raise BlueprintError(f"Agentic loop factory is unavailable: {e}")

    if not model.planner:
        raise BlueprintError("agentic_loop requires 'planner'")
    planner_agent = _resolve_agent_entry(model.planner)

    reg_obj: JSONObject = {}
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
        p = _make_agentic(planner_agent=planner_agent, agent_registry=reg_obj)  # type: ignore[arg-type]
    except Exception as e:
        raise BlueprintError(f"Failed to create agentic loop pipeline: {e}")

    if isinstance(model.output_template, str) and model.output_template.strip():
        try:
            step0 = p.steps[0]
            fmt_tpl = str(model.output_template)
            orig_mapper = getattr(step0, "loop_output_mapper", None)

            def _wrapped_output_mapper(output: object, ctx: BaseModel | None) -> object:
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
        inner = p.as_step(name=model.name)

        async def _runner(
            data: object,
            *,
            context: BaseModel | None = None,
            resources: AppResources | None = None,
        ) -> object:
            if not isinstance(data, str):
                raise BlueprintError("agentic_loop steps expect string input")
            return await inner.arun(data, context=context, resources=resources)

        return Step.from_callable(
            _runner,
            name=model.name,
            updates_context=model.updates_context,
            validate_fields=model.validate_fields,
            sink_to=model.sink_to,
            config=step_config,
        )
    except Exception as e:
        raise BlueprintError(f"Failed to wrap agentic loop pipeline as step: {e}")


def _resolve_tree_search_target(
    spec: object,
    *,
    label: str,
    compiled_agents: CompiledAgents | None,
    compiled_imports: CompiledImports | None,
) -> object:
    if spec is None:
        raise BlueprintError(f"tree_search requires '{label}'")
    return _resolve_spec_value(
        spec,
        label=label,
        compiled_agents=compiled_agents,
        compiled_imports=compiled_imports,
    )


def _resolve_callable_spec(
    spec: object,
    *,
    label: str,
) -> _CallableObj | None:
    if spec is None:
        return None
    obj = _resolve_spec_value(spec, label=label)
    if not callable(obj):
        raise BlueprintError(f"{label} must be callable")
    return obj


def build_tree_search_step(
    model: BlueprintStepModel,
    step_config: StepConfig,
    *,
    yaml_path: Optional[str],
    compiled_agents: CompiledAgents | None,
    compiled_imports: CompiledImports | None,
) -> AnyStep:
    _ = yaml_path
    from ..dsl.tree_search import TreeSearchStep

    proposer_obj = _resolve_tree_search_target(
        model.proposer,
        label="proposer",
        compiled_agents=compiled_agents,
        compiled_imports=compiled_imports,
    )
    evaluator_obj = _resolve_tree_search_target(
        model.evaluator,
        label="evaluator",
        compiled_agents=compiled_agents,
        compiled_imports=compiled_imports,
    )
    cost_fn = _resolve_callable_spec(model.cost_function, label="cost_function")
    candidate_validator = _resolve_callable_spec(
        model.candidate_validator, label="candidate_validator"
    )
    discovery_agent: object | None = None
    if model.discovery_agent is not None:
        discovery_agent = _resolve_tree_search_target(
            model.discovery_agent,
            label="discovery_agent",
            compiled_agents=compiled_agents,
            compiled_imports=compiled_imports,
        )

    static_invariants: list[InvariantRule] = []
    if model.static_invariants:
        for rule in model.static_invariants:
            if _is_invariant_rule(rule):
                static_invariants.append(rule)
                continue
            if isinstance(rule, dict):
                resolved = _resolve_callable_spec(rule, label="static_invariant")
                if resolved is not None and _is_invariant_callable(resolved):
                    static_invariants.append(resolved)
                    continue
            raise BlueprintError(f"Invalid static invariant: {rule!r}")

    kwargs: dict[str, object] = {}
    if model.branching_factor is not None:
        kwargs["branching_factor"] = int(model.branching_factor)
    if model.beam_width is not None:
        kwargs["beam_width"] = int(model.beam_width)
    if model.max_depth is not None:
        kwargs["max_depth"] = int(model.max_depth)
    if model.max_iterations is not None:
        kwargs["max_iterations"] = int(model.max_iterations)
    if model.path_max_tokens is not None:
        kwargs["path_max_tokens"] = int(model.path_max_tokens)
    if model.goal_score_threshold is not None:
        kwargs["goal_score_threshold"] = float(model.goal_score_threshold)
    if model.require_goal is not None:
        kwargs["require_goal"] = bool(model.require_goal)

    return TreeSearchStep(
        name=model.name,
        proposer=proposer_obj,
        evaluator=evaluator_obj,
        cost_function=cost_fn,
        candidate_validator=candidate_validator,
        discovery_agent=discovery_agent,
        static_invariants=static_invariants,
        config=step_config,
        **kwargs,
    )


def build_basic_step(
    model: BlueprintStepModel,
    step_config: StepConfig,
    *,
    yaml_path: Optional[str],
    compiled_agents: CompiledAgents | None,
    compiled_imports: CompiledImports | None,
    make_step_fn: BuildStep,
) -> AnyStep:
    _use_history_extra = None
    try:
        _use_history_extra = getattr(model, "_use_history_extra", None)
    except Exception:
        _use_history_extra = None
    agent_obj: object = _PassthroughAgent()
    st: AnyStep | None = None
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
            pipeline = compiled_imports[key]
            import_cfg: JSONObject = dict(model.config or {})
            inherit_context = bool(import_cfg.get("inherit_context", False))
            input_to_val = str(import_cfg.get("input_to", "initial_prompt"))
            allowed_input_to = {"initial_prompt", "import_artifacts", "both"}
            if input_to_val not in allowed_input_to:
                raise BlueprintError(
                    f"ImportStep config.input_to must be one of {sorted(allowed_input_to)}, got "
                    f"'{input_to_val}'. scratchpad has been removed."
                )
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
        st = Step[object, object](
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
        if isinstance(model.meta, dict):
            st.meta.update(model.meta)
    except Exception:
        pass
    # Enforce adapter allowlist metadata even when supplied via YAML/meta.
    meta = getattr(st, "meta", None)
    if isinstance(meta, dict) and meta.get("is_adapter"):
        adapter_id = meta.get("adapter_id")
        adapter_allow = meta.get("adapter_allow")
        if not adapter_id or not adapter_allow:
            raise BlueprintError(
                "Adapter steps must include adapter_id and adapter_allow (allowlist token)."
            )
    # Propagate explicit JSON schemas to step IO types for validation
    try:
        if isinstance(model.input_schema, dict):
            schema_type = model.input_schema.get("type")
            if schema_type and schema_type != "object":
                ann = _python_type_for_json_schema(model.input_schema)
                st.__step_input_type__ = ann if isinstance(ann, type) else object
            else:
                schema_model_in = generate_model_from_schema(
                    f"{model.name}Input", model.input_schema
                )
                st.__step_input_type__ = schema_model_in
    except Exception:
        pass
    try:
        if isinstance(model.output_schema, dict):
            schema_type = model.output_schema.get("type")
            if schema_type and schema_type != "object":
                ann = _python_type_for_json_schema(model.output_schema)
                st.__step_output_type__ = ann if isinstance(ann, type) else object
            else:
                schema_model_out = generate_model_from_schema(
                    f"{model.name}Output", model.output_schema
                )
                st.__step_output_type__ = schema_model_out
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
    agent_obj: object,
) -> AnyStep | None:
    st: AnyStep | None = None
    callable_obj: object = agent_obj
    _params_for_callable: JSONObject = {}
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

        def _with_params(func: Callable[..., object]) -> Callable[..., object]:
            _params_for_callable_local = dict(_params_for_callable)

            def _runner(data: object, *args: object, **kwargs: object) -> object:
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

        if _params_for_callable or is_builtin:
            if _is_callable_obj(agent_obj):
                callable_obj = _with_params(agent_obj)
            if skill_id_for_attr and not hasattr(callable_obj, "__name__"):
                try:
                    setattr(callable_obj, "__name__", skill_id_for_attr)
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
                setattr(st.agent, "__name__", skill_id_for_attr)
            except (AttributeError, TypeError):
                pass
        return st
    return None


def _attach_processing_meta(st: AnyStep, model: BlueprintStepModel) -> None:
    processing = getattr(model, "processing", None)
    if processing is None:
        return

    if isinstance(processing, dict):
        pc = ProcessingConfigModel.model_validate(processing)
    else:
        pc = processing

    proc_dict = pc.model_dump(exclude_none=True, by_alias=True)
    if proc_dict:
        st.meta.setdefault("processing", {})
        st.meta["processing"].update(proc_dict)


def build_state_machine_step(
    model: BlueprintStepModel,
    step_config: StepConfig,
    *,
    yaml_path: Optional[str],
    compiled_agents: CompiledAgents | None,
    compiled_imports: CompiledImports | None,
    build_branch: BuildBranch,
) -> AnyStep:
    from ..dsl.state_machine import StateMachineStep

    if not model.start_state:
        raise BlueprintError("StateMachine requires 'start_state'")
    if not isinstance(model.states, dict) or not model.states:
        raise BlueprintError("StateMachine.states must be a mapping of state → steps")

    coerced_states: dict[str, AnyPipeline] = {}
    for state_name, branch_spec in model.states.items():
        state_key = str(state_name)
        coerced_states[state_key] = build_branch(
            branch_spec,
            base_path=f"{yaml_path}.states.{state_key}" if yaml_path else None,
            compiled_agents=compiled_agents,
            compiled_imports=compiled_imports,
        )

    end_states = [str(x) for x in (model.end_states or [])]
    transitions_raw = list(model.transitions or [])

    sm = StateMachineStep(
        name=model.name,
        states=coerced_states,
        start_state=str(model.start_state),
        end_states=end_states,
        transitions=transitions_raw,
        config=step_config,
    )
    if yaml_path:
        try:
            sm.meta["yaml_path"] = yaml_path
        except Exception:
            pass
    return sm


__all__ = [
    "BuildStep",
    "build_agentic_loop_step",
    "build_basic_step",
    "build_cache_step",
    "build_hitl_step",
    "build_state_machine_step",
]
