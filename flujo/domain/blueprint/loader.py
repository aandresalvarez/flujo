from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Literal, Tuple
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_validator,
    AliasChoices,
)
import re
import yaml

from ..dsl import Pipeline, Step, StepConfig, ParallelStep
from ...exceptions import ConfigurationError
from ...infra.skill_registry import get_skill_registry
from ..models import UsageLimits
from .schema import AgentModel


class BlueprintError(ConfigurationError):
    pass


class BlueprintStepModel(BaseModel):
    """Declarative step spec (minimal v0).

    This intentionally supports only a safe subset to start, then we'll extend.
    """

    kind: Literal[
        "step",
        "parallel",
        "conditional",
        "loop",
        "map",
        "dynamic_router",
        "hitl",
        "cache",
    ] = Field(default="step")
    # Accept both 'name' and legacy 'step' keys for step name
    name: str = Field(validation_alias=AliasChoices("name", "step"))
    agent: Optional[Union[str, Dict[str, Any]]] = None
    # New: declarative reference to either a compiled agent (agents.<name>) or python path
    uses: Optional[str] = None
    # New: optional input templating/value override for this step
    input: Optional[Any] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    # Step flags
    updates_context: bool = False
    validate_fields: bool = False
    # Parallel / Conditional
    branches: Optional[Dict[str, Any]] = None
    # Conditional only (v0: simple string identifier for callable resolution)
    condition: Optional[str] = None
    default_branch: Optional[Any] = None
    # Loop only (v0)
    loop: Optional[Dict[str, Any]] = (
        None  # { body: [...], max_loops: int, exit_condition: str, initial_input_mapper: str, iteration_input_mapper: str, loop_output_mapper: str }
    )
    # Map only
    map: Optional[Dict[str, Any]] = None  # { iterable_input: str, body: [...] }
    # Dynamic router only
    router: Optional[Dict[str, Any]] = None  # { router_agent: str|dict, branches: {name: [...] } }
    # Fallback step (optional)
    fallback: Optional[Dict[str, Any]] = None
    # Usage limits
    usage_limits: Optional[Dict[str, Any]] = None
    # Plugins and validators
    plugins: Optional[List[Union[str, Dict[str, Any]]]] = None  # str path or {path, priority}
    validators: Optional[List[str]] = None  # list of import strings
    merge_strategy: Optional[str] = None
    on_branch_failure: Optional[str] = None
    context_include_keys: Optional[List[str]] = None
    field_mapping: Optional[Dict[str, List[str]]] = None
    ignore_branch_names: Optional[bool] = None
    # HITL specific (optional)
    message: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None
    # Cache specific (optional)
    wrapped_step: Optional[Dict[str, Any]] = None

    # ----------------------------
    # Field validators (compile-time safety for FSD-016 / Gap #9)
    # ----------------------------

    @field_validator("uses")
    @classmethod
    def _validate_uses_format(cls, value: Optional[str]) -> Optional[str]:
        """Validate that 'uses' is either 'agents.<name>' or a Python import path.

        - agents.<name> where <name> matches ^[A-Za-z_][A-Za-z0-9_]*$
        - imports.<alias> where <alias> matches ^[A-Za-z_][A-Za-z0-9_]*$
        - import path formats supported by loader: 'module:attr' or 'module.attr'
        """
        if value is None:
            return value
        uses_spec = value.strip()
        if uses_spec.startswith("agents."):
            # Must be agents.<identifier>
            m = re.fullmatch(r"agents\.([A-Za-z_][A-Za-z0-9_]*)", uses_spec)
            if not m:
                raise ValueError("uses must be 'agents.<name>' where <name> is a valid identifier")
            return uses_spec
        if uses_spec.startswith("imports."):
            m = re.fullmatch(r"imports\.([A-Za-z_][A-Za-z0-9_]*)", uses_spec)
            if not m:
                raise ValueError(
                    "uses must be 'imports.<alias>' where <alias> is a valid identifier"
                )
            return uses_spec
        # Otherwise require a plausible import path: module(.sub)*(:attr)?
        # Keep conservative to avoid false positives; allow letters, digits, and underscores
        import_path_pattern = re.compile(
            r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*(?::[A-Za-z_][A-Za-z0-9_]*)?$"
        )
        if not import_path_pattern.fullmatch(uses_spec):
            raise ValueError(
                "uses must be 'agents.<name>' or a valid Python import path like 'pkg.mod:attr'"
            )
        return uses_spec


class BlueprintPipelineModel(BaseModel):
    version: str = Field(default="0.1")
    # Allow arbitrary step dicts so custom primitives with extra fields are preserved
    steps: List[Dict[str, Any]]
    # New: top-level declarative agents section
    agents: Optional[Dict[str, "AgentModel"]] = None
    # New: top-level imports section mapping alias -> relative/absolute YAML path
    imports: Optional[Dict[str, str]] = None

    # Cross-model validation: ensure any 'uses: agents.<name>' references a declared agent
    @model_validator(mode="after")
    def _validate_agent_references(self) -> "BlueprintPipelineModel":
        if not self.steps:
            return self
        declared_agents = set((self.agents or {}).keys())
        declared_imports = set((self.imports or {}).keys())
        for idx, step in enumerate(self.steps):
            try:
                uses = None
                if isinstance(step, dict):
                    uses = step.get("uses")
                else:
                    uses = getattr(step, "uses", None)
                if isinstance(uses, str) and uses.startswith("agents."):
                    name = uses.split(".", 1)[1]
                    if name not in declared_agents:
                        raise ValueError(
                            f"Unknown declarative agent referenced at steps[{idx}].uses: {uses}"
                        )
                if isinstance(uses, str) and uses.startswith("imports."):
                    alias = uses.split(".", 1)[1]
                    if alias not in declared_imports:
                        raise ValueError(
                            f"Unknown imported pipeline alias at steps[{idx}].uses: {uses}"
                        )
            except Exception:
                # Best-effort validation; ignore shape issues to allow custom primitives
                pass
        return self


def _normalize_merge_strategy(value: Optional[str]) -> Any:
    # Access enum members at runtime to avoid mypy import/type alias issues
    from ..dsl.step import MergeStrategy as _MS

    if value is None:
        return _MS.CONTEXT_UPDATE
    try:
        return _MS[value.upper()]
    except Exception as e:
        raise BlueprintError(f"Invalid merge_strategy: {value}") from e


def _normalize_branch_failure(value: Optional[str]) -> Any:
    from ..dsl.step import BranchFailureStrategy as _BFS

    if value is None:
        return _BFS.PROPAGATE
    try:
        return _BFS[value.upper()]
    except Exception as e:
        raise BlueprintError(f"Invalid on_branch_failure: {value}") from e


def _finalize_step_types(step_obj: Step[Any, Any]) -> None:
    """Best-effort static type assignment for pipeline validation.

    - If the agent is a callable wrapper created by Step.from_callable, extract the
      original function from '_step_callable' and analyze it.
    - Else, try to analyze the agent object itself.
    """
    try:
        from flujo.signature_tools import analyze_signature as _analyze
        import inspect as _inspect

        def _is_default_type(t: Any) -> bool:
            return t is object or str(t) == "typing.Any"

        agent_obj = getattr(step_obj, "agent", None)
        fn = getattr(agent_obj, "_step_callable", None)
        # Unwrap bound method to original function if needed
        if hasattr(fn, "__func__"):
            try:
                fn = getattr(fn, "__func__")
            except Exception:
                pass
        if fn is None:
            # If the agent itself is a function/method, analyze it directly
            try:
                if _inspect.isfunction(agent_obj) or _inspect.ismethod(agent_obj):
                    fn = agent_obj
            except Exception:
                fn = None

        if fn is not None:
            sig = _analyze(fn)
            # Only overwrite when defaults are present
            try:
                if _is_default_type(getattr(step_obj, "__step_input_type__", object)):
                    step_obj.__step_input_type__ = getattr(sig, "input_type", object)
            except Exception:
                pass
            try:
                if _is_default_type(getattr(step_obj, "__step_output_type__", object)):
                    step_obj.__step_output_type__ = getattr(sig, "output_type", object)
            except Exception:
                pass
    except Exception:
        pass


def _make_step_from_blueprint(
    model: Any,
    *,
    yaml_path: Optional[str] = None,
    compiled_agents: Optional[Dict[str, Any]] = None,
    compiled_imports: Optional[Dict[str, Any]] = None,
) -> Step[Any, Any]:
    # Support both native BlueprintStepModel and raw dict for custom primitives
    if isinstance(model, dict):
        kind_val = str(model.get("kind", "step"))
        # Built-in kinds handled by existing logic via typed model
        if kind_val in {
            "step",
            "parallel",
            "conditional",
            "loop",
            "map",
            "dynamic_router",
            "hitl",
            "cache",
        }:
            model = BlueprintStepModel.model_validate(model)
        else:
            # Attempt framework registry lookup for custom primitives
            try:
                from ...framework import registry as _fwreg

                step_cls = _fwreg.get_step_class(kind_val)
            except Exception:
                step_cls = None
            if step_cls is not None:
                try:
                    # Let the custom Step class validate its own fields
                    step_obj = step_cls.model_validate(model)
                except Exception as e:
                    raise BlueprintError(
                        f"Failed to instantiate custom step for kind '{kind_val}': {e}"
                    )
                # Attach yaml_path for telemetry if available
                if yaml_path:
                    try:
                        step_obj.meta["yaml_path"] = yaml_path
                    except Exception:
                        pass
                return step_obj
            # No custom mapping found; raise a clear error for unknown kind
            raise BlueprintError(f"Unknown step kind: {kind_val}")

    # For v0: agent may be None; if provided, we resolve later via import string in a follow-up.
    # Normalize step config supporting 'timeout' alias -> 'timeout_s'
    if hasattr(model, "config") and model.config:
        cfg_dict = dict(model.config)
        if "timeout" in cfg_dict and "timeout_s" not in cfg_dict:
            try:
                cfg_dict["timeout_s"] = float(cfg_dict.pop("timeout"))
            except Exception:
                cfg_dict.pop("timeout", None)
        step_config = StepConfig(**cfg_dict)
    else:
        step_config = StepConfig()
    if getattr(model, "kind", None) == "parallel":
        if not model.branches:
            raise BlueprintError("parallel step requires branches")
        # Branch values are nested steps or pipelines in YAML; support list-of-steps or single step.
        branches_map: Dict[str, Pipeline[Any, Any]] = {}
        for branch_name, branch_spec in model.branches.items():
            branches_map[branch_name] = _build_pipeline_from_branch(
                branch_spec,
                base_path=f"{yaml_path}.branches.{branch_name}" if yaml_path else None,
                compiled_agents=compiled_agents,
            )
        return ParallelStep(
            name=model.name,
            branches=branches_map,
            context_include_keys=model.context_include_keys,
            merge_strategy=_normalize_merge_strategy(model.merge_strategy),
            on_branch_failure=_normalize_branch_failure(model.on_branch_failure),
            field_mapping=model.field_mapping,
            ignore_branch_names=bool(model.ignore_branch_names)
            if model.ignore_branch_names is not None
            else False,
            config=step_config,
        )
    elif getattr(model, "kind", None) == "conditional":
        from ..dsl.conditional import ConditionalStep

        if not model.branches:
            raise BlueprintError("conditional step requires branches")
        branches_map2: Dict[Any, Pipeline[Any, Any]] = {}
        for key, branch_spec in model.branches.items():
            branches_map2[key] = _build_pipeline_from_branch(
                branch_spec, compiled_agents=compiled_agents
            )
        # Resolve condition callable explicitly to avoid scope issues
        if model.condition:
            _cond_callable = _import_object(model.condition)
        else:

            def _cond_callable(output: Any, _ctx: Optional[Any]) -> Any:
                # trivial placeholder: pass through output as branch key if present, else use first key
                return output if output in branches_map2 else next(iter(branches_map2))

        default_branch = (
            _build_pipeline_from_branch(
                model.default_branch,
                base_path=f"{yaml_path}.default_branch" if yaml_path else None,
                compiled_agents=compiled_agents,
            )
            if model.default_branch
            else None
        )
        return ConditionalStep(
            name=model.name,
            condition_callable=_cond_callable,
            branches=branches_map2,
            default_branch_pipeline=default_branch,
            config=step_config,
        )
    elif getattr(model, "kind", None) == "loop":
        from ..dsl.loop import LoopStep

        if not model.loop or "body" not in model.loop:
            raise BlueprintError("loop step requires loop.body")
        body = _build_pipeline_from_branch(
            model.loop.get("body"),
            base_path=f"{yaml_path}.loop.body" if yaml_path else None,
            compiled_agents=compiled_agents,
        )
        max_loops = model.loop.get("max_loops")

        # --- PROPOSED CHANGE ---
        # Resolve all optional callable overrides
        _initial_mapper = None
        if model.loop.get("initial_input_mapper"):
            _initial_mapper = _import_object(model.loop["initial_input_mapper"])

        _iter_mapper = None
        if model.loop.get("iteration_input_mapper"):
            _iter_mapper = _import_object(model.loop["iteration_input_mapper"])

        _output_mapper = None
        if model.loop.get("loop_output_mapper"):
            _output_mapper = _import_object(model.loop["loop_output_mapper"])
        # --- END CHANGE ---

        # Optional callable overrides
        if model.loop.get("exit_condition"):
            _exit_condition = _import_object(model.loop["exit_condition"])  # runtime import
        else:

            def _exit_condition(
                _output: Any, _ctx: Optional[Any], *, _state: Optional[Dict[str, int]] = None
            ) -> bool:
                if _state is None:
                    _state = {"count": 0}
                _state["count"] += 1
                if isinstance(max_loops, int) and max_loops > 0:
                    return _state["count"] >= max_loops
                return _state["count"] >= 1

        return LoopStep(
            name=model.name,
            loop_body_pipeline=body,
            exit_condition_callable=_exit_condition,
            max_retries=max(1, int(max_loops)) if isinstance(max_loops, int) else 1,
            config=step_config,
            # --- PASS THE RESOLVED MAPPERS ---
            initial_input_to_loop_body_mapper=_initial_mapper,
            iteration_input_mapper=_iter_mapper,
            loop_output_mapper=_output_mapper,
        )
    elif getattr(model, "kind", None) == "map":
        from ..dsl.loop import MapStep

        if not model.map or "iterable_input" not in model.map or "body" not in model.map:
            raise BlueprintError("map step requires map.iterable_input and map.body")
        body = _build_pipeline_from_branch(
            model.map.get("body"),
            base_path=f"{yaml_path}.map.body" if yaml_path else None,
            compiled_agents=compiled_agents,
        )
        iterable_input = model.map.get("iterable_input")
        return MapStep.from_pipeline(
            name=model.name, pipeline=body, iterable_input=str(iterable_input)
        )
    elif getattr(model, "kind", None) == "dynamic_router":
        from ..dsl.dynamic_router import DynamicParallelRouterStep

        if not model.router or "router_agent" not in model.router or "branches" not in model.router:
            raise BlueprintError("dynamic_router requires router.router_agent and router.branches")
        router_agent = _resolve_agent_entry(model.router.get("router_agent") or "")
        branches_router: Dict[str, Pipeline[Any, Any]] = {}
        for bname, bspec in model.router.get("branches", {}).items():
            branches_router[bname] = _build_pipeline_from_branch(
                bspec,
                base_path=f"{yaml_path}.router.branches.{bname}" if yaml_path else None,
                compiled_agents=compiled_agents,
            )
        return DynamicParallelRouterStep(
            name=model.name,
            router_agent=router_agent,
            branches=branches_router,
            config=step_config,
        )
    elif getattr(model, "kind", None) == "hitl":
        # Human-in-the-loop step compiled from declarative YAML
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
            config=step_config,
        )
    elif getattr(model, "kind", None) == "cache":
        # Declarative cache wrapper for inner step
        from flujo.steps.cache_step import CacheStep as _CacheStep

        if not model.wrapped_step:
            raise BlueprintError("cache step requires 'wrapped_step'")
        inner_spec = BlueprintStepModel.model_validate(model.wrapped_step)
        inner_step = _make_step_from_blueprint(
            inner_spec,
            yaml_path=f"{yaml_path}.wrapped_step" if yaml_path else None,
            compiled_agents=compiled_agents,
            compiled_imports=compiled_imports,
        )
        return _CacheStep.cached(inner_step)
    else:
        # Simple step; resolve agent if provided, otherwise passthrough.
        agent_obj: Any = _PassthroughAgent()
        st: Optional[Step[Any, Any]] = None
        # Priority: 'uses' -> 'agent' -> passthrough
        if model.uses:
            uses_spec = model.uses.strip()
            if uses_spec.startswith("agents."):
                if not compiled_agents:
                    raise BlueprintError(
                        f"No compiled agents available but step uses '{uses_spec}'"
                    )
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
                        **(step_config.model_dump() if hasattr(step_config, "model_dump") else {}),
                    )
            elif uses_spec.startswith("imports."):
                # Wrap precompiled sub-pipeline as a single step
                if not compiled_imports:
                    raise BlueprintError(
                        f"No compiled imports available but step uses '{uses_spec}'"
                    )
                alias = uses_spec.split(".", 1)[1]
                if alias not in compiled_imports:
                    raise BlueprintError(f"Unknown imported pipeline referenced: {uses_spec}")
                try:
                    _sub_pipeline = compiled_imports[alias]
                    st = _sub_pipeline.as_step(name=model.name)
                except Exception as e:
                    raise BlueprintError(f"Failed to wrap imported pipeline '{alias}' as step: {e}")
            else:
                try:
                    _imported = _import_object(uses_spec)
                    agent_obj = _imported
                    if _is_async_callable(agent_obj):
                        st = Step.from_callable(
                            agent_obj,
                            name=model.name,
                            updates_context=model.updates_context,
                            validate_fields=model.validate_fields,
                            **(
                                step_config.model_dump()
                                if hasattr(step_config, "model_dump")
                                else {}
                            ),
                        )
                except Exception as e:
                    raise BlueprintError(f"Failed to resolve uses='{uses_spec}': {e}")
        elif model.agent:
            try:
                import inspect as _inspect

                if isinstance(model.agent, str):
                    _fn = _import_object(model.agent)
                    if _inspect.isfunction(_fn) or _is_async_callable(_fn):
                        st = Step.from_callable(
                            _fn,
                            name=model.name,
                            updates_context=model.updates_context,
                            validate_fields=model.validate_fields,
                            **(
                                step_config.model_dump()
                                if hasattr(step_config, "model_dump")
                                else {}
                            ),
                        )
            except Exception:
                pass
            if st is None:
                agent_obj = _resolve_agent_entry(model.agent)
                # If we have a registry-backed callable and YAML provided params, wrap to inject them
                _params_for_callable: Dict[str, Any] = {}
                try:
                    if isinstance(model.agent, dict):
                        maybe_params = model.agent.get("params")
                        if isinstance(maybe_params, dict):
                            _params_for_callable = dict(maybe_params)
                except Exception:
                    _params_for_callable = {}

                def _with_params(func: Any) -> Any:
                    # Create an async wrapper that merges YAML params and respects step input when provided
                    import inspect as __inspect

                    async def _runner(data: Any, **kwargs: Any) -> Any:
                        try:
                            call_kwargs = dict(_params_for_callable)
                            call_kwargs.update(
                                {
                                    k: v
                                    for k, v in kwargs.items()
                                    if k not in ("context", "pipeline_context")
                                }
                            )
                            if model.input is not None:
                                result = func(data, **call_kwargs)
                            else:
                                result = func(**call_kwargs)
                            if __inspect.isawaitable(result):
                                return await result
                            return result
                        except TypeError:
                            # Fallback: try passing data as first arg
                            result = func(data, **dict(_params_for_callable))
                            if __inspect.isawaitable(result):
                                return await result
                            return result

                    return _runner

                callable_obj: Any = agent_obj
                try:
                    if callable(agent_obj) and _params_for_callable:
                        callable_obj = _with_params(agent_obj)
                except Exception:
                    callable_obj = agent_obj

                if _is_async_callable(callable_obj):
                    st = Step.from_callable(
                        callable_obj,
                        name=model.name,
                        updates_context=model.updates_context,
                        validate_fields=model.validate_fields,
                        **(step_config.model_dump() if hasattr(step_config, "model_dump") else {}),
                    )
        # If still no callable-based step, create a plain Step with the agent_obj
        if st is None:
            st = Step[Any, Any](
                name=model.name,
                agent=agent_obj,
                config=step_config,
                updates_context=model.updates_context,
                validate_fields=model.validate_fields,
            )
        # Attach templated input if provided in blueprint
        try:
            if model.input is not None:
                st.meta["templated_input"] = model.input
        except Exception:
            pass
        # Finalize static types
        _finalize_step_types(st)
        # Optional usage limits
        if model.usage_limits is not None:
            try:
                st.usage_limits = UsageLimits(**model.usage_limits)
            except Exception:
                pass
        # Optional plugins
        for plugin, priority in _resolve_plugins(model.plugins or []):
            try:
                st.plugins.append((plugin, priority))
            except Exception:
                pass
        # Optional validators
        for validator in _resolve_validators(model.validators or []):
            try:
                st.validators.append(validator)
            except Exception:
                pass
        # Optional fallback
        if model.fallback is not None:
            try:
                # Fallback can be custom; accept dict directly to allow registry dispatch
                st.fallback_step = _make_step_from_blueprint(
                    model.fallback
                    if isinstance(model.fallback, dict)
                    else BlueprintStepModel.model_validate(model.fallback),
                    yaml_path=f"{yaml_path}.fallback" if yaml_path else None,
                )
            except Exception:
                pass
        # Attach yaml_path for telemetry
        if yaml_path:
            try:
                st.meta["yaml_path"] = yaml_path
            except Exception:
                pass
        return st


def _build_pipeline_from_branch(
    branch_spec: Any,
    *,
    base_path: Optional[str] = None,
    compiled_agents: Optional[Dict[str, Any]] = None,
    compiled_imports: Optional[Dict[str, Any]] = None,
) -> Pipeline[Any, Any]:
    # Accept either a list[BlueprintStepModel-like dicts] or a single dict
    if isinstance(branch_spec, list):
        steps: List[Step[Any, Any]] = []
        for idx, s in enumerate(branch_spec):
            steps.append(
                _make_step_from_blueprint(
                    s,
                    yaml_path=f"{base_path}.steps[{idx}]" if base_path is not None else None,
                    compiled_agents=compiled_agents,
                    compiled_imports=compiled_imports,
                )
            )
        return Pipeline.model_construct(steps=steps)
    elif isinstance(branch_spec, dict):
        return Pipeline.from_step(
            _make_step_from_blueprint(
                branch_spec,
                yaml_path=f"{base_path}.steps[0]" if base_path is not None else None,
                compiled_agents=compiled_agents,
                compiled_imports=compiled_imports,
            )
        )
    else:
        raise BlueprintError("Invalid branch specification; expected dict or list of dicts")


def build_pipeline_from_blueprint(
    model: BlueprintPipelineModel,
    compiled_agents: Optional[Dict[str, Any]] = None,
    compiled_imports: Optional[Dict[str, Any]] = None,
) -> Pipeline[Any, Any]:
    steps: List[Step[Any, Any]] = []
    for idx, s in enumerate(model.steps):
        steps.append(
            _make_step_from_blueprint(
                s,
                yaml_path=f"steps[{idx}]",
                compiled_agents=compiled_agents,
                compiled_imports=compiled_imports,
            )
        )
    p = Pipeline.model_construct(steps=steps)
    # Best-effort finalize types after Pipeline construction
    try:
        for st in p.steps:
            _finalize_step_types(st)
    except Exception:
        pass
    return p


def dump_pipeline_blueprint_to_yaml(pipeline: Pipeline[Any, Any]) -> str:
    """Serialize a Pipeline to a minimal YAML blueprint (v0)."""

    def step_to_yaml(step: Any) -> Dict[str, Any]:
        if isinstance(step, ParallelStep):
            branches: Dict[str, Any] = {}
            for k, p in step.branches.items():
                branches[str(k)] = [step_to_yaml(s) for s in p.steps]
            return {
                "kind": "parallel",
                "name": step.name,
                "branches": branches,
                "merge_strategy": getattr(step.merge_strategy, "name", None),
            }
        try:
            from ..dsl.loop import MapStep

            if isinstance(step, MapStep):
                body = getattr(step, "original_body_pipeline", None) or getattr(
                    step, "pipeline_to_run", None
                )
                body_steps: List[Dict[str, Any]] = []
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
                data: Dict[str, Any] = {
                    "kind": "conditional",
                    "name": step.name,
                    "branches": branches,
                }
                if step.default_branch_pipeline is not None:
                    data["default_branch"] = [
                        step_to_yaml(s) for s in step.default_branch_pipeline.steps
                    ]
                return data
        except Exception:
            pass
        try:
            from ..dsl.loop import LoopStep

            if isinstance(step, LoopStep):
                loop_data = {
                    "body": [step_to_yaml(s) for s in step.loop_body_pipeline.steps],
                    "max_loops": step.max_retries,
                }

                # Add mapper fields if they exist
                if (
                    hasattr(step, "initial_input_to_loop_body_mapper")
                    and step.initial_input_to_loop_body_mapper
                ):
                    # For now, we can't easily serialize the callable, so we'll skip it
                    # In a future enhancement, we could store the original import string
                    pass
                if hasattr(step, "iteration_input_mapper") and step.iteration_input_mapper:
                    # For now, we can't easily serialize the callable, so we'll skip it
                    pass
                if hasattr(step, "loop_output_mapper") and step.loop_output_mapper:
                    # For now, we can't easily serialize the callable, so we'll skip it
                    pass

                return {
                    "kind": "loop",
                    "name": step.name,
                    "loop": loop_data,
                }
        except Exception:
            pass
        try:
            # Pretty-print HumanInTheLoopStep as a first-class 'hitl' kind
            from ..dsl.step import HumanInTheLoopStep

            if isinstance(step, HumanInTheLoopStep):
                hitl_data: Dict[str, Any] = {
                    "kind": "hitl",
                    "name": step.name,
                }
                # Optional message_for_user
                try:
                    if getattr(step, "message_for_user", None):
                        hitl_data["message"] = getattr(step, "message_for_user")
                except Exception:
                    pass
                # Optional input_schema (pydantic model class or dict)
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
                return hitl_data
        except Exception:
            pass
        try:
            # Pretty-print CacheStep as a first-class 'cache' kind
            from flujo.steps.cache_step import CacheStep as _CacheStep

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

    data: Dict[str, Any] = {
        "version": "0.1",
        "steps": [step_to_yaml(s) for s in pipeline.steps],
    }
    return yaml.safe_dump(data, sort_keys=False)


def load_pipeline_blueprint_from_yaml(
    yaml_text: str, base_dir: Optional[str] = None
) -> Pipeline[Any, Any]:
    # Proactively auto-load skills to honor docs: load skills.yaml before parsing.
    # This ensures CLI and any programmatic use benefit from the same behavior.
    try:
        if base_dir:
            from ...infra.skills_catalog import (
                load_skills_catalog as _load_skills_catalog,
                load_skills_entry_points as _load_skills_entry_points,
            )

            _load_skills_catalog(base_dir)
            _load_skills_entry_points()
    except Exception:
        # Never fail blueprint loading due to skills discovery issues
        pass
    try:
        data = yaml.safe_load(yaml_text)
        if not isinstance(data, dict) or "steps" not in data:
            raise BlueprintError("YAML blueprint must be a mapping with a 'steps' key")
        bp = BlueprintPipelineModel.model_validate(data)
        # If declarative agents or imports are present, compile them first
        from .compiler import DeclarativeBlueprintCompiler  # lazy import

        if bp.agents or getattr(bp, "imports", None):
            try:
                compiler = DeclarativeBlueprintCompiler(bp, base_dir=base_dir)
                return compiler.compile_to_pipeline()
            except Exception as e:
                # Surface a clear error instead of silently falling back and failing later
                raise BlueprintError(
                    f"Failed to compile declarative blueprint (agents/imports): {e}"
                ) from e
        return build_pipeline_from_blueprint(bp)
    except ValidationError as ve:
        # Construct readable error with locations
        try:
            errs = ve.errors()
            messages = [
                f"{e.get('msg')} at {'.'.join(str(p) for p in e.get('loc', []))}" for e in errs
            ]
            raise BlueprintError("; ".join(messages)) from ve
        except Exception:
            raise BlueprintError(str(ve)) from ve
    except yaml.YAMLError as ye:
        # Surface line/column details when available
        mark = getattr(ye, "problem_mark", None)
        if mark is not None:
            msg = f"Invalid YAML at line {getattr(mark, 'line', -1) + 1}, column {getattr(mark, 'column', -1) + 1}: {getattr(ye, 'problem', ye)}"
        else:
            msg = f"Invalid YAML: {ye}"
        raise BlueprintError(msg) from ye


def _resolve_plugins(specs: List[Union[str, Dict[str, Any]]]) -> List[Tuple[Any, int]]:
    result: List[Tuple[Any, int]] = []
    for item in specs:
        try:
            if isinstance(item, str):
                obj = _import_object(item)
                result.append((obj, 0))
            elif isinstance(item, dict):
                path = item.get("path")
                prio = int(item.get("priority", 0))
                if path:
                    obj = _import_object(path)
                    result.append((obj, prio))
        except Exception:
            continue
    return result


def _resolve_validators(specs: List[str]) -> List[Any]:
    result: List[Any] = []
    for path in specs:
        try:
            result.append(_import_object(path))
        except Exception:
            continue
    return result


# ----------------------------
# Resolution helpers (v1)
# ----------------------------


def _import_object(path: str) -> Any:
    """Import an object from 'module:attr' or 'module.attr' path with allow-list enforcement.

    Security hardening:
    - Reject path traversal and illegal characters (.., /, \\ or leading '.')
    - Enforce allow-list from configuration
    """
    import importlib
    import re
    from ...infra.config_manager import get_config_manager

    module_name: str
    attr_name: Optional[str] = None
    # Basic sanitization: disallow path traversal or filesystem-style separators
    if ".." in path or "/" in path or "\\" in path or path.strip().startswith("."):
        raise BlueprintError("Invalid import path: traversal or illegal characters are not allowed")

    # Only allow Python identifier characters and dots/colon separator
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\.]*(:[A-Za-z_][A-Za-z0-9_]*)?", path):
        raise BlueprintError("Invalid import path format")

    if ":" in path:
        module_name, attr_name = path.split(":", 1)
    else:
        parts = path.split(".")
        if len(parts) < 2:
            module_name = path
            attr_name = None
        else:
            module_name, attr_name = ".".join(parts[:-1]), parts[-1]

    # Enforce allow-list from flujo.toml: [settings] blueprint_allowed_imports = ["pkg", "pkg.sub"]
    try:
        cfg = get_config_manager().load_config()
        allowed: Optional[list[str]] = None
        if cfg and getattr(cfg, "settings", None) and isinstance(cfg.settings, object):
            # Accept both nested settings entry or top-level key 'blueprint_allowed_imports' in TOML
            allowed = getattr(cfg.settings, "blueprint_allowed_imports", None)
        if allowed is None:
            # Fallback: look for top-level attribute if provided
            allowed = getattr(cfg, "blueprint_allowed_imports", None)
        # Normalize to list of strings
        if allowed is not None and not isinstance(allowed, list):
            allowed = None
        if allowed is not None:
            # Permit if module_name matches or is a submodule of any allowed entry
            def _is_allowed(target: str) -> bool:
                return any(target == a or target.startswith(a + ".") for a in allowed or [])

            if not _is_allowed(module_name):
                raise BlueprintError(
                    f"Import of module '{module_name}' is not allowed. Configure 'blueprint_allowed_imports' in flujo.toml."
                )
    except Exception:
        # On configuration access failure, default to deny-by-default for safety
        raise BlueprintError(
            "Failed to verify allowed imports from configuration; refusing to import modules from YAML."
        )

    module = importlib.import_module(module_name)
    return getattr(module, attr_name) if attr_name else module


def _is_async_callable(obj: Any) -> bool:
    try:
        import inspect

        return inspect.iscoroutinefunction(obj)
    except Exception:
        return False


class _PassthroughAgent:
    async def run(self, x: Any, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - trivial
        return x


def _resolve_agent(agent_spec: str) -> Any:
    obj = _import_object(agent_spec)
    # If it's a class, try to instantiate with no args
    try:
        import inspect

        if inspect.isclass(obj):
            return obj()
        return obj
    except Exception:
        return obj


def _resolve_agent_entry(agent: Union[str, Dict[str, Any]]) -> Any:
    if isinstance(agent, str):
        return _resolve_agent(agent)
    if isinstance(agent, dict):
        skill_id = agent.get("id")
        params = agent.get("params", {})
        if skill_id:
            reg = get_skill_registry()
            entry = reg.get(skill_id)
            if entry is None:
                # Fallback: try to import skill_id as module path or attribute
                try:
                    obj = _import_object(skill_id)
                    if callable(obj):
                        return obj(**params)
                    return obj
                except Exception:
                    raise BlueprintError(f"Unknown skill id: {skill_id}")
            factory = entry.get("factory")
            try:
                if callable(factory):
                    return factory(**params)
                return factory
            except TypeError as e:
                raise BlueprintError(f"Failed to instantiate skill '{skill_id}': {e}") from e
        # Fallback to import string in dict under 'path'
        path = agent.get("path")
        if path:
            return _resolve_agent(path)
    raise BlueprintError("Invalid agent specification")
