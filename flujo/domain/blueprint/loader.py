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


class _CoercionConfig(BaseModel):
    tolerant_level: int = 0
    max_unescape_depth: int | None = None
    anyof_strategy: str | None = None
    allow: Dict[str, List[str]] | None = None

    @model_validator(mode="after")
    def _validate_values(self) -> "_CoercionConfig":
        if self.tolerant_level not in (0, 1, 2):
            raise ValueError("coercion.tolerant_level must be 0, 1 or 2")
        if self.max_unescape_depth is not None and self.max_unescape_depth < 0:
            raise ValueError("coercion.max_unescape_depth must be >= 0")
        if self.anyof_strategy is not None and str(self.anyof_strategy).lower() not in {
            "first-pass"
        }:
            raise ValueError("coercion.anyof_strategy must be 'first-pass' if set")
        if self.allow:
            valid = {
                "integer": {"str->int"},
                "number": {"str->float"},
                "boolean": {"str->bool"},
                "array": {"str->array"},
            }
            for k, v in self.allow.items():
                if k not in valid:
                    raise ValueError(f"coercion.allow has invalid key '{k}'")
                for t in v:
                    if t not in valid[k]:
                        raise ValueError(f"coercion.allow[{k}] has invalid transform '{t}'")
        return self


class _ReasoningPrecheckConfig(BaseModel):
    enabled: bool = False
    validator_agent: Any | None = None
    agent: Any | None = None
    delimiters: List[str] | None = None
    goal_context_key: str | None = None
    score_threshold: float | None = None
    required_context_keys: List[str] | None = None
    inject_feedback: str | None = None
    retry_guidance_prefix: str | None = None
    context_feedback_key: str | None = None
    consensus_agent: Any | None = None
    consensus_samples: int | None = None
    consensus_threshold: float | None = None

    @model_validator(mode="after")
    def _validate_values(self) -> "_ReasoningPrecheckConfig":
        if self.delimiters is not None and not (
            isinstance(self.delimiters, list) and len(self.delimiters) >= 2
        ):
            raise ValueError("reasoning_precheck.delimiters must be a list of at least 2 strings")
        if self.inject_feedback is not None and str(self.inject_feedback).lower() not in {
            "prepend",
            "context_key",
        }:
            raise ValueError(
                "reasoning_precheck.inject_feedback must be 'prepend' or 'context_key'"
            )
        if self.consensus_samples is not None and self.consensus_samples < 2:
            raise ValueError("reasoning_precheck.consensus_samples must be >= 2")
        return self


class ProcessingConfigModel(BaseModel):
    structured_output: str | None = None
    aop: str | None = None
    coercion: _CoercionConfig | None = None
    output_schema: Dict[str, Any] | None = Field(default=None, alias="schema")
    enforce_grammar: bool | None = None
    reasoning_precheck: _ReasoningPrecheckConfig | None = None

    model_config = {
        "populate_by_name": True,
        # Avoid pydantic BaseModel attribute collision warnings for 'schema' alias
        "protected_namespaces": (),
    }

    @model_validator(mode="after")
    def _normalize(self) -> "ProcessingConfigModel":
        if self.structured_output is not None:
            val = str(self.structured_output).lower()
            if val not in {"off", "auto", "openai_json", "outlines", "xgrammar"}:
                raise ValueError(
                    "processing.structured_output must be one of off|auto|openai_json|outlines|xgrammar"
                )
            self.structured_output = val
        if self.aop is not None:
            val = str(self.aop).lower()
            if val not in {"off", "minimal", "full"}:
                raise ValueError("processing.aop must be one of off|minimal|full")
            self.aop = val
        return self


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
        "agentic_loop",
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
    # Parallel reduce sugar
    reduce: Optional[Union[str, Dict[str, Any]]] = None
    # Conditional only (v0: simple string identifier for callable resolution)
    condition: Optional[str] = None
    # NEW: Expression-based conditional (mutually exclusive with 'condition')
    condition_expression: Optional[str] = None
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
    sink_to: Optional[str] = None  # Context path to sink HITL response
    # Cache specific (optional)
    wrapped_step: Optional[Dict[str, Any]] = None
    # Agentic loop sugar (M5)
    planner: Optional[str] = None  # import path or agents.<name>
    registry: Optional[Union[str, Dict[str, Any]]] = None  # import path to dict or inline map
    output_template: Optional[str] = None
    # AROS/processing options (stored under step.meta["processing"])
    processing: Optional[Dict[str, Any]] = None

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
    # Optional human-friendly pipeline name (top-level YAML: `name:`)
    name: Optional[str] = None
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
        # Enhancement: use agent-declared target_output_type when available (e.g., AsyncAgentWrapper)
        try:
            if _is_default_type(getattr(step_obj, "__step_output_type__", object)) and hasattr(
                agent_obj, "target_output_type"
            ):
                out_t = getattr(agent_obj, "target_output_type")
                if out_t is not None:
                    step_obj.__step_output_type__ = out_t
        except Exception:
            pass
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
        # Best-effort static typing; ignore errors during signature analysis
        pass


# ----------------------------
# Module-level helpers (deduplicated)
# ----------------------------


def _resolve_context_target(ctx: Any, target: str) -> tuple[Any, Any]:
    """Resolve a context.* target path to (parent, key).

    - Supports dict and attribute-style traversal.
    - Creates intermediate dicts/attributes when missing.
    - Returns (None, None) when the target is invalid.
    """
    try:
        if not isinstance(target, str) or not target.startswith("context."):
            return None, None
        parts = target.split(".")[1:]
        cur = ctx
        parent = None
        key = None
        for p in parts:
            parent = cur
            key = p
            try:
                if isinstance(cur, dict):
                    if p not in cur:
                        try:
                            cur[p] = {}
                        except Exception:
                            pass
                    cur = cur.get(p)
                    continue
            except Exception:
                pass
            nxt = getattr(cur, p, None)
            if nxt is None:
                try:
                    setattr(cur, p, {})
                    nxt = getattr(cur, p, None)
                except Exception:
                    nxt = None
            cur = nxt
        return parent, key
    except Exception:
        return None, None


def _render_template_value(prev_output: Any, ctx: Any, tpl: Any) -> Any:
    """Render a template string using context and previous output.

    Exposes variables: {context, previous_step, steps, resume_input}
    """
    try:
        from ...utils.template_vars import (
            TemplateContextProxy as _TCP,
            get_steps_map_from_context as _get_steps,
            StepValueProxy as _SVP,
        )
        from ...utils.prompting import AdvancedPromptFormatter as _Fmt

        steps_map = _get_steps(ctx)
        steps_wrapped = {k: v if isinstance(v, _SVP) else _SVP(v) for k, v in steps_map.items()}
        fmt_ctx = {
            "context": _TCP(ctx, steps=steps_wrapped),
            "previous_step": prev_output,
            "steps": steps_wrapped,
        }

        # Add resume_input if HITL history exists
        try:
            if ctx and hasattr(ctx, "hitl_history") and ctx.hitl_history:
                fmt_ctx["resume_input"] = ctx.hitl_history[-1].human_response
        except Exception:
            pass  # resume_input will be undefined if no HITL history

        return _Fmt(str(tpl)).format(**fmt_ctx)
    except Exception:
        # Best effort: return raw string representation
        try:
            return str(tpl)
        except Exception:
            return tpl


def _make_step_from_blueprint(
    model: Any,
    *,
    yaml_path: Optional[str] = None,
    compiled_agents: Optional[Dict[str, Any]] = None,
    compiled_imports: Optional[Dict[str, Any]] = None,
) -> Step[Any, Any]:
    # Support both native BlueprintStepModel and raw dict for custom primitives
    if isinstance(model, dict):
        # Preserve custom per-step flags before validation (e.g., use_history)
        _raw_use_history = None
        try:
            if "use_history" in model:
                _raw_use_history = bool(model.get("use_history"))
        except Exception:
            _raw_use_history = None
        kind_val = str(model.get("kind", "step"))
        # Pre-normalize boolean branch keys for conditional steps (FSD-026)
        if kind_val == "conditional":
            try:
                branches_raw = model.get("branches")
                if isinstance(branches_raw, dict):
                    coerced: Dict[str, Any] = {}
                    for _k, _v in branches_raw.items():
                        if isinstance(_k, bool):
                            coerced[str(_k).lower()] = _v
                        else:
                            coerced[str(_k)] = _v if _k not in coerced else _v
                    model = dict(model)
                    model["branches"] = coerced
            except Exception:
                # Best-effort: keep original if anything goes wrong
                pass
        # Built-in kinds handled by existing logic via typed model
        # First-class handling for StateMachine (bypasses typed BlueprintStepModel)
        if kind_val == "StateMachine":
            # Build states → Pipeline map with compiled imports/agents available
            from ..dsl.state_machine import StateMachineStep as _StateMachineStep

            name = str(model.get("name", "StateMachine"))
            start_state = str(model.get("start_state"))
            end_states_val = model.get("end_states") or []
            if not isinstance(end_states_val, list):
                end_states_val = [end_states_val]
            end_states = [str(x) for x in end_states_val]

            states_raw = model.get("states") or {}
            if not isinstance(states_raw, dict):
                raise BlueprintError("StateMachine.states must be a mapping of state → steps")

            coerced_states: Dict[str, Pipeline[Any, Any]] = {}
            for _state_name, _branch_spec in states_raw.items():
                coerced_states[str(_state_name)] = _build_pipeline_from_branch(
                    _branch_spec,
                    base_path=f"{yaml_path}.states.{_state_name}" if yaml_path else None,
                    compiled_agents=compiled_agents,
                    compiled_imports=compiled_imports,
                )

            # Transitions: pass raw list of dicts to be validated by the model
            transitions_raw = model.get("transitions")
            if transitions_raw is not None and not isinstance(transitions_raw, list):
                raise BlueprintError("StateMachine.transitions must be a list of rules")
            # Normalize YAML 1.1 boolean key coercion for 'on' (PyYAML may parse 'on' as True)
            coerced_transitions = []
            if isinstance(transitions_raw, list):
                for _idx, _rule in enumerate(transitions_raw):
                    if isinstance(_rule, dict):
                        _coerced: Dict[str, Any] = {}
                        for _k, _v in _rule.items():
                            if _k is True:  # YAML 'on' key parsed as boolean True
                                _coerced["on"] = _v
                            elif _k is False:  # defensive symmetry (not expected here)
                                _coerced["off"] = _v  # unlikely used; preserve intent
                            else:
                                _coerced[str(_k)] = _v
                        coerced_transitions.append(_coerced)
                    else:
                        coerced_transitions.append(_rule)

            sm = _StateMachineStep(
                name=name,
                states=coerced_states,
                start_state=start_state,
                end_states=end_states,
                transitions=coerced_transitions or [],
            )
            # Attach yaml_path for telemetry if available
            if yaml_path:
                try:
                    sm.meta["yaml_path"] = yaml_path
                except Exception:
                    pass
            return sm

        if kind_val in {
            "step",
            "parallel",
            "conditional",
            "loop",
            "map",
            "dynamic_router",
            "hitl",
            "cache",
            "agentic_loop",
        }:
            model = BlueprintStepModel.model_validate(model)
            # Validate processing early to surface errors before step assembly
            try:
                proc_raw = getattr(model, "processing", None)
                if isinstance(proc_raw, dict) and proc_raw:
                    pc = ProcessingConfigModel.model_validate(proc_raw)
                    # replace with normalized dict using aliases (keeps 'schema' key)
                    try:
                        setattr(
                            model, "processing", pc.model_dump(exclude_none=True, by_alias=True)
                        )
                    except Exception:
                        pass
            except Exception as e:
                raise BlueprintError(f"Invalid processing configuration: {e}") from e
            # Attach preserved extras to the validated model for later consumption via meta
            try:
                if _raw_use_history is not None:
                    setattr(model, "_use_history_extra", _raw_use_history)
            except Exception:
                pass
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
        st_par: ParallelStep[Any] = ParallelStep(
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
        # Declarative reduce sugar for parallel outputs
        try:
            reduce_spec = model.reduce
        except Exception:
            reduce_spec = None
        if isinstance(reduce_spec, (str, dict)) and reduce_spec:
            try:
                branch_order = list(branches_map.keys())

                mode: str
                if isinstance(reduce_spec, str):
                    mode = reduce_spec.strip().lower()
                else:
                    mode = str(reduce_spec.get("mode", "")).strip().lower()

                def _reduce(output_map: Dict[str, Any], _ctx: Optional[Any]) -> Any:
                    if mode == "keys":
                        # Preserve declared branch order
                        return [bn for bn in branch_order if bn in output_map]
                    if mode == "values":
                        return [output_map[bn] for bn in branch_order if bn in output_map]
                    if mode == "union":
                        # Merge dict outputs with last-wins by branch order
                        acc: Dict[str, Any] = {}
                        for bn in branch_order:
                            val = output_map.get(bn)
                            if isinstance(val, dict):
                                acc.update(val)
                        return acc
                    if mode == "concat":
                        # Concatenate list outputs in branch order
                        res: list[Any] = []
                        for bn in branch_order:
                            val = output_map.get(bn)
                            if isinstance(val, list):
                                res.extend(val)
                            elif val is not None:
                                res.append(val)
                        return res
                    if mode == "first":
                        for bn in branch_order:
                            if bn in output_map:
                                return output_map[bn]
                        return None
                    if mode == "last":
                        for bn in reversed(branch_order):
                            if bn in output_map:
                                return output_map[bn]
                        return None
                    # Default: return original map
                    return output_map

                try:
                    st_par.meta["parallel_reduce_mapper"] = _reduce
                except Exception:
                    pass
            except Exception:
                pass
        return st_par
    elif getattr(model, "kind", None) == "conditional":
        from ..dsl.conditional import ConditionalStep

        if not model.branches:
            raise BlueprintError("conditional step requires branches")
        branches_map2: Dict[Any, Pipeline[Any, Any]] = {}
        for key, branch_spec in model.branches.items():
            branches_map2[key] = _build_pipeline_from_branch(
                branch_spec,
                base_path=f"{yaml_path}.branches.{key}" if yaml_path else None,
                compiled_agents=compiled_agents,
            )
        # Resolve condition callable explicitly to avoid scope issues
        if model.condition:
            try:
                _cond_callable = _import_object(model.condition)

                # Validate that condition is synchronous (not async)
                import asyncio

                if asyncio.iscoroutinefunction(_cond_callable):
                    raise BlueprintError(
                        f"condition '{model.condition}' must be synchronous.\n"
                        f"Conditional step conditions are called synchronously and cannot be async functions.\n"
                        f"\n"
                        f"Change your function from:\n"
                        f"  async def my_condition(data, context) -> Any:\n"
                        f"      ...\n"
                        f"\n"
                        f"To:\n"
                        f"  def my_condition(data, context) -> Any:\n"
                        f"      ...\n"
                        f"\n"
                        f"Remove 'async' and any 'await' calls in your condition function.\n"
                        f"See: https://flujo.dev/docs/user_guide/pipeline_branching#conditional-steps"
                    )
            except Exception as exc:
                # Improve ergonomics: common mistake is providing an inline Python lambda
                # which is intentionally not supported in YAML for security reasons.
                try:
                    _cond_str = str(model.condition).strip()
                except Exception:
                    _cond_str = ""
                # Match 'lambda ...' with optional leading '(' and whitespace
                if re.match(r"^\(?\s*lambda\b", _cond_str):
                    raise BlueprintError(
                        "Invalid condition value: inline Python (e.g., a lambda expression) is not supported in YAML. "
                        "Use 'condition_expression' for inline logic or reference an importable callable like 'pkg.mod:func'.\n"
                        'Example: condition_expression: "{{ previous_step }}"'
                    ) from exc
                # Otherwise, rewrap with clear field context and actionable guidance
                raise BlueprintError(
                    f"Failed to resolve condition '{_cond_str}' (field: condition). "
                    "Provide a Python import path like 'pkg.mod:func' or use 'condition_expression'. "
                    f"Underlying error: {exc}"
                ) from exc
        elif model.condition_expression:
            try:
                from ...utils.expressions import compile_expression_to_callable as _compile_expr

                _expr_fn = _compile_expr(str(model.condition_expression))

                def _cond_callable(output: Any, _ctx: Optional[Any]) -> Any:
                    return _expr_fn(output, _ctx)

            except Exception as e:
                raise BlueprintError(f"Invalid condition_expression: {e}") from e
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
        st_cond: ConditionalStep[Any] = ConditionalStep(
            name=model.name,
            condition_callable=_cond_callable,
            branches=branches_map2,
            default_branch_pipeline=default_branch,
            config=step_config,
        )
        try:
            if model.condition_expression:
                st_cond.meta["condition_expression"] = str(model.condition_expression)
        except Exception:
            pass
        return st_cond
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

            # Validate that exit_condition is synchronous (not async)
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
                    _output: Any, _ctx: Optional[Any], *, _state: Optional[Dict[str, int]] = None
                ) -> bool:
                    return bool(_expr_fn2(_output, _ctx))

            except Exception as e:
                raise BlueprintError(f"Invalid exit_expression: {e}") from e
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

        # Declarative loop state sugar (M4): compile simple mappers when 'loop.state' is present
        from typing import Callable as _Callable, Optional as _Optional

        _initial_mapper_override: _Optional[_Callable[[Any, Optional[Any]], Any]] = None
        _iteration_mapper_override: _Optional[_Callable[[Any, Optional[Any], int], Any]] = None
        _output_mapper_override: _Optional[_Callable[[Any, Optional[Any]], Any]] = None

        # Additional compiled helpers
        _state_apply_fn: _Optional[_Callable[[Any, Any], None]] = None
        _compiled_init_ops: _Optional[_Callable[[Any, Any], None]] = None
        _compiled_iter_prop: _Optional[_Callable[[Any, Optional[Any], int], Any]] = None

        try:
            state_spec = model.loop.get("state") if isinstance(model.loop, dict) else None
        except Exception:
            state_spec = None

        if isinstance(state_spec, dict) and any(
            k in state_spec for k in ("append", "set", "merge")
        ):
            try:
                import json as _json

                ops_append = state_spec.get("append") or []
                ops_set = state_spec.get("set") or []
                ops_merge = state_spec.get("merge") or []

                # Use module-level helpers for target resolution and template rendering
                _resolve_target = _resolve_context_target

                def _render_value(output: Any, ctx: Any, tpl: str) -> str:
                    return str(_render_template_value(output, ctx, tpl))

                def _apply_state_ops(output: Any, ctx: Any) -> None:
                    # append
                    for spec in ops_append:
                        try:
                            target = str(spec.get("target"))
                            parent, key = _resolve_target(ctx, target)
                            if parent is None or key is None:
                                continue
                            val = _render_value(output, ctx, spec.get("value", ""))
                            seq = None
                            try:
                                seq = parent[key]
                            except Exception:
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
                    # set
                    for spec in ops_set:
                        try:
                            target = str(spec.get("target"))
                            parent, key = _resolve_target(ctx, target)
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
                    # merge (value must render to mapping JSON)
                    for spec in ops_merge:
                        try:
                            target = str(spec.get("target"))
                            parent, key = _resolve_target(ctx, target)
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

                # Expose state-ops applier for possible composition with propagation
                _state_apply_fn = _apply_state_ops

                def _initial_mapper_override(input_data: Any, ctx: Optional[Any]) -> Any:
                    # Identity passthrough; state ops apply after each iteration
                    return input_data

                def _iteration_mapper_override(
                    output: Any, ctx: Optional[Any], _iteration: int
                ) -> Any:
                    if ctx is not None:
                        _apply_state_ops(output, ctx)
                    return output

                def _output_mapper_override(output: Any, ctx: Optional[Any]) -> Any:
                    return output

            except Exception:
                # If any failure occurs while compiling state sugar, ignore and fall back to defaults
                pass

        # --- Declarative 'init' block (runs once before first iteration) ---
        try:
            init_spec = model.loop.get("init") if isinstance(model.loop, dict) else None
        except Exception:
            init_spec = None
        if init_spec is not None:
            try:
                import json as _json2

                # Normalize init spec to ops lists
                ops_init_append: list[dict[str, Any]] = []
                ops_init_set: list[dict[str, Any]] = []
                ops_init_merge: list[dict[str, Any]] = []

                if isinstance(init_spec, dict):
                    # Support same shape as state sugar
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
                                # allow either {append: target, value: ...} or {append: {target, value}}
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
                                            "value": op.get("value") or "{}",
                                        }
                                    )
                                elif isinstance(_tmp_m, dict):
                                    d = _tmp_m
                                    ops_init_merge.append(
                                        {
                                            "target": d.get("target"),
                                            "value": d.get("value") or "{}",
                                        }
                                    )
                        except Exception:
                            continue
                else:
                    # Unsupported shape → ignore safely
                    ops_init_append, ops_init_set, ops_init_merge = [], [], []

                def _compiled_init(prev_output: Any, ctx: Any) -> None:
                    # append
                    for spec in ops_init_append:
                        try:
                            parent, key = _resolve_context_target(ctx, str(spec.get("target")))
                            if parent is None or key is None:
                                continue
                            val = _render_template_value(prev_output, ctx, spec.get("value", ""))
                            seq = None
                            try:
                                seq = parent[key]
                            except Exception:
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
                    # set
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
                    # merge
                    for spec in ops_init_merge:
                        try:
                            parent, key = _resolve_context_target(ctx, str(spec.get("target")))
                            if parent is None or key is None:
                                continue
                            val_raw = _render_template_value(
                                prev_output, ctx, spec.get("value", "{}")
                            )
                            try:
                                val_obj = _json2.loads(val_raw)
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

                _compiled_init_ops = _compiled_init
            except Exception:
                _compiled_init_ops = None

        # --- Declarative 'propagation.next_input' → iteration_input_mapper ---
        try:
            propagation_spec = (
                model.loop.get("propagation") if isinstance(model.loop, dict) else None
            )
        except Exception:
            propagation_spec = None
        try:
            next_input_spec = None
            if isinstance(propagation_spec, dict):
                next_input_spec = propagation_spec.get("next_input")
            elif isinstance(propagation_spec, str):
                # allow shorthand: propagation: context | previous_output | auto | <template>
                next_input_spec = propagation_spec
            if isinstance(next_input_spec, str) and next_input_spec.strip():
                import json as _json3
                from ...utils.template_vars import (
                    TemplateContextProxy as _TCP3,
                    get_steps_map_from_context as _get_steps3,
                    StepValueProxy as _SVP3,
                )
                from ...utils.prompting import AdvancedPromptFormatter as _Fmt3

                spec_raw = next_input_spec.strip().lower()
                # Resolve 'auto' preset based on loop body steps using updates_context
                if spec_raw == "auto":
                    has_updates = False
                    try:
                        if hasattr(body, "steps") and isinstance(getattr(body, "steps"), list):
                            for _st in getattr(body, "steps"):
                                if bool(getattr(_st, "updates_context", False)):
                                    has_updates = True
                                    break
                    except Exception:
                        has_updates = False
                    spec_str = "context" if has_updates else "previous_output"
                else:
                    spec_str = next_input_spec.strip()

                def _iter_prop(prev_output: Any, ctx: Optional[Any], _iteration: int) -> Any:
                    if spec_str.lower() == "context":
                        return ctx
                    if spec_str.lower() == "previous_output":
                        return prev_output
                    if ctx is None:
                        return prev_output
                    steps_map0 = _get_steps3(ctx)
                    steps_wrapped = {
                        k: v if isinstance(v, _SVP3) else _SVP3(v) for k, v in steps_map0.items()
                    }
                    fmt_ctx = {
                        "context": _TCP3(ctx, steps=steps_wrapped),
                        "previous_step": prev_output,
                        "steps": steps_wrapped,
                    }
                    rendered = _Fmt3(spec_str).format(**fmt_ctx)
                    # Attempt JSON decode for object/array
                    try:
                        if rendered and rendered.strip()[:1] in ("{", "["):
                            return _json3.loads(rendered)
                    except Exception:
                        pass
                    return rendered

                _compiled_iter_prop = _iter_prop
        except Exception:
            _compiled_iter_prop = None

        # --- Declarative 'output' or 'output_template' → loop_output_mapper ---
        _output_tpl_override: _Optional[_Callable[[Any, Optional[Any]], Any]] = None
        try:
            output_template_spec = None
            output_mapping_spec = None
            if isinstance(model.loop, dict):
                output_template_spec = model.loop.get("output_template")
                output_mapping_spec = model.loop.get("output")
            if isinstance(output_template_spec, str) and output_template_spec.strip():
                fmt_tpl = output_template_spec.strip()

                def _out_tpl(prev_output: Any, ctx: Optional[Any]) -> Any:
                    if ctx is None:
                        return prev_output
                    return _render_template_value(prev_output, ctx, fmt_tpl)

                _output_tpl_override = _out_tpl
            elif isinstance(output_mapping_spec, dict) and output_mapping_spec:
                mapping_items: list[tuple[str, str]] = [
                    (str(k), str(v)) for k, v in output_mapping_spec.items()
                ]

                def _out_map(prev_output: Any, ctx: Optional[Any]) -> Any:
                    if ctx is None:
                        return {k: None for k, _ in mapping_items}
                    out: Dict[str, Any] = {}
                    for mk, mtpl in mapping_items:
                        try:
                            out[mk] = _render_template_value(prev_output, ctx, mtpl)
                        except Exception:
                            out[mk] = None
                    return out

                _output_tpl_override = _out_map
        except Exception:
            _output_tpl_override = None

        # Compose iteration mapper precedence:
        # - If propagation is provided, use it; if state ops exist, apply them first
        if _compiled_iter_prop is not None:
            if _state_apply_fn is not None:

                def _iter_composed(_o: Any, _c: Optional[Any], _i: int) -> Any:
                    try:
                        if _c is not None:
                            _state_apply_fn(_o, _c)
                    except Exception:
                        pass
                    return _compiled_iter_prop(_o, _c, _i)

                _iteration_mapper_override = _iter_composed
            else:
                _iteration_mapper_override = _compiled_iter_prop
        else:
            # No propagation override; if state ops exist, keep default behavior (apply ops, pass through)
            if _state_apply_fn is not None and _iteration_mapper_override is None:

                def _iter_state_only(_o: Any, _c: Optional[Any], _i: int) -> Any:
                    try:
                        if _c is not None:
                            _state_apply_fn(_o, _c)
                    except Exception:
                        pass
                    return _o

                _iteration_mapper_override = _iter_state_only

        # Output precedence: explicit output/_template wins over state sugar default
        if _output_tpl_override is not None:
            _output_mapper_override = _output_tpl_override

        st_loop: LoopStep[Any] = LoopStep(
            name=model.name,
            loop_body_pipeline=body,
            exit_condition_callable=_exit_condition,
            max_retries=max(1, int(max_loops)) if isinstance(max_loops, int) else 1,
            config=step_config,
            # --- PASS THE RESOLVED MAPPERS ---
            initial_input_to_loop_body_mapper=(
                _initial_mapper_override
                if _initial_mapper_override is not None
                else _initial_mapper
            ),
            iteration_input_mapper=(
                _iteration_mapper_override
                if _iteration_mapper_override is not None
                else _iter_mapper
            ),
            loop_output_mapper=(
                _output_mapper_override if _output_mapper_override is not None else _output_mapper
            ),
        )
        # Attach compiled init ops for the policy hook to run once before iteration 1
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
        # Conversation-related options (FSD-033). Store in meta for policy consumption.
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
            # Best-effort: ignore malformed fields
            pass
        return st_loop
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
        st_map: MapStep[Any] = MapStep.from_pipeline(
            name=model.name, pipeline=body, iterable_input=str(iterable_input)
        )

        # MapStep declarative 'init' sugar (runs once before mapping begins)
        try:
            map_init = model.map.get("init") if isinstance(model.map, dict) else None
        except Exception:
            map_init = None
        if map_init is not None:
            try:
                import json as _jsonm

                m_ops_append: list[dict[str, Any]] = []
                m_ops_set: list[dict[str, Any]] = []
                m_ops_merge: list[dict[str, Any]] = []
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
                                m_ops_set.append(
                                    {"target": op.get("set"), "value": op.get("value")}
                                )
                            elif "append" in op:
                                _spec_a = op.get("append")
                                if isinstance(_spec_a, str):
                                    m_ops_append.append(
                                        {"target": _spec_a, "value": op.get("value")}
                                    )
                                elif isinstance(_spec_a, dict):
                                    m_ops_append.append(
                                        {
                                            "target": _spec_a.get("target"),
                                            "value": _spec_a.get("value"),
                                        }
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

                def _map_init_ops(prev_output: Any, ctx: Any) -> None:
                    # append
                    for spec in m_ops_append:
                        try:
                            parent, key = _resolve_context_target(ctx, str(spec.get("target")))
                            if parent is None or key is None:
                                continue
                            val = _render_template_value(prev_output, ctx, spec.get("value", ""))
                            seq = None
                            try:
                                seq = parent[key]
                            except Exception:
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
                    # set
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
                    # merge
                    for spec in m_ops_merge:
                        try:
                            parent, key = _resolve_context_target(ctx, str(spec.get("target")))
                            if parent is None or key is None:
                                continue
                            val_raw = _render_template_value(
                                prev_output, ctx, spec.get("value", "{}")
                            )
                            try:
                                val_obj = _jsonm.loads(val_raw)
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

        # MapStep declarative 'finalize' sugar (post-aggregation output mapping)
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

                    def _finalize_mapper(prev_output: Any, ctx: Optional[Any]) -> Any:
                        if ctx is None:
                            return prev_output
                        return _render_template_value(prev_output, ctx, tpl)

                    finalize_mapper = _finalize_mapper
                elif isinstance(output_mapping_spec, dict) and output_mapping_spec:
                    items = [(str(k), str(v)) for k, v in output_mapping_spec.items()]

                    def _finalize_map(prev_output: Any, ctx: Optional[Any]) -> Any:
                        if ctx is None:
                            return {k: None for k, _ in items}
                        out: Dict[str, Any] = {}
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
            sink_to=model.sink_to,
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
    elif getattr(model, "kind", None) == "agentic_loop":
        # YAML sugar to create an agentic loop using the recipes factory
        try:
            from ...recipes.factories import make_agentic_loop_pipeline as _make_agentic
        except Exception as e:
            raise BlueprintError(f"Agentic loop factory is unavailable: {e}")

        # Resolve planner agent
        if not model.planner:
            raise BlueprintError("agentic_loop requires 'planner'")
        planner_agent = _resolve_agent_entry(model.planner)

        # Resolve registry
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

        # Optionally apply an output template via loop_output_mapper wrapper
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

        # Wrap as a single step for YAML
        try:
            return p.as_step(name=model.name)
        except Exception as e:
            raise BlueprintError(f"Failed to wrap agentic loop pipeline as step: {e}")
    else:
        # Simple step; resolve agent if provided, otherwise passthrough.
        # Recover preserved extras if present on the validated model
        _use_history_extra = None
        try:
            _use_history_extra = getattr(model, "_use_history_extra", None)
        except Exception:
            _use_history_extra = None
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
                        sink_to=model.sink_to,
                        **(step_config.model_dump() if hasattr(step_config, "model_dump") else {}),
                    )
            elif uses_spec.startswith("imports."):
                # Wrap precompiled sub-pipeline as a first-class ImportStep for policy-driven execution
                if compiled_imports is None:
                    raise BlueprintError(
                        f"No compiled imports available but step uses '{uses_spec}'"
                    )
                alias = uses_spec.split(".", 1)[1]
                if alias not in compiled_imports:
                    raise BlueprintError(f"Unknown imported pipeline referenced: {uses_spec}")
                try:
                    from ..dsl.import_step import (
                        ImportStep as _ImportStep,
                        OutputMapping as _OutputMapping,
                    )

                    _sub_pipeline = compiled_imports[alias]
                    # Extract import-specific config knobs if provided
                    raw_cfg: Dict[str, Any] = {}
                    try:
                        raw_cfg = dict(getattr(model, "config", {}) or {})
                    except Exception:
                        raw_cfg = {}
                    input_to = str(raw_cfg.get("input_to", "initial_prompt")).strip().lower()
                    if input_to not in {"initial_prompt", "scratchpad", "both"}:
                        input_to = "initial_prompt"
                    # Defaults per first principles: do not inherit context by default; do inherit conversation
                    inherit_context = bool(raw_cfg.get("inherit_context", False))
                    input_scratchpad_key = raw_cfg.get("input_scratchpad_key", "initial_input")
                    # Accept either a dict mapping (backward compat) or a list of {child, parent}
                    outputs_spec = raw_cfg.get("outputs", None)
                    outputs_list: Optional[list[_OutputMapping]] = None
                    if isinstance(outputs_spec, dict):
                        outputs_list = []
                        # Convert mapping child_path -> parent_path into list of OutputMapping
                        for c_path, p_path in outputs_spec.items():
                            try:
                                outputs_list.append(
                                    _OutputMapping(child=str(c_path), parent=str(p_path))
                                )
                            except Exception:
                                continue
                    elif isinstance(outputs_spec, list):
                        outputs_list = []
                        for item in outputs_spec:
                            try:
                                if isinstance(item, dict) and "child" in item and "parent" in item:
                                    outputs_list.append(
                                        _OutputMapping(
                                            child=str(item["child"]), parent=str(item["parent"])
                                        )
                                    )
                            except Exception:
                                continue
                    inherit_conversation = bool(raw_cfg.get("inherit_conversation", True))
                    on_failure = str(raw_cfg.get("on_failure", "abort")).strip().lower()
                    if on_failure not in {"abort", "skip", "continue_with_default"}:
                        on_failure = "abort"
                    # Optional HITL propagation flag (defaults True)
                    propagate_hitl = bool(raw_cfg.get("propagate_hitl", True))

                    from typing import cast as _cast

                    st = _ImportStep(
                        name=model.name,
                        pipeline=_sub_pipeline,
                        inherit_context=inherit_context,
                        input_to=_cast(Literal["initial_prompt", "scratchpad", "both"], input_to),
                        input_scratchpad_key=input_scratchpad_key
                        if isinstance(input_scratchpad_key, str)
                        else "initial_input",
                        outputs=outputs_list,
                        inherit_conversation=inherit_conversation,
                        propagate_hitl=propagate_hitl,
                        on_failure=_cast(
                            Literal["abort", "skip", "continue_with_default"], on_failure
                        ),
                        # Preserve updates_context flag for merge behavior
                        updates_context=model.updates_context,
                        # Pass through generic step config for consistency
                        config=step_config,
                    )
                    # Record the import alias for validation/reporting purposes
                    meta = getattr(st, "meta", None)
                    if isinstance(meta, dict):
                        meta["import_alias"] = alias
                except Exception as e:
                    raise BlueprintError(
                        f"Failed to wrap imported pipeline '{alias}' as ImportStep: {e}"
                    )
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
                            sink_to=model.sink_to,
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
                            sink_to=model.sink_to,
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

                    # Check if this is a builtin skill by looking at the original skill_id from YAML
                    is_builtin = False
                    skill_id = None
                    try:
                        if isinstance(model.agent, dict):
                            skill_id = model.agent.get("id", "")
                            is_builtin = isinstance(skill_id, str) and skill_id.startswith(
                                "flujo.builtins."
                            )
                    except (AttributeError, KeyError, TypeError):
                        is_builtin = False

                    async def _runner(data: Any, **kwargs: Any) -> Any:
                        try:
                            call_kwargs = dict(_params_for_callable)
                            # For builtin skills, handle parameter passing specially
                            if is_builtin:
                                # For builtin skills with explicit params, unpack them as kwargs
                                # For builtins without params, pass data positionally
                                # Always try to inject context from runtime if available (for skills like context_merge)
                                non_context_params = {
                                    k: v for k, v in call_kwargs.items() if k != "context"
                                }

                                if non_context_params:
                                    # Has explicit params from agent.params in YAML
                                    # Add runtime context if available (builtins like context_merge need it)
                                    if "context" in kwargs and "context" not in call_kwargs:
                                        call_kwargs["context"] = kwargs["context"]
                                    result = func(**call_kwargs)
                                else:
                                    # No explicit params - pass data positionally only
                                    # Don't inject context (stringify doesn't accept it)
                                    result = func(data)
                            else:
                                # For non-builtin, allow all kwargs except pipeline_context
                                context_kwargs = {
                                    k: v
                                    for k, v in kwargs.items()
                                    if k not in ("context", "pipeline_context")
                                }
                                call_kwargs.update(context_kwargs)
                                if model.input is not None:
                                    result = func(data, **call_kwargs)
                                else:
                                    result = func(**call_kwargs)
                            if __inspect.isawaitable(result):
                                return await result
                            return result
                        except TypeError as e:
                            # Fallback: try passing data as first arg (but not for builtins)
                            if is_builtin:
                                raise BlueprintError(
                                    f"Builtin skill {getattr(func, '__name__', 'unknown')} failed: {e}"
                                ) from e
                            result = func(data, **dict(_params_for_callable))
                            if __inspect.isawaitable(result):
                                return await result
                            return result

                    # Preserve original function identity for validators
                    # Copy __name__ from original function or use skill_id
                    try:
                        if skill_id:
                            _runner.__name__ = skill_id
                        elif hasattr(func, "__name__"):
                            _runner.__name__ = func.__name__
                    except (AttributeError, TypeError):
                        pass  # Fine if we can't set __name__

                    return _runner

                callable_obj: Any = agent_obj
                try:
                    # Apply wrapper if there are params OR if this is a builtin (needs param normalization)
                    # Re-check builtin status here since is_builtin was defined in _with_params scope
                    is_builtin_check = False
                    skill_id_for_attr = None
                    try:
                        if isinstance(model.agent, dict):
                            skill_id_for_attr = model.agent.get("id", "")
                            is_builtin_check = isinstance(
                                skill_id_for_attr, str
                            ) and skill_id_for_attr.startswith("flujo.builtins.")
                    except (AttributeError, KeyError, TypeError):
                        is_builtin_check = False

                    should_wrap = callable(agent_obj) and (_params_for_callable or is_builtin_check)
                    if should_wrap:
                        callable_obj = _with_params(agent_obj)
                        # Preserve skill_id as __name__ for validators (e.g., V-S2 stringify detection)
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
                    # Preserve skill_id on agent for validators (e.g., V-S2 stringify detection)
                    if skill_id_for_attr and st is not None and st.agent is not None:
                        try:
                            st.agent.__name__ = skill_id_for_attr
                        except (AttributeError, TypeError):
                            pass
        # If still no callable-based step, create a plain Step with the agent_obj
        if st is None:
            st = Step[Any, Any](
                name=model.name,
                agent=agent_obj,
                config=step_config,
                updates_context=model.updates_context,
                validate_fields=model.validate_fields,
                sink_to=model.sink_to,
            )
        # Attach per-step AROS processing config to step.meta for policies
        try:
            if isinstance(model.processing, dict) and model.processing:
                # Validate and normalize processing config
                try:
                    pc = ProcessingConfigModel.model_validate(model.processing)
                    proc_dict = pc.model_dump(exclude_none=True, by_alias=True)
                except Exception as e:
                    raise BlueprintError(f"Invalid processing configuration: {e}") from e
                st.meta.setdefault("processing", {})
                st.meta["processing"].update(proc_dict)
        except Exception:
            pass
        # Attach templated input if provided in blueprint
        try:
            if model.input is not None:
                st.meta["templated_input"] = model.input
        except Exception:
            pass
        # Attach per-step use_history opt-out/in if preserved
        try:
            if _use_history_extra is not None:
                st.meta["use_history"] = bool(_use_history_extra)
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
        # Developer ergonomics: allow inline pipeline dicts like { steps: [...] }
        try:
            if "steps" in branch_spec:
                steps_val = branch_spec.get("steps")
                if isinstance(steps_val, list):
                    step_list: List[Step[Any, Any]] = []
                    for idx, s in enumerate(steps_val):
                        step_list.append(
                            _make_step_from_blueprint(
                                s,
                                yaml_path=(
                                    f"{base_path}.steps[{idx}]" if base_path is not None else None
                                ),
                                compiled_agents=compiled_agents,
                                compiled_imports=compiled_imports,
                            )
                        )
                    return Pipeline.model_construct(steps=step_list)
                else:
                    path_txt = base_path or "<branch>"
                    raise BlueprintError(
                        "Invalid inline pipeline: 'steps' must be a list of step dicts. "
                        f"Found type={type(steps_val).__name__} at {path_txt}.\n"
                        "Hint: either provide a list directly (e.g., states.s1: [ ... ]) or "
                        "define a single step dict without the 'steps:' wrapper."
                    )
        except BlueprintError:
            raise
        except Exception:
            # Fall through to single-step handling on unexpected shapes
            pass

        # Single step dict (canonical path)
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
    p: Pipeline[Any, Any] = Pipeline.model_construct(steps=steps)
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
            from ..dsl.dynamic_router import DynamicParallelRouterStep

            if isinstance(step, DynamicParallelRouterStep):
                branches = {
                    str(k): [step_to_yaml(s) for s in p.steps] for k, p in step.branches.items()
                }
                router_data: Dict[str, Any] = {"branches": branches}
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
                router_step_data: Dict[str, Any] = {
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
                # Optional sink_to
                try:
                    if getattr(step, "sink_to", None):
                        hitl_data["sink_to"] = getattr(step, "sink_to")
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


_skills_base_dir_stack: list[str] = []


def _push_skills_base_dir(dir_path: Optional[str]) -> None:
    if dir_path:
        try:
            _skills_base_dir_stack.append(dir_path)
        except Exception:
            pass


def _pop_skills_base_dir() -> None:
    try:
        if _skills_base_dir_stack:
            _skills_base_dir_stack.pop()
    except Exception:
        pass


def _current_skills_base_dir() -> Optional[str]:
    try:
        return _skills_base_dir_stack[-1] if _skills_base_dir_stack else None
    except Exception:
        return None


def load_pipeline_blueprint_from_yaml(
    yaml_text: str, base_dir: Optional[str] = None, source_file: Optional[str] = None
) -> Pipeline[Any, Any]:
    """Load a Pipeline from YAML with correct relative-import semantics.

    When ``base_dir`` is provided, temporarily add it to ``sys.path`` so that
    imports like ``skills.*`` inside that directory resolve during both
    validation and runtime. This ensures parity between ``dev validate`` and
    ``run`` without requiring PYTHONPATH/sitecustomize hacks.
    """
    import sys as _sys
    import os

    _pushed_sys_path = False
    # Ensure skills resolution for this load is scoped to base_dir
    _push_skills_base_dir(base_dir)
    if base_dir:
        try:
            base_dir_abs = os.path.abspath(base_dir)
            if base_dir_abs not in _sys.path:
                _sys.path.insert(0, base_dir_abs)
                _pushed_sys_path = True
        except Exception:
            # Non-fatal; relative skills may still be resolved via registry
            _pushed_sys_path = False
    # Proactively auto-load skills to honor docs: load skills.yaml before parsing.
    # This ensures CLI and any programmatic use benefit from the same behavior.
    try:
        # Build a best-effort YAML location index (path -> (line, col)) and comment-based suppressions
        # using ruamel.yaml when available.
        loc_index: Dict[str, Tuple[int, int]] = {}
        sup_index: Dict[str, List[str]] = {}
        try:
            from ruamel.yaml import YAML as _RYAML
            from ruamel.yaml.comments import CommentedMap as _CMap, CommentedSeq as _CSeq

            def _build_index(txt: str) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, List[str]]]:
                yaml_rt = _RYAML(typ="rt")
                root = yaml_rt.load(txt)
                idx: Dict[str, Tuple[int, int]] = {}
                sup: Dict[str, List[str]] = {}

                def _extract_ignores(comment_obj: Any) -> List[str]:
                    pats: List[str] = []
                    try:
                        import re as _re

                        texts: List[str] = []
                        if comment_obj is None:
                            return pats
                        # comment_obj may be a list/tuple of tokens or strings
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
                                # Split on comma/space
                                for part in body.replace("\t", " ").split(","):
                                    tok = part.strip()
                                    if tok:
                                        # Also split on whitespace to handle multi-space lists
                                        for sub in tok.split():
                                            if sub and sub not in pats:
                                                pats.append(sub)
                    except Exception:
                        return pats
                    return pats

                def _recurse(node: Any, path: str) -> None:
                    try:
                        if isinstance(node, _CMap):
                            for k, v in node.items():
                                key_path = f"{path}.{k}" if path else str(k)
                                # Record key position for mapping entries (e.g., fallback, wrapped_step, branches)
                                try:
                                    if hasattr(node, "lc") and hasattr(node.lc, "key"):
                                        pos = node.lc.key(k)
                                        if isinstance(pos, tuple) and len(pos) >= 2:
                                            idx[key_path] = (int(pos[0]) + 1, int(pos[1]) + 1)
                                    # Comments attached to mapping keys
                                    if hasattr(node, "ca") and hasattr(node.ca, "items"):
                                        ent = node.ca.items.get(k)
                                        if ent:
                                            pats = _extract_ignores(ent)
                                            if pats:
                                                sup.setdefault(key_path, []).extend(pats)
                                                # If this mapping is within a step item, also attach to the step path
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
                                # Special-case: steps list — record each item's position
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
                                        # Also index a synthetic '.steps[i]' path to align with loader yaml_path convention
                                        idx[f"{path}.steps[{i}]"] = line_col
                                    # Extract suppressions from sequence item comments
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
            from ...infra.skills_catalog import (
                load_skills_catalog as _load_skills_catalog,
                load_skills_entry_points as _load_skills_entry_points,
            )

            _load_skills_catalog(base_dir)
            _load_skills_entry_points()
    except Exception:
        # Never fail blueprint loading due to skills discovery issues
        pass
    # Ensure core builtin skills are registered so YAML agent ids like
    # 'flujo.builtins.stringify' resolve without requiring a separate import.
    try:  # best-effort; never fail if unavailable
        import flujo.builtins  # noqa: F401
    except Exception:
        pass
    try:
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
            p = build_pipeline_from_blueprint(bp)
            # Propagate top-level blueprint name onto the Pipeline object so downstream
            # components (CLI runner, tracing) can display and persist a meaningful name.
            try:
                name_val = getattr(bp, "name", None)
                if isinstance(name_val, str):
                    name_val_stripped = name_val.strip()
                    if name_val_stripped:
                        # Attach as a dynamic attribute; Pipeline does not model `name` as a field.
                        try:
                            setattr(p, "name", name_val_stripped)
                        except Exception:
                            # Bypass pydantic attribute guard if necessary
                            try:
                                object.__setattr__(p, "name", name_val_stripped)
                            except Exception:
                                pass
            except Exception:
                pass
            # Attach source_file on the Pipeline for import validation caching/cycle detection
            try:
                if source_file:
                    try:
                        setattr(p, "_source_file", str(source_file))
                    except Exception:
                        object.__setattr__(p, "_source_file", str(source_file))
            except Exception:
                pass
            # Attach ruamel-derived line/column to steps when yaml_path is present
            try:
                from ..dsl import Pipeline as _DPipe, Step as _DStep

                def _attach_step_loc(st: Any) -> None:
                    try:
                        meta = getattr(st, "meta", None)
                        if isinstance(meta, dict) and "yaml_path" in meta:
                            ypath = str(meta.get("yaml_path"))
                            if ypath in loc_index:
                                ln, col = loc_index[ypath]
                                info = {"path": ypath, "line": int(ln), "column": int(col)}
                                if source_file:
                                    info["file"] = str(source_file)
                                st.meta["_yaml_loc"] = info
                            # Attach comment-based suppressions when present
                            pats = sup_index.get(ypath) or []
                            if pats:
                                try:
                                    lst = st.meta.setdefault("suppress_rules", [])
                                    for ptn in pats:
                                        if ptn not in lst:
                                            lst.append(ptn)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    # Recurse into nested constructs: fallback, branches, default branch, loop/map bodies, wrapped
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
                    # State machine states
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

                def _attach_pipe_loc(pipe: Any) -> None:
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
    finally:
        # Remove base_dir from sys.path if we added it
        try:
            if _pushed_sys_path and base_dir:
                base_dir_abs = os.path.abspath(base_dir)
                if base_dir_abs in _sys.path:
                    _sys.path.remove(base_dir_abs)
        except Exception:
            pass
        # Pop scoped skills base_dir
        _pop_skills_base_dir()


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

    # Scoped resolution for child-local skills without sys.modules collisions
    if module_name == "skills" or module_name.startswith("skills."):
        base_dir = _current_skills_base_dir()
        if base_dir:
            try:
                import importlib.util as _iu
                import hashlib as _hashlib
                import os as _os
                import sys as _sys
                import types as _types

                # Build filesystem path under the child base dir
                tail = module_name.split(".", 1)[1] if module_name != "skills" else ""
                parts = [p for p in tail.split(".") if p]
                skills_root = _os.path.join(base_dir, "skills")

                # Compute the source path (file or package __init__.py)
                mod_path = _os.path.join(skills_root, *parts)
                is_package = False
                py_path = mod_path + (".py" if parts else _os.sep + "__init__.py")
                if parts and not _os.path.exists(py_path):
                    # Try package directory with __init__.py
                    pkg_init = _os.path.join(skills_root, *parts, "__init__.py")
                    if _os.path.exists(pkg_init):
                        py_path = pkg_init
                        is_package = True
                elif not parts:
                    # 'skills' root is always a package
                    is_package = True
                if not _os.path.exists(py_path):
                    raise BlueprintError(
                        f"Unable to locate module '{module_name}' under '{base_dir}/skills'"
                    )

                # Derive a unique, stable package prefix for this child
                token = _hashlib.sha1(str(base_dir).encode("utf-8")).hexdigest()[:10]
                root_pkg = f"__flujo_import__{token}"
                skills_pkg = f"{root_pkg}.skills"

                # Ensure root and skills packages exist with proper __path__
                if root_pkg not in _sys.modules:
                    root_mod = _types.ModuleType(root_pkg)
                    # __path__ marks as package; empty list is acceptable for a synthetic root
                    root_mod.__path__ = []
                    root_mod.__package__ = root_pkg
                    _sys.modules[root_pkg] = root_mod
                if skills_pkg not in _sys.modules:
                    skills_mod = _types.ModuleType(skills_pkg)
                    skills_mod.__path__ = [skills_root]
                    skills_mod.__package__ = skills_pkg
                    _sys.modules[skills_pkg] = skills_mod

                # Ensure any intermediate packages for dotted parts exist
                pkg_prefix = skills_pkg
                pkg_dir = skills_root
                for i, part in enumerate(parts[:-1]):
                    pkg_prefix = f"{pkg_prefix}.{part}"
                    pkg_dir = _os.path.join(pkg_dir, part)
                    if pkg_prefix not in _sys.modules:
                        pm = _types.ModuleType(pkg_prefix)
                        pm.__path__ = [pkg_dir]
                        pm.__package__ = pkg_prefix
                        _sys.modules[pkg_prefix] = pm

                # Determine fully qualified module name under isolated package
                fqmn = skills_pkg if not parts else f"{skills_pkg}.{'.'.join(parts)}"

                if fqmn in _sys.modules:
                    mod = _sys.modules[fqmn]
                else:
                    # For packages, provide submodule_search_locations
                    subloc = [mod_path] if is_package else None
                    spec = _iu.spec_from_file_location(
                        fqmn, py_path, submodule_search_locations=subloc
                    )
                    if spec is None or spec.loader is None:
                        raise BlueprintError(
                            f"Unable to locate module '{module_name}' at '{py_path}'"
                        )
                    mod = _iu.module_from_spec(spec)
                    _sys.modules[fqmn] = mod
                    spec.loader.exec_module(mod)
                return getattr(mod, attr_name) if attr_name else mod
            except Exception as e:
                raise BlueprintError(
                    f"Failed to import child-local module '{module_name}': {e}"
                ) from e

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

    # Normalize input prior to field validation so we can coerce boolean
    # branch keys for conditional steps (FSD-026)
    @model_validator(mode="before")
    @classmethod
    def _normalize_conditional_branch_keys(cls, data: Any) -> Any:
        """Coerce boolean branch keys to strings for conditional steps.

        YAML parses unquoted 'true'/'false' as booleans, which previously
        failed validation because 'branches' expected string keys. For
        kind: conditional, convert boolean keys to their lowercase string
        equivalents ('true'/'false').
        """
        try:
            if isinstance(data, dict) and str(data.get("kind", "")) == "conditional":
                branches = data.get("branches")
                if isinstance(branches, dict):
                    coerced: Dict[str, Any] = {}
                    for k, v in branches.items():
                        if isinstance(k, bool):
                            coerced[str(k).lower()] = v
                        else:
                            coerced[str(k)] = v if k not in coerced else v
                    new_data = dict(data)
                    new_data["branches"] = coerced
                    return new_data
        except Exception:
            # Be conservative: on any error, return original input
            return data
        return data
