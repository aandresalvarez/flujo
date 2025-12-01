from __future__ import annotations

from typing import Any, Optional, cast

from ..dsl import Pipeline, Step, StepConfig
from .loader_models import (
    BlueprintError,
    BlueprintPipelineModel,
    BlueprintStepModel,
    ProcessingConfigModel,
)
from .loader_steps_branching import (
    build_conditional_step,
    build_dynamic_router_step,
    build_parallel_step,
)
from .loader_steps_common import _finalize_step_types
from .loader_steps_loop import build_loop_step, build_map_step
from .loader_steps_misc import (
    build_agentic_loop_step,
    build_basic_step,
    build_cache_step,
    build_hitl_step,
)


def _make_step_from_blueprint(
    model: Any,
    *,
    yaml_path: Optional[str] = None,
    compiled_agents: Optional[dict[str, Any]] = None,
    compiled_imports: Optional[dict[str, Any]] = None,
) -> Step[Any, Any]:
    if isinstance(model, dict):
        _raw_use_history = None
        try:
            if "use_history" in model:
                _raw_use_history = bool(model.get("use_history"))
        except Exception:
            _raw_use_history = None
        kind_val = str(model.get("kind", "step"))
        if kind_val == "conditional":
            try:
                branches_raw = model.get("branches")
                if isinstance(branches_raw, dict):
                    coerced: dict[str, Any] = {}
                    for _k, _v in branches_raw.items():
                        if isinstance(_k, bool):
                            coerced[str(_k).lower()] = _v
                        else:
                            coerced[str(_k)] = _v if _k not in coerced else _v
                    model = dict(model)
                    model["branches"] = coerced
            except Exception:
                pass
        if kind_val == "StateMachine":
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
            coerced_states: dict[str, Pipeline[Any, Any]] = {}
            for _state_name, _branch_spec in states_raw.items():
                coerced_states[str(_state_name)] = _build_pipeline_from_branch(
                    _branch_spec,
                    base_path=f"{yaml_path}.states.{_state_name}" if yaml_path else None,
                    compiled_agents=compiled_agents,
                    compiled_imports=compiled_imports,
                )
            transitions_raw = model.get("transitions")
            if transitions_raw is not None and not isinstance(transitions_raw, list):
                raise BlueprintError("StateMachine.transitions must be a list of rules")
            coerced_transitions = []
            if isinstance(transitions_raw, list):
                for _idx, _rule in enumerate(transitions_raw):
                    if isinstance(_rule, dict):
                        _coerced: dict[str, Any] = {}
                        for _k, _v in _rule.items():
                            if _k is True:
                                _coerced["on"] = _v
                            elif _k is False:
                                _coerced["off"] = _v
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
            try:
                proc_raw = getattr(model, "processing", None)
                if isinstance(proc_raw, dict) and proc_raw:
                    pc = ProcessingConfigModel.model_validate(proc_raw)
                    try:
                        setattr(
                            model, "processing", pc.model_dump(exclude_none=True, by_alias=True)
                        )
                    except Exception:
                        pass
            except Exception as e:
                raise BlueprintError(f"Invalid processing configuration: {e}") from e
            try:
                if _raw_use_history is not None:
                    setattr(model, "_use_history_extra", _raw_use_history)
            except Exception:
                pass
        else:
            try:
                from ...framework import registry as _fwreg

                step_cls = _fwreg.get_step_class(kind_val)
            except Exception:
                step_cls = None
            if step_cls is not None:
                try:
                    step_obj = step_cls.model_validate(model)
                except Exception as e:
                    raise BlueprintError(
                        f"Failed to instantiate custom step for kind '{kind_val}': {e}"
                    )
                if yaml_path:
                    try:
                        step_obj.meta["yaml_path"] = yaml_path
                    except Exception:
                        pass
                return step_obj
            raise BlueprintError(f"Unknown step kind: {kind_val}")

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
        return cast(
            Step[Any, Any],
            build_parallel_step(
                model,
                step_config,
                yaml_path=yaml_path,
                compiled_agents=compiled_agents,
                compiled_imports=compiled_imports,
                build_branch=_build_pipeline_from_branch,
            ),
        )
    if getattr(model, "kind", None) == "conditional":
        return cast(
            Step[Any, Any],
            build_conditional_step(
                model,
                step_config,
                yaml_path=yaml_path,
                compiled_agents=compiled_agents,
                compiled_imports=compiled_imports,
                build_branch=_build_pipeline_from_branch,
            ),
        )
    if getattr(model, "kind", None) == "loop":
        return cast(
            Step[Any, Any],
            build_loop_step(
                model,
                step_config,
                yaml_path=yaml_path,
                compiled_agents=compiled_agents,
                compiled_imports=compiled_imports,
                build_branch=_build_pipeline_from_branch,
            ),
        )
    if getattr(model, "kind", None) == "map":
        return cast(
            Step[Any, Any],
            build_map_step(
                model,
                step_config,
                yaml_path=yaml_path,
                compiled_agents=compiled_agents,
                compiled_imports=compiled_imports,
                build_branch=_build_pipeline_from_branch,
            ),
        )
    if getattr(model, "kind", None) == "dynamic_router":
        return cast(
            Step[Any, Any],
            build_dynamic_router_step(
                model,
                step_config,
                yaml_path=yaml_path,
                compiled_agents=compiled_agents,
                compiled_imports=compiled_imports,
                build_branch=_build_pipeline_from_branch,
            ),
        )
    if getattr(model, "kind", None) == "hitl":
        return cast(Step[Any, Any], build_hitl_step(model, step_config))
    if getattr(model, "kind", None) == "cache":
        return cast(
            Step[Any, Any],
            build_cache_step(
                model,
                yaml_path=yaml_path,
                compiled_agents=compiled_agents,
                compiled_imports=compiled_imports,
                make_step_fn=_make_step_from_blueprint,
            ),
        )
    if getattr(model, "kind", None) == "agentic_loop":
        return cast(Step[Any, Any], build_agentic_loop_step(model, step_config))

    return build_basic_step(
        model,
        step_config,
        yaml_path=yaml_path,
        compiled_agents=compiled_agents,
        compiled_imports=compiled_imports,
        make_step_fn=_make_step_from_blueprint,
    )


def _build_pipeline_from_branch(
    branch_spec: Any,
    *,
    base_path: Optional[str] = None,
    compiled_agents: Optional[dict[str, Any]] = None,
    compiled_imports: Optional[dict[str, Any]] = None,
) -> Pipeline[Any, Any]:
    if isinstance(branch_spec, list):
        steps: list[Step[Any, Any]] = []
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
    if isinstance(branch_spec, dict):
        try:
            if "steps" in branch_spec:
                steps_val = branch_spec.get("steps")
                if isinstance(steps_val, list):
                    step_list: list[Step[Any, Any]] = []
                    for idx, s in enumerate(steps_val):
                        step_list.append(
                            _make_step_from_blueprint(
                                s,
                                yaml_path=f"{base_path}.steps[{idx}]"
                                if base_path is not None
                                else None,
                                compiled_agents=compiled_agents,
                                compiled_imports=compiled_imports,
                            )
                        )
                    return Pipeline.model_construct(steps=step_list)
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
            pass

        return Pipeline.from_step(
            _make_step_from_blueprint(
                branch_spec,
                yaml_path=f"{base_path}.steps[0]" if base_path is not None else None,
                compiled_agents=compiled_agents,
                compiled_imports=compiled_imports,
            )
        )
    raise BlueprintError("Invalid branch specification; expected dict or list of dicts")


def build_pipeline_from_blueprint(
    model: BlueprintPipelineModel,
    compiled_agents: Optional[dict[str, Any]] = None,
    compiled_imports: Optional[dict[str, Any]] = None,
) -> Pipeline[Any, Any]:
    steps: list[Step[Any, Any]] = []
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
    try:
        for st in p.steps:
            _finalize_step_types(st)
    except Exception:
        pass
    return p


__all__ = [
    "_build_pipeline_from_branch",
    "_make_step_from_blueprint",
    "build_pipeline_from_blueprint",
]
