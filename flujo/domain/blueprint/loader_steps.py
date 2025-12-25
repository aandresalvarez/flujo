from __future__ import annotations

from typing import Optional, TypeAlias, TypeGuard

from ..dsl import Pipeline, Step, StepConfig
from ..dsl.step import InvariantRule
from .loader_models import (
    BlueprintError,
    BlueprintPipelineModel,
    BlueprintStepModel,
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
    build_tree_search_step,
    build_state_machine_step,
)

AnyStep: TypeAlias = Step[object, object]
AnyPipeline: TypeAlias = Pipeline[object, object]
CompiledAgents: TypeAlias = dict[str, object]
CompiledImports: TypeAlias = dict[str, AnyPipeline]


def _is_invariant_rule(rule: object) -> TypeGuard[InvariantRule]:
    return isinstance(rule, str) or callable(rule)


def _make_step_from_blueprint(
    model: object,
    *,
    yaml_path: Optional[str] = None,
    compiled_agents: CompiledAgents | None = None,
    compiled_imports: CompiledImports | None = None,
) -> AnyStep:
    kind_val = "step"
    _raw_use_history: bool | None = None
    if isinstance(model, dict):
        try:
            if "use_history" in model:
                _raw_use_history = bool(model.get("use_history"))
        except Exception:
            _raw_use_history = None
        kind_val = str(model.get("kind", "step"))
    else:
        # It's an object (likely BlueprintStepModel)
        try:
            kind_val = getattr(model, "kind", "step")
        except Exception:
            kind_val = "step"

    # --- REGISTRY DISPATCH ---
    from .builder_registry import get_builder

    builder = get_builder(kind_val)
    if builder is not None:
        # Built-in kinds use the BlueprintStepModel → StepConfig normalization path.
        if kind_val not in {
            "step",
            "parallel",
            "conditional",
            "loop",
            "map",
            "dynamic_router",
            "hitl",
            "cache",
            "agentic_loop",
            "tree_search",
            "StateMachine",
        }:
            return builder(
                model,
                yaml_path=yaml_path,
                compiled_agents=compiled_agents,
                compiled_imports=compiled_imports,
            )

        # Common prep for registered builders (validation + config)
        # Note: We validate to BlueprintStepModel for all built-in kinds, including StateMachine.
        if not isinstance(model, BlueprintStepModel):
            if isinstance(model, dict):
                try:
                    model = BlueprintStepModel.model_validate(model)
                except Exception as e:
                    raise BlueprintError(
                        f"Invalid step configuration for kind '{kind_val}': {e}"
                    ) from e
                try:
                    if _raw_use_history is not None:
                        setattr(model, "_use_history_extra", _raw_use_history)
                except Exception:
                    pass
            else:
                raise BlueprintError(
                    f"Invalid step model: expected dict or BlueprintStepModel, got {type(model).__name__}"
                )

        # Extract config
        step_config = StepConfig()
        if hasattr(model, "config") and model.config:
            cfg_dict = dict(model.config)
            if "timeout" in cfg_dict and "timeout_s" not in cfg_dict:
                try:
                    cfg_dict["timeout_s"] = float(cfg_dict.pop("timeout"))
                except Exception:
                    cfg_dict.pop("timeout", None)
            step_config = StepConfig(**cfg_dict)

        return builder(
            model,
            step_config,
            yaml_path=yaml_path,
            compiled_agents=compiled_agents,
            compiled_imports=compiled_imports,
            # We pass these common kwargs that adapters can use
            # Adapters will filter what they need or we must standardize
        )

    raise BlueprintError(f"Unknown step kind: {kind_val}")


def _build_pipeline_from_branch(
    branch_spec: object,
    *,
    base_path: Optional[str] = None,
    compiled_agents: CompiledAgents | None = None,
    compiled_imports: CompiledImports | None = None,
) -> AnyPipeline:
    if isinstance(branch_spec, list):
        steps: list[AnyStep] = []
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
                    step_list: list[AnyStep] = []
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


# --- REGISTRY POPULATION ---


def _register_defaults() -> None:
    from .builder_registry import register_builder

    # Adapters
    def _adapt_parallel(
        model: BlueprintStepModel,
        config: StepConfig,
        *,
        yaml_path: Optional[str] = None,
        compiled_agents: CompiledAgents | None = None,
        compiled_imports: CompiledImports | None = None,
    ) -> AnyStep:
        return build_parallel_step(
            model,
            config,
            yaml_path=yaml_path,
            compiled_agents=compiled_agents,
            compiled_imports=compiled_imports,
            build_branch=_build_pipeline_from_branch,
        )

    def _adapt_conditional(
        model: BlueprintStepModel,
        config: StepConfig,
        *,
        yaml_path: Optional[str] = None,
        compiled_agents: CompiledAgents | None = None,
        compiled_imports: CompiledImports | None = None,
    ) -> AnyStep:
        return build_conditional_step(
            model,
            config,
            yaml_path=yaml_path,
            compiled_agents=compiled_agents,
            compiled_imports=compiled_imports,
            build_branch=_build_pipeline_from_branch,
        )

    def _adapt_loop(
        model: BlueprintStepModel,
        config: StepConfig,
        *,
        yaml_path: Optional[str] = None,
        compiled_agents: CompiledAgents | None = None,
        compiled_imports: CompiledImports | None = None,
    ) -> AnyStep:
        return build_loop_step(
            model,
            config,
            yaml_path=yaml_path,
            compiled_agents=compiled_agents,
            compiled_imports=compiled_imports,
            build_branch=_build_pipeline_from_branch,
        )

    def _adapt_map(
        model: BlueprintStepModel,
        config: StepConfig,
        *,
        yaml_path: Optional[str] = None,
        compiled_agents: CompiledAgents | None = None,
        compiled_imports: CompiledImports | None = None,
    ) -> AnyStep:
        return build_map_step(
            model,
            config,
            yaml_path=yaml_path,
            compiled_agents=compiled_agents,
            compiled_imports=compiled_imports,
            build_branch=_build_pipeline_from_branch,
        )

    def _adapt_router(
        model: BlueprintStepModel,
        config: StepConfig,
        *,
        yaml_path: Optional[str] = None,
        compiled_agents: CompiledAgents | None = None,
        compiled_imports: CompiledImports | None = None,
    ) -> AnyStep:
        return build_dynamic_router_step(
            model,
            config,
            yaml_path=yaml_path,
            compiled_agents=compiled_agents,
            compiled_imports=compiled_imports,
            build_branch=_build_pipeline_from_branch,
        )

    def _adapt_cache(
        model: BlueprintStepModel,
        _config: StepConfig,
        *,
        yaml_path: Optional[str] = None,
        compiled_agents: CompiledAgents | None = None,
        compiled_imports: CompiledImports | None = None,
    ) -> AnyStep:
        # cache step signature is specific
        return build_cache_step(
            model,
            yaml_path=yaml_path,
            compiled_agents=compiled_agents,
            compiled_imports=compiled_imports,
            make_step_fn=_make_step_from_blueprint,
        )

    def _adapt_basic(
        model: BlueprintStepModel,
        config: StepConfig,
        *,
        yaml_path: Optional[str] = None,
        compiled_agents: CompiledAgents | None = None,
        compiled_imports: CompiledImports | None = None,
    ) -> AnyStep:
        return build_basic_step(
            model,
            config,
            yaml_path=yaml_path,
            compiled_agents=compiled_agents,
            compiled_imports=compiled_imports,
            make_step_fn=_make_step_from_blueprint,
        )

    def _adapt_hitl(
        model: BlueprintStepModel,
        config: StepConfig,
        *,
        yaml_path: Optional[str] = None,
        compiled_agents: CompiledAgents | None = None,
        compiled_imports: CompiledImports | None = None,
    ) -> AnyStep:
        _ = (yaml_path, compiled_agents, compiled_imports)
        return build_hitl_step(model, config)

    def _adapt_agentic_loop(
        model: BlueprintStepModel,
        config: StepConfig,
        *,
        yaml_path: Optional[str] = None,
        compiled_agents: CompiledAgents | None = None,
        compiled_imports: CompiledImports | None = None,
    ) -> AnyStep:
        _ = (yaml_path, compiled_agents, compiled_imports)
        return build_agentic_loop_step(model, config)

    def _adapt_state_machine(
        model: BlueprintStepModel,
        config: StepConfig,
        *,
        yaml_path: Optional[str] = None,
        compiled_agents: CompiledAgents | None = None,
        compiled_imports: CompiledImports | None = None,
    ) -> AnyStep:
        return build_state_machine_step(
            model,
            config,
            yaml_path=yaml_path,
            compiled_agents=compiled_agents,
            compiled_imports=compiled_imports,
            build_branch=_build_pipeline_from_branch,
        )

    def _adapt_tree_search(
        model: BlueprintStepModel,
        config: StepConfig,
        *,
        yaml_path: Optional[str] = None,
        compiled_agents: CompiledAgents | None = None,
        compiled_imports: CompiledImports | None = None,
    ) -> AnyStep:
        return build_tree_search_step(
            model,
            config,
            yaml_path=yaml_path,
            compiled_agents=compiled_agents,
            compiled_imports=compiled_imports,
        )

    register_builder("parallel", _adapt_parallel)
    register_builder("conditional", _adapt_conditional)
    register_builder("loop", _adapt_loop)
    register_builder("map", _adapt_map)
    register_builder("dynamic_router", _adapt_router)
    register_builder("cache", _adapt_cache)
    register_builder("hitl", _adapt_hitl)
    register_builder("agentic_loop", _adapt_agentic_loop)
    register_builder("tree_search", _adapt_tree_search)
    register_builder("StateMachine", _adapt_state_machine)
    register_builder("step", _adapt_basic)


# Register on import
_register_defaults()


def build_pipeline_from_blueprint(
    model: BlueprintPipelineModel,
    compiled_agents: CompiledAgents | None = None,
    compiled_imports: CompiledImports | None = None,
) -> AnyPipeline:
    steps: list[AnyStep] = []
    for idx, s in enumerate(model.steps):
        steps.append(
            _make_step_from_blueprint(
                s,
                yaml_path=f"steps[{idx}]",
                compiled_agents=compiled_agents,
                compiled_imports=compiled_imports,
            )
        )
    pipeline_invariants: list[InvariantRule] = []
    if model.static_invariants:
        from .loader_steps_misc import _resolve_callable_spec

        for rule in model.static_invariants:
            if _is_invariant_rule(rule):
                pipeline_invariants.append(rule)
                continue
            if isinstance(rule, dict):
                resolved = _resolve_callable_spec(rule, label="pipeline_invariant")
                if resolved is not None and _is_invariant_rule(resolved):
                    pipeline_invariants.append(resolved)
                    continue
            raise BlueprintError(f"Invalid pipeline invariant: {rule!r}")

    p: AnyPipeline = Pipeline.model_construct(steps=steps, static_invariants=pipeline_invariants)
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
