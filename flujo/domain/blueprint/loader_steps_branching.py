from __future__ import annotations

import re
from typing import Callable, Optional, TypeAlias, TypeGuard

from ..dsl import ParallelStep, Pipeline, Step, StepConfig
from ..dsl.step import BranchKey
from ..models import BaseModel
from .loader_models import BlueprintError, BlueprintStepModel
from .loader_resolution import _import_object, _resolve_agent_entry
from .loader_steps_common import _normalize_branch_failure, _normalize_merge_strategy
from .loader_steps_misc import _resolve_callable_spec

AnyPipeline: TypeAlias = Pipeline[object, object]
AnyStep: TypeAlias = Step[object, object]
CompiledAgents: TypeAlias = dict[str, object]
CompiledImports: TypeAlias = dict[str, AnyPipeline]
BuildBranch = Callable[..., AnyPipeline]
ConditionCallable: TypeAlias = Callable[[object, BaseModel | None], object]


def _is_condition_callable(obj: object) -> TypeGuard[ConditionCallable]:
    return callable(obj)


def build_parallel_step(
    model: BlueprintStepModel,
    step_config: StepConfig,
    *,
    yaml_path: Optional[str],
    compiled_agents: CompiledAgents | None,
    compiled_imports: CompiledImports | None,
    build_branch: BuildBranch,
) -> ParallelStep[BaseModel]:
    if not model.branches:
        raise BlueprintError("parallel step requires branches")
    branches_map: dict[str, AnyPipeline] = {}
    for branch_name, branch_spec in model.branches.items():
        branches_map[branch_name] = build_branch(
            branch_spec,
            base_path=f"{yaml_path}.branches.{branch_name}" if yaml_path else None,
            compiled_agents=compiled_agents,
            compiled_imports=compiled_imports,
        )
    st_par: ParallelStep[BaseModel] = ParallelStep(
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
    _attach_parallel_reduce(model, st_par, branches_map)
    return st_par


def _attach_parallel_reduce(
    model: BlueprintStepModel,
    st_par: ParallelStep[BaseModel],
    branches_map: dict[str, AnyPipeline],
) -> None:
    try:
        reduce_spec = model.reduce
    except Exception:
        reduce_spec = None
    if not isinstance(reduce_spec, (str, dict)) or not reduce_spec:
        return
    try:
        branch_order = list(branches_map.keys())
        mode: str
        callable_spec: object | None = None
        if isinstance(reduce_spec, str):
            mode = reduce_spec.strip().lower()
        else:
            mode = str(reduce_spec.get("mode", "")).strip().lower()
            callable_spec = reduce_spec.get("callable") or reduce_spec.get("path")

        if callable_spec is not None:
            reducer = _resolve_callable_spec(callable_spec, label="reduce")
            st_par.reduce = reducer
            return

        if mode in {"majority_vote", "code_consensus", "judge_selection"}:
            from ..consensus import majority_vote, code_consensus, judge_selection

            if mode == "majority_vote":
                st_par.reduce = majority_vote
                return
            if mode == "code_consensus":
                st_par.reduce = code_consensus
                return
            if mode == "judge_selection":
                evaluator_spec = None
                if isinstance(reduce_spec, dict):
                    evaluator_spec = reduce_spec.get("evaluator")
                evaluator = None
                if evaluator_spec is not None:
                    try:
                        evaluator = _resolve_agent_entry(evaluator_spec)
                    except Exception:
                        evaluator = None
                st_par.reduce = judge_selection(evaluator)
                return

        if mode and mode not in {"keys", "values", "union", "concat", "first", "last"}:
            reducer = _resolve_callable_spec(mode, label="reduce")
            if reducer is not None:
                st_par.reduce = reducer
                return

        def _reduce(output_map: dict[str, object], _ctx: BaseModel | None) -> object:
            if mode == "keys":
                return [bn for bn in branch_order if bn in output_map]
            if mode == "values":
                return [output_map[bn] for bn in branch_order if bn in output_map]
            if mode == "union":
                acc: dict[str, object] = {}
                for bn in branch_order:
                    val = output_map.get(bn)
                    if isinstance(val, dict):
                        for k, v in val.items():
                            acc[str(k)] = v
                return acc
            if mode == "concat":
                res: list[object] = []
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
            return output_map

        try:
            st_par.meta["parallel_reduce_mapper"] = _reduce
        except Exception:
            pass
    except Exception:
        pass


def build_conditional_step(
    model: BlueprintStepModel,
    step_config: StepConfig,
    *,
    yaml_path: Optional[str],
    compiled_agents: CompiledAgents | None,
    compiled_imports: CompiledImports | None,
    build_branch: BuildBranch,
) -> AnyStep:
    from ..dsl.conditional import ConditionalStep

    if not model.branches:
        raise BlueprintError("conditional step requires branches")
    branches_map: dict[BranchKey, AnyPipeline] = {}
    for key, branch_spec in model.branches.items():
        branches_map[str(key) if not isinstance(key, (str, bool, int)) else key] = build_branch(
            branch_spec,
            base_path=f"{yaml_path}.branches.{key}" if yaml_path else None,
            compiled_agents=compiled_agents,
            compiled_imports=compiled_imports,
        )
    _cond_callable = _build_condition_callable(model, branches_map)
    default_branch = (
        build_branch(
            model.default_branch,
            base_path=f"{yaml_path}.default_branch" if yaml_path else None,
            compiled_agents=compiled_agents,
            compiled_imports=compiled_imports,
        )
        if model.default_branch
        else None
    )
    st_cond: ConditionalStep[BaseModel] = ConditionalStep(
        name=model.name,
        condition_callable=_cond_callable,
        branches=branches_map,
        default_branch_pipeline=default_branch,
        config=step_config,
    )
    try:
        if model.condition_expression:
            st_cond.meta["condition_expression"] = str(model.condition_expression)
    except Exception:
        pass
    return st_cond


def _build_condition_callable(
    model: BlueprintStepModel, branches_map: dict[BranchKey, AnyPipeline]
) -> Callable[[object, BaseModel | None], BranchKey]:
    if model.condition:
        try:
            raw_callable = _import_object(model.condition)
            if not _is_condition_callable(raw_callable):
                raise BlueprintError(
                    f"condition '{model.condition}' did not resolve to a callable."
                )
            _cond_callable = raw_callable
            import asyncio

            if asyncio.iscoroutinefunction(_cond_callable):
                raise BlueprintError(
                    f"condition '{model.condition}' must be synchronous.\n"
                    f"Conditional step conditions are called synchronously and cannot be async functions.\n"
                    f"\n"
                    f"Change your function from:\n"
                    f"  async def my_condition(data, context) -> object:\n"
                    f"      ...\n"
                    f"\n"
                    f"To:\n"
                    f"  def my_condition(data, context) -> object:\n"
                    f"      ...\n"
                    f"\n"
                    f"Remove 'async' and any 'await' calls in your condition function.\n"
                    f"See: https://flujo.dev/docs/user_guide/pipeline_branching#conditional-steps"
                )

            def _checked(output: object, ctx: BaseModel | None) -> BranchKey:
                res = _cond_callable(output, ctx)
                if isinstance(res, (str, bool, int)):
                    return res
                raise BlueprintError(
                    "condition callable must return a branch key (str | bool | int); "
                    f"got {type(res)}"
                )

            return _checked
        except Exception as exc:
            try:
                _cond_str = str(model.condition).strip()
            except Exception:
                _cond_str = ""
            if re.match(r"^\(?\s*lambda\b", _cond_str):
                raise BlueprintError(
                    "Invalid condition value: inline Python (e.g., a lambda expression) is not supported in YAML. "
                    "Use 'condition_expression' for inline logic or reference an importable callable like 'pkg.mod:func'.\n"
                    'Example: condition_expression: "{{ previous_step }}"'
                ) from exc
            raise BlueprintError(
                f"Failed to resolve condition '{_cond_str}' (field: condition). "
                "Provide a Python import path like 'pkg.mod:func' or use 'condition_expression'. "
                f"Underlying error: {exc}"
            ) from exc
    if model.condition_expression:
        try:
            from ...utils.expressions import compile_expression_to_callable as _compile_expr

            _expr_fn = _compile_expr(str(model.condition_expression))

            def _expr_cond(output: object, _ctx: BaseModel | None) -> BranchKey:
                res = _expr_fn(output, _ctx)
                if isinstance(res, (str, bool, int)):
                    return res
                raise BlueprintError(
                    "condition_expression must evaluate to a branch key (str | bool | int); "
                    f"got {type(res)}"
                )

            return _expr_cond
        except Exception as e:
            raise BlueprintError(f"Invalid condition_expression: {e}") from e

    def _default_cond(output: object, _ctx: BaseModel | None) -> BranchKey:
        if isinstance(output, (str, bool, int)) and output in branches_map:
            return output
        return next(iter(branches_map))

    return _default_cond


def build_dynamic_router_step(
    model: BlueprintStepModel,
    step_config: StepConfig,
    *,
    yaml_path: Optional[str],
    compiled_agents: CompiledAgents | None,
    compiled_imports: CompiledImports | None,
    build_branch: BuildBranch,
) -> AnyStep:
    from ..dsl.dynamic_router import DynamicParallelRouterStep

    if not model.router or "router_agent" not in model.router or "branches" not in model.router:
        raise BlueprintError("dynamic_router requires router.router_agent and router.branches")
    router_agent = _resolve_agent_entry(model.router.get("router_agent") or "")
    branches_router: dict[str, AnyPipeline] = {}
    for bname, bspec in model.router.get("branches", {}).items():
        branches_router[bname] = build_branch(
            bspec,
            base_path=f"{yaml_path}.router.branches.{bname}" if yaml_path else None,
            compiled_agents=compiled_agents,
            compiled_imports=compiled_imports,
        )
    return DynamicParallelRouterStep(
        name=model.name,
        router_agent=router_agent,
        branches=branches_router,
        config=step_config,
    )


__all__ = [
    "BuildBranch",
    "build_conditional_step",
    "build_dynamic_router_step",
    "build_parallel_step",
]
