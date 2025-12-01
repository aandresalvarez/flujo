from __future__ import annotations

import re
from typing import Any, Callable, Optional

from ..dsl import ParallelStep, Pipeline, StepConfig
from .loader_models import BlueprintError, BlueprintStepModel
from .loader_resolution import _import_object, _resolve_agent_entry
from .loader_steps_common import _normalize_branch_failure, _normalize_merge_strategy

BuildBranch = Callable[..., Pipeline[Any, Any]]


def build_parallel_step(
    model: BlueprintStepModel,
    step_config: StepConfig,
    *,
    yaml_path: Optional[str],
    compiled_agents: Optional[dict[str, Any]],
    compiled_imports: Optional[dict[str, Any]],
    build_branch: BuildBranch,
) -> ParallelStep[Any]:
    if not model.branches:
        raise BlueprintError("parallel step requires branches")
    branches_map: dict[str, Pipeline[Any, Any]] = {}
    for branch_name, branch_spec in model.branches.items():
        branches_map[branch_name] = build_branch(
            branch_spec,
            base_path=f"{yaml_path}.branches.{branch_name}" if yaml_path else None,
            compiled_agents=compiled_agents,
            compiled_imports=compiled_imports,
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
    _attach_parallel_reduce(model, st_par, branches_map)
    return st_par


def _attach_parallel_reduce(
    model: BlueprintStepModel,
    st_par: ParallelStep[Any],
    branches_map: dict[str, Pipeline[Any, Any]],
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
        if isinstance(reduce_spec, str):
            mode = reduce_spec.strip().lower()
        else:
            mode = str(reduce_spec.get("mode", "")).strip().lower()

        def _reduce(output_map: dict[str, Any], _ctx: Optional[Any]) -> Any:
            if mode == "keys":
                return [bn for bn in branch_order if bn in output_map]
            if mode == "values":
                return [output_map[bn] for bn in branch_order if bn in output_map]
            if mode == "union":
                acc: dict[str, Any] = {}
                for bn in branch_order:
                    val = output_map.get(bn)
                    if isinstance(val, dict):
                        acc.update(val)
                return acc
            if mode == "concat":
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
    compiled_agents: Optional[dict[str, Any]],
    compiled_imports: Optional[dict[str, Any]],
    build_branch: BuildBranch,
) -> Any:
    from ..dsl.conditional import ConditionalStep

    if not model.branches:
        raise BlueprintError("conditional step requires branches")
    branches_map: dict[Any, Pipeline[Any, Any]] = {}
    for key, branch_spec in model.branches.items():
        branches_map[key] = build_branch(
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
    st_cond: ConditionalStep[Any] = ConditionalStep(
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
    model: BlueprintStepModel, branches_map: dict[Any, Pipeline[Any, Any]]
) -> Any:
    if model.condition:
        try:
            _cond_callable = _import_object(model.condition)
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
            return _cond_callable
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

            def _expr_cond(output: Any, _ctx: Optional[Any]) -> Any:
                return _expr_fn(output, _ctx)

            return _expr_cond
        except Exception as e:
            raise BlueprintError(f"Invalid condition_expression: {e}") from e

    def _default_cond(output: Any, _ctx: Optional[Any]) -> Any:
        return output if output in branches_map else next(iter(branches_map))

    return _default_cond


def build_dynamic_router_step(
    model: BlueprintStepModel,
    step_config: StepConfig,
    *,
    yaml_path: Optional[str],
    compiled_agents: Optional[dict[str, Any]],
    compiled_imports: Optional[dict[str, Any]],
    build_branch: BuildBranch,
) -> Any:
    from ..dsl.dynamic_router import DynamicParallelRouterStep

    if not model.router or "router_agent" not in model.router or "branches" not in model.router:
        raise BlueprintError("dynamic_router requires router.router_agent and router.branches")
    router_agent = _resolve_agent_entry(model.router.get("router_agent") or "")
    branches_router: dict[str, Pipeline[Any, Any]] = {}
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
