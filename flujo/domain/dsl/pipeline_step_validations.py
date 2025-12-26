from __future__ import annotations

import logging
import typing
from typing import TYPE_CHECKING, TypeVar

from ..pipeline_validation import ValidationFinding, ValidationReport
from .pipeline_validation_helpers import (
    _get_adapter_allowlist,
    _is_unbounded_type,
    _is_wildcard_type,
)

if TYPE_CHECKING:  # pragma: no cover
    from .pipeline import Pipeline

_PipeInT = TypeVar("_PipeInT")
_PipeOutT = TypeVar("_PipeOutT")


def run_step_validations(
    pipeline: "Pipeline[_PipeInT, _PipeOutT]",
    report: ValidationReport,
    *,
    raise_on_error: bool,
) -> None:
    """Validate per-step agents, types, duplicate instances, and fallbacks."""
    from typing import get_origin, get_args, Union as TypingUnion
    import types as _types
    import re as _re

    def _compatible(a: object, b: object) -> bool:
        """Strict compatibility: no wildcard fallthrough, explicit bridges only."""
        if _is_unbounded_type(a) or _is_unbounded_type(b):
            return False

        origin_a, origin_b = get_origin(a), get_origin(b)
        _UnionType = getattr(_types, "UnionType", None)

        try:
            from pydantic import BaseModel as _PydanticBaseModel

            if isinstance(a, type) and issubclass(a, _PydanticBaseModel):
                if b is dict or origin_b is dict:
                    return True
        except Exception:
            pass

        if origin_b is TypingUnion or (_UnionType is not None and origin_b is _UnionType):
            return any(_compatible(a, arg) for arg in get_args(b))
        if origin_a is TypingUnion or (_UnionType is not None and origin_a is _UnionType):
            return all(_compatible(arg, b) for arg in get_args(a))

        try:
            b_eff = origin_b if origin_b is not None else b
            a_eff = origin_a if origin_a is not None else a
            if not isinstance(b_eff, type) or not isinstance(a_eff, type):
                return False
            return issubclass(a_eff, b_eff)
        except Exception as e:  # pragma: no cover
            logging.warning("_compatible: issubclass(%s, %s) raised %s", a, b, e)
            return False

    seen_steps: set[int] = set()

    def _root_key(key: str) -> str:
        try:
            return key.split(".", 1)[0].strip()
        except Exception:
            return key

    _BASE_INITIAL_ROOTS: frozenset[str] = frozenset(
        {
            "initial_prompt",
            "run_id",
            "hitl_history",
            "command_log",
            "conversation_history",
            "steps",
            "call_count",
        }
    )

    try:
        from .conditional import ConditionalStep as _ConditionalStep
    except Exception:
        _ConditionalStep = None  # type: ignore[misc,assignment]
    try:
        from .parallel import ParallelStep as _ParallelStep
    except Exception:
        _ParallelStep = None  # type: ignore[misc,assignment]
    try:
        from .import_step import ImportStep as _ImportStep
    except Exception:
        _ImportStep = None  # type: ignore[misc,assignment]

    try:
        from ...infra.settings import get_settings as _get_settings

        strict_mode = bool(getattr(_get_settings(), "strict_dsl", True))
    except Exception:
        strict_mode = True

    adapter_allowlist = _get_adapter_allowlist()

    from ...utils.scratchpad import is_merge_scratchpad, is_scratchpad_path

    def _validate_pipeline(
        current: "Pipeline[_PipeInT, _PipeOutT]",
        available_roots: set[str],
        produced_paths: set[str],
        prev_step: object | None,
        prev_out_type: object | None,
    ) -> set[str]:
        for idx_step, step in enumerate(getattr(current, "steps", []) or []):
            meta = getattr(step, "meta", None)
            _yloc = meta.get("_yaml_loc") if isinstance(meta, dict) else None
            templated_input = meta.get("templated_input") if isinstance(meta, dict) else None
            is_adapter_step = bool(meta.get("is_adapter")) if isinstance(meta, dict) else False
            if is_adapter_step:
                adapter_id = meta.get("adapter_id") if isinstance(meta, dict) else None
                adapter_allow = meta.get("adapter_allow") if isinstance(meta, dict) else None
                adapter_id_str = str(adapter_id).strip() if isinstance(adapter_id, str) else ""
                adapter_allow_str = (
                    str(adapter_allow).strip() if isinstance(adapter_allow, str) else ""
                )
                # Only check for missing metadata here; allowlist validation happens later
                if not adapter_id_str or not adapter_allow_str:
                    report.errors.append(
                        ValidationFinding(
                            rule_id="V-ADAPT-META",
                            severity="error",
                            message=(
                                f"Adapter step '{getattr(step, 'name', '')}' must declare adapter_id and "
                                "adapter_allow (allowlist token)."
                            ),
                            step_name=getattr(step, "name", None),
                        )
                    )
            if id(step) in seen_steps:
                report.warnings.append(
                    ValidationFinding(
                        rule_id="V-A3",
                        severity="warning",
                        message=(
                            "The same Step object instance is used more than once in the pipeline. "
                            "This may cause side effects if the step is stateful."
                        ),
                        step_name=step.name,
                    )
                )
            else:
                seen_steps.add(id(step))

            if (not getattr(step, "is_complex", False)) and step.agent is None:
                report.errors.append(
                    ValidationFinding(
                        rule_id="V-A1",
                        severity="error",
                        message=(
                            "Step '{name}' is missing an agent. Assign one via `Step('name', agent=...)` "
                            "or by using a step factory like `@step` or `Step.from_callable()`."
                        ).format(name=step.name),
                        step_name=step.name,
                    )
                )
            else:
                target = getattr(step.agent, "_agent", step.agent)
                func = getattr(target, "_step_callable", getattr(target, "run", None))
                if func is not None:
                    try:
                        from ...signature_tools import (
                            analyze_signature,
                        )  # Local import to avoid cycles

                        analyze_signature(func)
                    except Exception as e:  # pragma: no cover - defensive
                        report.warnings.append(
                            ValidationFinding(
                                rule_id="V-A4-ERR",
                                severity="warning",
                                message=(
                                    f"Could not analyze signature for agent in step '{step.name}': {e}"
                                ),
                                step_name=step.name,
                            )
                        )

            if _ConditionalStep is not None and isinstance(step, _ConditionalStep):
                branch_paths_sets: list[set[str]] = []
                branches = getattr(step, "branches", {}) or {}
                for branch in branches.values():
                    try:
                        child_paths = _validate_pipeline(
                            branch, set(available_roots), set(produced_paths), None, None
                        )
                        branch_paths_sets.append(set(child_paths))
                    except Exception:
                        branch_paths_sets.append(set())
                        continue
                default_branch = getattr(step, "default_branch_pipeline", None)
                if default_branch is not None:
                    try:
                        child_paths = _validate_pipeline(
                            default_branch, set(available_roots), set(produced_paths), None, None
                        )
                        branch_paths_sets.append(set(child_paths))
                    except Exception:
                        branch_paths_sets.append(set())

                if not branch_paths_sets:
                    guaranteed_paths: set[str] = set()
                elif len(branch_paths_sets) == 1:
                    guaranteed_paths = branch_paths_sets[0]
                else:
                    guaranteed_paths = set.intersection(*branch_paths_sets)

                produced_paths.update(guaranteed_paths)
                available_roots.update(_root_key(p) for p in guaranteed_paths)
                prev_step = step
                prev_out_type = getattr(step, "__step_output_type__", object)
                continue

            if _ParallelStep is not None and isinstance(step, _ParallelStep):
                try:
                    from .step import MergeStrategy as _MergeStrategy  # local import
                except Exception:
                    _MergeStrategy = None  # type: ignore[misc,assignment]

                merge_strategy = getattr(step, "merge_strategy", None)

                if is_merge_scratchpad(merge_strategy):
                    report.errors.append(
                        ValidationFinding(
                            rule_id="V-P-SCRATCHPAD",
                            severity="error",
                            message=(
                                f"Parallel step '{step.name}' uses forbidden merge strategy targeting 'scratchpad'."
                            ),
                            step_name=getattr(step, "name", None),
                        )
                    )

                if (
                    _MergeStrategy is not None
                    and (
                        merge_strategy == _MergeStrategy.CONTEXT_UPDATE
                        or (
                            isinstance(merge_strategy, str)
                            and merge_strategy.lower() == _MergeStrategy.CONTEXT_UPDATE.value
                        )
                    )
                    and not bool(getattr(step, "ignore_branch_names", False))
                ):
                    fm = getattr(step, "field_mapping", None)
                    context_include = getattr(step, "context_include_keys", None) or []
                    if isinstance(fm, dict) and fm:
                        seen: set[str] = set()
                        dup: set[str] = set()
                        for dests in fm.values():
                            if not isinstance(dests, (list, tuple)):
                                continue
                            for d in dests:
                                key = str(d)
                                if key in seen:
                                    dup.add(key)
                                else:
                                    seen.add(key)
                        if dup or context_include:
                            report.errors.append(
                                ValidationFinding(
                                    rule_id="V-P1",
                                    severity="error",
                                    message=(
                                        f"Parallel step '{step.name}' merges overlapping keys via field_mapping: {sorted(dup)}."
                                        if dup
                                        else (
                                            "Parallel step uses context_include_keys with field_mapping "
                                            "but no destination keys provided."
                                        )
                                    ),
                                    step_name=getattr(step, "name", None),
                                )
                            )
                    else:
                        if context_include:
                            report.errors.append(
                                ValidationFinding(
                                    rule_id="V-P1",
                                    severity="error",
                                    message=(
                                        f"Parallel step '{step.name}' uses merge_strategy=CONTEXT_UPDATE with "
                                        "context_include_keys but no field_mapping; branches may conflict."
                                    ),
                                    step_name=getattr(step, "name", None),
                                )
                            )

                context_include = getattr(step, "context_include_keys", None) or []
                include_roots: set[str] = {
                    _root_key(str(k))
                    for k in context_include
                    if isinstance(k, str) and str(k).strip()
                }
                if include_roots:
                    branch_available_roots = set(_BASE_INITIAL_ROOTS) | include_roots
                    branch_produced_paths = {
                        p for p in produced_paths if _root_key(p) in include_roots
                    }
                else:
                    branch_available_roots = set(available_roots)
                    branch_produced_paths = set(produced_paths)

                branch_outputs_by_name: dict[str, set[str]] = {}
                branches = getattr(step, "branches", {}) or {}
                for branch_name, branch in branches.items():
                    branch_key = str(branch_name)
                    try:
                        child_paths = _validate_pipeline(
                            branch,
                            set(branch_available_roots),
                            set(branch_produced_paths),
                            None,
                            None,
                        )
                        branch_outputs_by_name[branch_key] = set(child_paths)
                    except Exception:
                        branch_outputs_by_name[branch_key] = set()
                        continue

                normalized_ms: str | None = None
                if _MergeStrategy is not None and isinstance(merge_strategy, _MergeStrategy):
                    normalized_ms = str(getattr(merge_strategy, "value", "")).strip().lower()
                elif isinstance(merge_strategy, str):
                    normalized_ms = merge_strategy.strip().lower()

                merged_paths: set[str] = set()
                if normalized_ms == "no_merge":
                    merged_paths = set()
                elif normalized_ms == "context_update":
                    fm = getattr(step, "field_mapping", None)
                    if isinstance(fm, dict) and fm:
                        for bn, bpaths in branch_outputs_by_name.items():
                            if bn in fm:
                                dests = fm.get(bn)
                                if isinstance(dests, (list, tuple)):
                                    for d in dests:
                                        key = str(d).strip()
                                        if key:
                                            merged_paths.add(key)
                                else:
                                    # Unknown schema: fall back to pessimistic behavior for this branch.
                                    pass
                            else:
                                merged_paths.update(bpaths)
                    else:
                        for bpaths in branch_outputs_by_name.values():
                            merged_paths.update(bpaths)
                elif normalized_ms == "overwrite":
                    # Conservative: only propagate keys we can prove are copied back.
                    # With context_include_keys this is the included subset; otherwise treat as opaque.
                    if include_roots and branch_outputs_by_name:
                        try:
                            last = sorted(branch_outputs_by_name.keys())[-1]
                        except Exception:
                            last = None
                        if last is not None:
                            merged_paths.update(
                                p
                                for p in branch_outputs_by_name.get(last, set())
                                if _root_key(p) in include_roots
                            )
                elif callable(merge_strategy):
                    report.warnings.append(
                        ValidationFinding(
                            rule_id="V-P-MERGE",
                            severity="warning",
                            message=(
                                f"Parallel step '{step.name}' uses a callable merge_strategy; "
                                "static context key propagation cannot be validated reliably."
                            ),
                            step_name=getattr(step, "name", None),
                            suggestion=(
                                "Prefer MergeStrategy.CONTEXT_UPDATE with field_mapping to make merges explicit."
                            ),
                        )
                    )
                else:
                    # Default: assume a merge-like strategy that can propagate branch-produced keys.
                    for bpaths in branch_outputs_by_name.values():
                        merged_paths.update(bpaths)

                produced_paths.update(merged_paths)
                available_roots.update(_root_key(p) for p in merged_paths)
                prev_step = step
                prev_out_type = getattr(step, "__step_output_type__", object)
                continue

            if _ImportStep is not None and isinstance(step, _ImportStep):
                child = getattr(step, "pipeline", None)
                if child is not None:
                    try:
                        inherit = bool(getattr(step, "inherit_context", False))
                        child_available_roots = (
                            set(available_roots)
                            if inherit
                            else (set(_BASE_INITIAL_ROOTS) | {"import_artifacts"})
                        )
                        child_produced_paths = set(produced_paths) if inherit else set()

                        child_paths = _validate_pipeline(
                            child, child_available_roots, child_produced_paths, None, None
                        )

                        if bool(getattr(step, "updates_context", False)):
                            outs = getattr(step, "outputs", None)
                            if outs is None:
                                produced_paths.update(child_paths)
                                available_roots.update(_root_key(p) for p in child_paths)
                            elif isinstance(outs, list):
                                if outs:
                                    for om in outs:
                                        try:
                                            parent_path = str(getattr(om, "parent", "")).strip()
                                            child_path = str(getattr(om, "child", "")).strip()
                                        except Exception:
                                            parent_path = ""
                                            child_path = ""
                                        for path in (parent_path, child_path):
                                            if path:
                                                produced_paths.add(path)
                                                available_roots.add(_root_key(path))
                    except Exception:
                        pass

            in_type = getattr(step, "__step_input_type__", object)
            templated_input_present = False
            try:
                meta = getattr(step, "meta", None)
                if isinstance(meta, dict) and meta.get("templated_input") is not None:
                    templated_input_present = True
            except Exception:
                templated_input_present = False
            if prev_step is not None and prev_out_type is not None:
                if (
                    not templated_input_present
                    and not is_adapter_step
                    and not _compatible(prev_out_type, in_type)
                ):
                    prev_name = str(getattr(prev_step, "name", ""))
                    report.errors.append(
                        ValidationFinding(
                            rule_id="V-A2",
                            severity="error",
                            message=(
                                f"Type mismatch: Output of '{prev_name}' (returns `{prev_out_type}`) "
                                f"is not compatible with '{step.name}' (expects `{in_type}`). "
                                "For best results, use a static type checker like mypy to catch these issues before runtime."
                            ),
                            step_name=step.name,
                        )
                    )

            required_keys = [
                k for k in getattr(step, "input_keys", []) if isinstance(k, str) and k.strip()
            ]
            missing_keys: list[str] = []
            weak_keys: list[str] = []
            for rk in required_keys:
                root = _root_key(rk)
                if rk in produced_paths:
                    continue
                if root in available_roots:
                    weak_keys.append(rk)
                    continue
                missing_keys.append(rk)

            if missing_keys:
                report.errors.append(
                    ValidationFinding(
                        rule_id="V-CTX1",
                        severity="error",
                        message=(
                            f"Step '{step.name}' requires context keys {missing_keys} "
                            "that are not produced earlier in the pipeline."
                        ),
                        step_name=step.name,
                    )
                )
            if weak_keys:
                report.warnings.append(
                    ValidationFinding(
                        rule_id="V-CTX2",
                        severity="warning",
                        message=(
                            f"Step '{step.name}' requires context paths {weak_keys} but only their root keys "
                            "are available. Declare precise output_keys (e.g., 'import_artifacts.field' or other typed fields) in producer steps."
                        ),
                        step_name=step.name,
                    )
                )

            def _strict_types_match(src: object, dst: object, *, is_adapter: bool) -> bool:
                """Strict type compatibility: disallow wildcard fallthrough and dict-to-object bypass.

                Pydantic->dict bridging is only allowed when step is an adapter.
                """
                if _is_unbounded_type(src) or _is_unbounded_type(dst):
                    return False
                origin_s, origin_d = get_origin(src), get_origin(dst)
                try:
                    from pydantic import BaseModel as _PydanticBaseModel

                    if isinstance(src, type) and issubclass(src, _PydanticBaseModel):
                        # Allow Pydantic model outputs to flow into dict expectations only via adapters.
                        if dst is dict or origin_d is dict:
                            return is_adapter
                except Exception:
                    pass
                if origin_d is typing.Union:
                    return any(
                        _strict_types_match(src, arg, is_adapter=is_adapter)
                        for arg in get_args(dst)
                    )
                if origin_s is typing.Union:
                    return all(
                        _strict_types_match(arg, dst, is_adapter=is_adapter)
                        for arg in get_args(src)
                    )
                src_eff = origin_s if origin_s is not None else src
                dst_eff = origin_d if origin_d is not None else dst
                if not isinstance(src_eff, type) or not isinstance(dst_eff, type):
                    return False
                try:
                    return issubclass(src_eff, dst_eff)
                except Exception:
                    return False

            def _generate_type_error_suggestion(
                step_name: str,
                prev_output_type: object,
                curr_input_type: object,
                is_wildcard: bool,
            ) -> str:
                """Generate actionable suggestions for type mismatch errors."""
                # Check if step was created from a callable (has type info)
                step_has_callable = hasattr(step, "__step_input_type__")

                # Try to get a readable type name
                def _type_name(t: object) -> str:
                    if t is object:
                        return "object"
                    if t is str:
                        return "str"
                    if t is dict:
                        return "dict"
                    if t is list:
                        return "list"
                    origin = get_origin(t)
                    if origin is not None:
                        args = get_args(t)
                        if origin is typing.Union:
                            return f"Union[{', '.join(_type_name(a) for a in args)}]"
                        return str(t)
                    return str(t)

                prev_type_str = _type_name(prev_output_type)
                curr_type_str = _type_name(curr_input_type)

                suggestions: list[str] = []

                if is_wildcard:
                    # V-A2-STRICT: Generic type (object/Generic) detected
                    suggestions.append(
                        f"WHY: Strict mode forbids '{curr_type_str}' to prevent silent data loss. "
                        "Using generic types can cause steps to receive unexpected data types at runtime."
                    )
                    suggestions.append("")
                    suggestions.append("FIX OPTIONS:")
                    suggestions.append("")
                    suggestions.append("1. Narrow your type hint (RECOMMENDED):")
                    if step_has_callable:
                        suggestions.append(
                            f"   Change your function signature from:\n"
                            f"   async def {step_name}(data: object, ...):\n"
                            f"   To:\n"
                            f"   async def {step_name}(data: {prev_type_str}, ...):"
                        )
                    else:
                        suggestions.append(
                            f"   Set the step's input type explicitly:\n"
                            f"   step_instance.__step_input_type__ = {prev_type_str}"
                        )
                    suggestions.append("")
                    suggestions.append(
                        "2. Use the @adapter_step decorator (if this is a data bridge):"
                    )
                    suggestions.append(
                        f"   @adapter_step(\n"
                        f'       name="{step_name}",\n'
                        f'       adapter_id="your-adapter-id",\n'
                        f'       adapter_allow="your-token"\n'
                        f"   )\n"
                        f"   async def {step_name}(data: {curr_type_str}, ...):\n"
                        f"       # Transform data here\n"
                        f"       return transformed_data"
                    )
                    suggestions.append("")
                    suggestions.append(
                        "   Note: Adapter steps require allowlist registration. "
                        "See docs/cookbook/adapter_step.md for details."
                    )
                else:
                    # V-A2-TYPE: Concrete type mismatch
                    suggestions.append(
                        f"WHY: Output type '{prev_type_str}' cannot be safely converted to input type '{curr_type_str}'. "
                        "This mismatch can cause runtime errors or data corruption."
                    )
                    suggestions.append("")
                    suggestions.append("FIX OPTIONS:")
                    suggestions.append("")
                    suggestions.append(
                        "1. Align the types (if the previous step's output is wrong):"
                    )
                    suggestions.append(
                        f"   Update the previous step to return '{curr_type_str}' instead of '{prev_type_str}'."
                    )
                    suggestions.append("")
                    suggestions.append("2. Use @adapter_step to bridge the types:")
                    try:
                        from pydantic import BaseModel as _PydanticBaseModel

                        is_pydantic_to_dict = (
                            isinstance(prev_output_type, type)
                            and issubclass(prev_output_type, _PydanticBaseModel)
                            and (curr_input_type is dict or get_origin(curr_input_type) is dict)
                        )
                        if is_pydantic_to_dict:
                            suggestions.append(
                                f"   This is a Pydantic model → dict conversion. Use an adapter:\n"
                                f"   @adapter_step(\n"
                                f'       name="{step_name}",\n'
                                f'       adapter_id="your-adapter-id",\n'
                                f'       adapter_allow="your-token"\n'
                                f"   )\n"
                                f"   async def {step_name}(data: {prev_type_str}, ...):\n"
                                f"       return data.model_dump()  # Convert Pydantic to dict"
                            )
                        else:
                            suggestions.append(
                                f"   @adapter_step(\n"
                                f'       name="{step_name}",\n'
                                f'       adapter_id="your-adapter-id",\n'
                                f'       adapter_allow="your-token"\n'
                                f"   )\n"
                                f"   async def {step_name}(data: {prev_type_str}, ...):\n"
                                f"       # Transform {prev_type_str} to {curr_type_str}\n"
                                f"       return transformed_data"
                            )
                    except Exception:
                        suggestions.append(
                            f"   @adapter_step(\n"
                            f'       name="{step_name}",\n'
                            f'       adapter_id="your-adapter-id",\n'
                            f'       adapter_allow="your-token"\n'
                            f"   )\n"
                            f"   async def {step_name}(data: {prev_type_str}, ...):\n"
                            f"       # Transform {prev_type_str} to {curr_type_str}\n"
                            f"       return transformed_data"
                        )
                    suggestions.append("")
                    suggestions.append(
                        "   Note: Adapter steps require allowlist registration. "
                        "See docs/cookbook/adapter_step.md for details."
                    )

                return "\n".join(suggestions)

            if is_adapter_step:
                adapter_id = meta.get("adapter_id") if isinstance(meta, dict) else None
                adapter_token = meta.get("adapter_allow") if isinstance(meta, dict) else None
                if not adapter_id or adapter_id not in adapter_allowlist:
                    report.errors.append(
                        ValidationFinding(
                            rule_id="V-ADAPT-ALLOW",
                            severity="error",
                            message=(
                                f"Adapter step '{getattr(step, 'name', '')}' lacks an allowlisted adapter_id."
                            ),
                            step_name=getattr(step, "name", None),
                        )
                    )
                elif adapter_allowlist.get(adapter_id) != adapter_token:
                    report.errors.append(
                        ValidationFinding(
                            rule_id="V-ADAPT-ALLOW",
                            severity="error",
                            message=(
                                f"Adapter step '{getattr(step, 'name', '')}' missing correct adapter token "
                                f"(expected '{adapter_allowlist.get(adapter_id)}')."
                            ),
                            step_name=getattr(step, "name", None),
                        )
                    )

            if prev_step is not None:
                prev_updates_context = bool(getattr(prev_step, "updates_context", False))
                curr_accepts_input = getattr(step, "__step_input_type__", object)
                prev_produces_output = getattr(prev_step, "__step_output_type__", object)

                def _templated_input_consumes_prev(_step: object, prev_name: str) -> bool:
                    try:
                        meta2 = getattr(_step, "meta", None)
                        templ = meta2.get("templated_input") if isinstance(meta2, dict) else None
                        if not isinstance(templ, str):
                            return False
                        if "{{" not in templ or "}}" not in templ:
                            return False
                        prev_esc = _re.escape(str(prev_name)) if prev_name else ""
                        for m in _re.finditer(r"\{\{(.*?)\}\}", templ, flags=_re.DOTALL):
                            expr = m.group(1)
                            if not isinstance(expr, str):
                                continue
                            if _re.search(r"\bprevious_step\b", expr):
                                return True
                            if prev_esc:
                                pat1 = rf"\bsteps\s*\.\s*{prev_esc}\b"
                                pat2 = rf"\bsteps\s*\[\s*['\"]{prev_esc}['\"]\s*\]"
                                if _re.search(pat1, expr) or _re.search(pat2, expr):
                                    return True
                        return False
                    except Exception:
                        return False

                curr_generic = _is_unbounded_type(curr_accepts_input)
                meta2 = getattr(step, "meta", None)
                templated_input = meta2.get("templated_input") if isinstance(meta2, dict) else None
                has_explicit_templated_input = templated_input is not None
                if (
                    (not prev_updates_context)
                    and (prev_produces_output is not None)
                    and (curr_generic or has_explicit_templated_input)
                ):
                    prev_name = str(getattr(prev_step, "name", ""))
                    if not _templated_input_consumes_prev(step, prev_name):
                        report.warnings.append(
                            ValidationFinding(
                                rule_id="V-A5",
                                severity="warning",
                                message=(
                                    f"The output of step '{prev_name}' is not used by '{step.name}'."
                                ),
                                step_name=prev_name,
                                suggestion=(
                                    "Set updates_context=True on the producing step or insert an adapter step to consume its output."
                                ),
                            )
                        )

                # Disallow implicit wildcard bridging without explicit adapter
                if _is_wildcard_type(curr_accepts_input) and prev_produces_output is not None:
                    if not is_adapter_step:
                        prev_name = str(getattr(prev_step, "name", ""))
                        suggestion = _generate_type_error_suggestion(
                            step_name=step.name,
                            prev_output_type=prev_produces_output,
                            curr_input_type=curr_accepts_input,
                            is_wildcard=True,
                        )
                        report.errors.append(
                            ValidationFinding(
                                rule_id="V-A2-STRICT",
                                severity="error",
                                message=(
                                    f"Step '{step.name}' accepts '{curr_accepts_input}' which is too generic "
                                    f"for upstream output '{prev_name}' ({prev_produces_output})."
                                ),
                                step_name=getattr(step, "name", None),
                                suggestion=suggestion,
                            )
                        )

                # Fail on concrete type mismatches in strict mode (non-generic, non-adapter).
                if (
                    strict_mode
                    and prev_produces_output is not None
                    and curr_accepts_input is not None
                    and not is_adapter_step
                    and not curr_generic
                ):
                    if not _strict_types_match(
                        prev_produces_output, curr_accepts_input, is_adapter=is_adapter_step
                    ):
                        prev_name = str(getattr(prev_step, "name", ""))
                        suggestion = _generate_type_error_suggestion(
                            step_name=step.name,
                            prev_output_type=prev_produces_output,
                            curr_input_type=curr_accepts_input,
                            is_wildcard=False,
                        )
                        report.errors.append(
                            ValidationFinding(
                                rule_id="V-A2-TYPE",
                                severity="error",
                                message=(
                                    f"Type mismatch: Output of '{prev_name}' "
                                    f"({prev_produces_output}) is not compatible with '{step.name}' "
                                    f"input ({curr_accepts_input})."
                                ),
                                step_name=getattr(step, "name", None),
                                suggestion=suggestion,
                            )
                        )

            fb = getattr(step, "fallback_step", None)
            if fb is not None:
                step_in = getattr(step, "__step_input_type__", object)
                fb_in = getattr(fb, "__step_input_type__", object)
                if not _compatible(step_in, fb_in):
                    report.errors.append(
                        ValidationFinding(
                            rule_id="V-F1",
                            severity="error",
                            message=(
                                f"Fallback step '{getattr(fb, 'name', 'unknown')}' expects input `{fb_in}`, "
                                f"which is not compatible with original step '{step.name}' input `{step_in}`."
                            ),
                            step_name=step.name,
                            suggestion=(
                                "Ensure the fallback step accepts the same input type as the original step or add an adapter."
                            ),
                        )
                    )

            produced_keys = [
                k for k in getattr(step, "output_keys", []) if isinstance(k, str) and k.strip()
            ]
            sink_target = getattr(step, "sink_to", None)
            if isinstance(sink_target, str) and sink_target.strip():
                produced_keys.append(sink_target)
            for pk in produced_keys:
                produced_paths.add(pk)
                root = _root_key(pk)
                if is_scratchpad_path(pk):
                    report.errors.append(
                        ValidationFinding(
                            rule_id=f"CTX-{root.upper()}",
                            severity="error",
                            message=(
                                f"Step '{step.name}' attempts to write to '{pk}' but 'scratchpad' is removed."
                            ),
                            step_name=getattr(step, "name", None),
                        )
                    )
                available_roots.add(root)

            if getattr(step, "updates_context", False) and not produced_keys:
                if (
                    _ImportStep is not None
                    and isinstance(step, _ImportStep)
                    and bool(getattr(step, "updates_context", False))
                ):
                    outs = getattr(step, "outputs", None)
                    if outs is None or (isinstance(outs, list) and bool(outs)):
                        prev_step = step
                        prev_out_type = getattr(step, "__step_output_type__", object)
                        continue
                report.errors.append(
                    ValidationFinding(
                        rule_id="CTX-OUTPUT-KEYS",
                        severity="error",
                        message=(
                            f"Step '{step.name}' sets updates_context=True but declares no output_keys/sink_to. "
                            "Declare typed context fields to persist outputs."
                        ),
                        step_name=getattr(step, "name", None),
                        location_path=_yloc.get("path")
                        if isinstance(_yloc, dict)
                        else f"steps[{idx_step}]",
                        file=_yloc.get("file") if isinstance(_yloc, dict) else None,
                        line=_yloc.get("line") if isinstance(_yloc, dict) else None,
                        column=_yloc.get("column") if isinstance(_yloc, dict) else None,
                    )
                )

            prev_step = step
            prev_out_type = getattr(step, "__step_output_type__", object)
        return produced_paths

    _validate_pipeline(pipeline, set(_BASE_INITIAL_ROOTS), set(), None, None)
