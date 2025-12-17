from __future__ import annotations

from typing import Optional
from pydantic import BaseModel as PydanticBaseModel
from ..base_model import BaseModel as FlujoBaseModel

from ..dsl import Step
from ..dsl.step import BranchFailureStrategy, MergeStrategy
from .loader_models import BlueprintError
from ...utils.scratchpad import is_merge_scratchpad


def _normalize_merge_strategy(value: Optional[str]) -> MergeStrategy:
    if value is None:
        return MergeStrategy.CONTEXT_UPDATE
    if is_merge_scratchpad(value):
        raise BlueprintError(
            "Merge strategy 'merge_scratchpad' is not allowed (scratchpad banned)."
        )
    try:
        return MergeStrategy[value.upper()]
    except Exception as e:
        raise BlueprintError(f"Invalid merge_strategy: {value}") from e


def _normalize_branch_failure(value: Optional[str]) -> BranchFailureStrategy:
    if value is None:
        return BranchFailureStrategy.PROPAGATE
    try:
        return BranchFailureStrategy[value.upper()]
    except Exception as e:
        raise BlueprintError(f"Invalid on_branch_failure: {value}") from e


def _finalize_step_types(step_obj: Step[object, object]) -> None:
    """Best-effort static type assignment for pipeline validation."""
    try:
        from flujo.signature_tools import analyze_signature as _analyze
        import inspect as _inspect

        def _as_type(candidate: object) -> type[object] | None:
            return candidate if isinstance(candidate, type) else None

        def _is_default_type(t: object) -> bool:
            return t is object or not isinstance(t, type)

        def _unwrap_primitive_wrapper(t: object) -> object:
            try:
                if (
                    isinstance(t, type)
                    and issubclass(t, PydanticBaseModel)
                    and issubclass(t, FlujoBaseModel)
                ):
                    fields = getattr(t, "model_fields", {})
                    if len(fields) == 1 and "value" in fields:
                        fld = fields["value"]
                        ann = getattr(fld, "annotation", None)
                        outer = getattr(fld, "outer_type_", None)
                        return ann or outer or t
            except Exception:
                return t
            return t

        agent_obj = getattr(step_obj, "agent", None)
        try:
            if _is_default_type(getattr(step_obj, "__step_output_type__", object)) and hasattr(
                agent_obj, "target_output_type"
            ):
                out_t = getattr(agent_obj, "target_output_type")
                if out_t is not None:
                    resolved = _as_type(_unwrap_primitive_wrapper(out_t))
                    if resolved is not None:
                        step_obj.__step_output_type__ = resolved
        except Exception:
            pass
        fn = getattr(agent_obj, "_step_callable", None)
        if hasattr(fn, "__func__"):
            try:
                fn = getattr(fn, "__func__")
            except Exception:
                pass
        if fn is None:
            try:
                if _inspect.isfunction(agent_obj) or _inspect.ismethod(agent_obj):
                    fn = agent_obj
            except Exception:
                fn = None

        if fn is not None:
            sig = _analyze(fn)
            try:
                if _is_default_type(getattr(step_obj, "__step_input_type__", object)):
                    resolved = _as_type(getattr(sig, "input_type", object))
                    if resolved is not None:
                        step_obj.__step_input_type__ = resolved
            except Exception:
                pass
            try:
                if _is_default_type(getattr(step_obj, "__step_output_type__", object)):
                    resolved = _as_type(getattr(sig, "output_type", object))
                    if resolved is not None:
                        step_obj.__step_output_type__ = resolved
            except Exception:
                pass
    except Exception:
        pass


__all__ = [
    "_finalize_step_types",
    "_normalize_branch_failure",
    "_normalize_merge_strategy",
]
