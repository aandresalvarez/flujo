from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel as PydanticBaseModel
from ..base_model import BaseModel as FlujoBaseModel

from ..dsl import Step
from .loader_models import BlueprintError


def _normalize_merge_strategy(value: Optional[str]) -> Any:
    from ..dsl.step import MergeStrategy as _MS

    if value is None:
        return _MS.CONTEXT_UPDATE
    if str(value).lower() == "merge_scratchpad":
        raise BlueprintError(
            "Merge strategy 'merge_scratchpad' is not allowed (scratchpad banned)."
        )
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
    """Best-effort static type assignment for pipeline validation."""
    try:
        from flujo.signature_tools import analyze_signature as _analyze
        import inspect as _inspect

        def _is_default_type(t: Any) -> bool:
            return t is object or str(t) == "typing.Any"

        def _unwrap_primitive_wrapper(t: Any) -> Any:
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
                    step_obj.__step_output_type__ = _unwrap_primitive_wrapper(out_t)
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


__all__ = [
    "_finalize_step_types",
    "_normalize_branch_failure",
    "_normalize_merge_strategy",
]
