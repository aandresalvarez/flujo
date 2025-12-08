"""Type validation for step-to-step data flow."""

from __future__ import annotations

from typing import Any, Type, TypeVar, get_args, get_origin, Union, TYPE_CHECKING, cast

from ...exceptions import TypeMismatchError
from .context_manager import _types_compatible

if TYPE_CHECKING:
    from flujo.domain.dsl.step import Step as DSLStep

T = TypeVar("T")


def _is_any_type(obj: object) -> bool:
    try:
        return obj is Any or getattr(obj, "__name__", None) == "Any"
    except Exception:
        return False


class TypeValidator:
    """Validates type compatibility between pipeline steps."""

    @staticmethod
    def validate_step_output(
        step: "DSLStep[Any, Any]",
        step_result: Any,
        next_step: "DSLStep[Any, Any] | None",
    ) -> None:
        """Validate that step output is compatible with next step's expected input.

        Args:
            step: The step that produced the output
            step_result: The output from the step
            next_step: The next step in the pipeline (if any)

        Raises:
            TypeMismatchError: If types are incompatible
        """
        if next_step is None:
            return

        expected = TypeValidator.get_step_input_type(next_step)
        is_background = (
            getattr(getattr(step, "config", None), "execution_mode", None) == "background"
        )

        if is_background:
            # Background steps pass their input through to downstream steps
            actual_type = TypeValidator.get_step_input_type(step)
        else:
            actual_type = type(step_result)

        # Only allow None if the expected type is compatible with None
        if step_result is None and not is_background:
            import types

            origin = get_origin(expected)
            if origin is Union:
                if type(None) in get_args(expected):
                    return
            elif hasattr(types, "UnionType") and isinstance(expected, types.UnionType):
                if type(None) in expected.__args__:
                    return
            if _is_any_type(expected):
                return
            raise TypeMismatchError(
                f"Type mismatch: Output of '{step.name}' was None, but '{next_step.name}' expects '{expected}'. "
                "For best results, use a static type checker like mypy to catch these issues before runtime."
            )

        if not _types_compatible(actual_type, expected):
            raise TypeMismatchError(
                f"Type mismatch: Output of '{step.name}' (returns `{actual_type}`) "
                f"is not compatible with '{next_step.name}' (expects `{expected}`). "
                "For best results, use a static type checker like mypy to catch these issues before runtime."
            )

    @staticmethod
    def get_step_input_type(step: "DSLStep[Any, Any]") -> Type[Any]:
        """Get the expected input type for a step."""
        candidate = getattr(step, "__step_input_type__", Any)
        if isinstance(candidate, type):
            return candidate
        return cast(Type[Any], Any)

    @staticmethod
    def get_step_output_type(step: "DSLStep[Any, Any]") -> Type[Any]:
        """Get the output type for a step."""
        candidate = getattr(step, "__step_output_type__", Any)
        if isinstance(candidate, type):
            return candidate
        return cast(Type[Any], Any)
