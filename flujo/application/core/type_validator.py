"""Type validation for step-to-step data flow."""

from __future__ import annotations

from typing import Any, Type, TypeVar, get_args, get_origin, Union

from ...domain.dsl.step import Step
from ...exceptions import TypeMismatchError
from ..context_manager import _types_compatible

T = TypeVar("T")


class TypeValidator:
    """Validates type compatibility between pipeline steps."""

    @staticmethod
    def validate_step_output(
        step: Step[Any, Any],
        step_result: Any,
        next_step: Step[Any, Any] | None,
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

        expected = getattr(next_step, "__step_input_type__", Any)
        actual_type = type(step_result)

        # Only allow None if the expected type is compatible with None
        if step_result is None:
            import types

            origin = get_origin(expected)
            if origin is Union:
                if type(None) in get_args(expected):
                    return
            elif hasattr(types, "UnionType") and isinstance(expected, types.UnionType):
                if type(None) in expected.__args__:
                    return
            if expected is Any:
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
    def get_step_input_type(step: Step[Any, Any]) -> Type[Any]:
        """Get the expected input type for a step."""
        return getattr(step, "__step_input_type__", Any)

    @staticmethod
    def get_step_output_type(step: Step[Any, Any]) -> Type[Any]:
        """Get the output type for a step."""
        return getattr(step, "__step_output_type__", Any)
