from abc import abstractmethod
from typing import Protocol, Any, runtime_checkable, Optional, Dict, Callable, Tuple
from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    """The standard output from any validator, providing a clear pass/fail signal and feedback."""

    is_valid: bool
    feedback: Optional[str] = None
    validator_name: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class Validator(Protocol):
    """A generic, stateful protocol for any component that can validate a step's output."""

    name: str

    async def validate(
        self,
        output_to_check: Any,
        *,
        context: Optional[BaseModel] = None,
    ) -> ValidationResult:
        """Validates the given output."""
        ...


class BaseValidator(Validator):
    """A helpful base class for creating validators.

    This class provides a concrete implementation of the Validator protocol,
    making it easy to create custom validators by subclassing and implementing
    the validate method.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or self.__class__.__name__

    @abstractmethod
    async def validate(
        self,
        output_to_check: Any,
        *,
        context: Optional[BaseModel] = None,
    ) -> ValidationResult: ...


def validator(func: Callable[[Any], Tuple[bool, Optional[str]]]) -> Validator:
    """Decorator to create a stateless Validator from a function.

    This decorator allows you to easily convert a simple function into a
    Validator that can be used in validation pipelines. The function should
    take the output to check and return a tuple of (is_valid, feedback).

    Args:
        func: A function that takes output_to_check and returns (bool, str|None)

    Returns:
        A Validator instance that wraps the function

    Example:
        @validator
        def contains_hello(output: str) -> tuple[bool, str | None]:
            if "hello" in output.lower():
                return True, "Contains 'hello'"
            else:
                return False, "Does not contain 'hello'"
    """

    class FunctionalValidator(BaseValidator):
        async def validate(
            self,
            output_to_check: Any,
            *,
            context: Optional[BaseModel] = None,
        ) -> ValidationResult:
            try:
                is_valid, feedback = func(output_to_check)
                return ValidationResult(
                    is_valid=is_valid,
                    feedback=feedback,
                    validator_name=func.__name__,
                )
            except Exception as e:  # pragma: no cover - defensive
                return ValidationResult(
                    is_valid=False,
                    feedback=f"Validator function raised an exception: {e}",
                    validator_name=func.__name__,
                )

    return FunctionalValidator(name=func.__name__)
