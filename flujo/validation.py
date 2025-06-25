from abc import abstractmethod
from typing import Any, Optional, Callable, Tuple
from pydantic import BaseModel

from .domain.validation import Validator, ValidationResult


class BaseValidator(Validator):
    """A helpful base class for creating validators."""

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
    """Decorator to create a stateless Validator from a function."""

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
