from abc import abstractmethod
from typing import Any, Optional
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
    ) -> ValidationResult:
        ...
