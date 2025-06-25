from typing import Protocol, Any, runtime_checkable, Optional
from pydantic import BaseModel

class ValidationResult(BaseModel):
    """The standard output from any validator, providing a clear pass/fail signal and feedback."""

    is_valid: bool
    feedback: Optional[str] = None
    validator_name: str


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
