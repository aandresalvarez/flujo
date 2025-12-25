from __future__ import annotations
from abc import abstractmethod
from typing import Protocol, Any, runtime_checkable, Optional, Callable, Tuple, TYPE_CHECKING
from typing import TypeVar, Literal
from flujo.domain.base_model import BaseModel
from pydantic import Field, model_validator
from flujo.type_definitions.common import JSONObject


_F = TypeVar("_F", bound=Callable[..., object])

if TYPE_CHECKING:  # pragma: no cover

    def _typed_model_validator(*, mode: Literal["before"]) -> Callable[[_F], _F]: ...

else:

    def _typed_model_validator(*, mode: Literal["before"]) -> Callable[[_F], _F]:
        return model_validator(mode=mode)


class ValidationResult(BaseModel):
    """The standard output from any validator, providing a pass/fail signal and scoring."""

    is_valid: bool
    score: float = 1.0
    diff: JSONObject | None = None
    feedback: Optional[str] = None
    validator_name: str
    metadata: JSONObject = Field(default_factory=dict)

    @_typed_model_validator(mode="before")
    @classmethod
    def _default_score_from_validity(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        if "score" in data and data["score"] is None:
            data = dict(data)
            data.pop("score", None)
        if "score" not in data and data.get("is_valid") is False:
            data = dict(data)
            data["score"] = 0.0
        return data


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


ValidatorReturn = (
    Tuple[bool, Optional[str]]
    | Tuple[bool, Optional[str], float]
    | Tuple[bool, Optional[str], float, JSONObject]
    | bool
)


def validator(func: Callable[..., ValidatorReturn]) -> Validator:
    """Decorator to create a stateless Validator from a function.

    This decorator allows you to easily convert a simple function into a
    Validator that can be used in validation pipelines. The function should
    take the output to check and return a tuple of (is_valid, feedback[, score, diff]).

    Args:
        func: A function that takes output_to_check and returns (bool, str|None[, score, diff])

    Returns:
        A Validator instance that wraps the function

    Example:
        @validator
        def contains_hello(output: str) -> tuple[bool, Optional[str]]:
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
                import inspect

                sig = inspect.signature(func)
                params = sig.parameters
                has_context = "context" in params
                context_param = params.get("context") if has_context else None

                # Decide how to pass context based on parameter kind
                if has_context and context_param is not None:
                    if context_param.kind in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    ):
                        result = func(output_to_check, context)
                    else:  # KEYWORD_ONLY
                        result = func(output_to_check, context=context)
                else:
                    result = func(output_to_check)

                # Support bool or tuple returns: (bool, feedback[, score, diff])
                if isinstance(result, tuple) and len(result) >= 1:
                    is_valid = bool(result[0])
                    feedback = result[1] if len(result) > 1 else None
                    score = result[2] if len(result) > 2 else None
                    diff = result[3] if len(result) > 3 else None
                else:
                    is_valid = bool(result)
                    feedback = None
                    score = None
                    diff = None

                payload: dict[str, object] = {
                    "is_valid": is_valid,
                    "feedback": feedback,
                    "validator_name": func.__name__,
                }
                if score is not None:
                    payload["score"] = score
                if diff is not None:
                    payload["diff"] = diff

                return ValidationResult(**payload)
            except Exception as e:  # pragma: no cover - defensive
                return ValidationResult(
                    is_valid=False,
                    feedback=f"Validator function raised an exception: {e}",
                    validator_name=func.__name__,
                )

    return FunctionalValidator(name=func.__name__)
