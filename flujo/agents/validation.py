from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Type, Any as _Any
from pydantic import BaseModel, ValidationError

from ..exceptions import AgentIOValidationError


def validated_agent(
    input_model: Type[BaseModel], output_model: Type[BaseModel]
) -> Callable[[Type[_Any]], Type[_Any]]:
    """Class decorator that validates the ``run`` method's input and output."""

    def decorator(agent_class: Type[_Any]) -> Type[_Any]:
        original_run = agent_class.run

        @wraps(original_run)
        async def validated_run(self: _Any, data: Any, **kwargs: Any) -> Any:
            try:
                validated_input = input_model.model_validate(data)
            except ValidationError as e:
                raise AgentIOValidationError(
                    f"Input validation failed for {agent_class.__name__}"
                ) from e

            raw_output = await original_run(self, validated_input, **kwargs)

            try:
                return output_model.model_validate(raw_output)
            except ValidationError as e:
                raise AgentIOValidationError(
                    f"Output validation failed for {agent_class.__name__}"
                ) from e

        agent_class.run = validated_run
        return agent_class

    return decorator
