from __future__ import annotations

from typing import Any, Optional, Type

from pydantic import ValidationError
from pydantic import BaseModel as PydanticBaseModel

from ...infra import telemetry
from ...domain.models import BaseModel

__all__ = ["_build_context_update", "_inject_context"]


def _build_context_update(output: Any) -> dict[str, Any] | None:
    """Return context update dict extracted from a step output."""
    if isinstance(output, (BaseModel, PydanticBaseModel)):
        return output.model_dump(exclude_unset=True)
    if isinstance(output, dict):
        return output
    return None


def _inject_context(
    context: BaseModel,
    update_data: dict[str, Any],
    context_model: Type[BaseModel],
) -> Optional[str]:
    """Apply ``update_data`` to ``context`` validating against ``context_model``.

    Returns an error message if validation fails, otherwise ``None``.
    """
    original = context.model_dump()
    for key, value in update_data.items():
        setattr(context, key, value)
    try:
        validated = context_model.model_validate(context.model_dump())
        context.__dict__.update(validated.__dict__)
    except ValidationError as e:
        for key, value in original.items():
            setattr(context, key, value)
        telemetry.logfire.error(f"Context update failed Pydantic validation: {e}")
        return str(e)
    return None
