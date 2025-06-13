from __future__ import annotations

from typing import Protocol, runtime_checkable, Any
from pydantic import BaseModel


class PluginOutcome(BaseModel):
    """Result returned by a validation plugin."""

    success: bool
    feedback: str | None = None


@runtime_checkable
class ValidationPlugin(Protocol):
    """Protocol that all validation plugins must implement."""

    async def validate(self, data: dict[str, Any]) -> PluginOutcome:  # pragma: no cover - signature only
        ...
