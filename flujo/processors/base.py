from __future__ import annotations

from typing import Protocol, Any, Optional

from ..domain.models import BaseModel


class Processor(Protocol):
    """Generic processor interface."""

    name: str

    async def process(self, data: Any, context: Optional[BaseModel] = None) -> Any:
        """Process data with optional pipeline context."""
        ...
