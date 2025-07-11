from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel


def flujo_default_serializer(obj: Any) -> Any:
    """Serialize common objects for ``orjson`` dumping."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError


__all__ = ["flujo_default_serializer"]
