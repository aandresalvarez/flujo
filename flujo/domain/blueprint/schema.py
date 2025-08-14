from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel


class AgentModel(BaseModel):
    model: str
    system_prompt: str
    output_schema: Dict[str, Any]


__all__ = ["AgentModel"]
