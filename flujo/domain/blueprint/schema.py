from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel


class AgentModel(BaseModel):
    model: str
    system_prompt: str
    output_schema: Dict[str, Any]
    # Optional provider-specific controls (e.g., GPT-5: reasoning, text verbosity)
    model_settings: Optional[Dict[str, Any]] = None
    # Optional execution controls
    timeout: Optional[int] = None
    max_retries: Optional[int] = None


__all__ = ["AgentModel"]
