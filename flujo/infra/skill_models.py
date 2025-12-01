from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from flujo.type_definitions.common import JSONObject


@dataclass
class SkillRegistration:
    """Typed representation of a skill/agent registration entry."""

    id: str
    factory: Callable[..., Any] | Any
    description: Optional[str] = None
    input_schema: Optional[JSONObject] = None
    output_schema: Optional[JSONObject] = None
    capabilities: Optional[list[str]] = None
    safety_level: Optional[str] = None
    auth_required: Optional[bool] = None
    auth_scope: Optional[str] = None
    side_effects: Optional[bool] = None
    arg_schema: Optional[JSONObject] = None
    version: str | None = None
    scope: str | None = None
