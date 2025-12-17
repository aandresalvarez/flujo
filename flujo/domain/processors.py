from __future__ import annotations

from typing import List, Any, TYPE_CHECKING

from pydantic import Field

from .models import BaseModel

if TYPE_CHECKING:  # pragma: no cover
    from ..processors.base import Processor  # noqa: F401


class AgentProcessors(BaseModel):
    """Collections of prompt and output processors."""

    # model_config inherited from BaseModel

    prompt_processors: List[Any] = Field(default_factory=list)
    output_processors: List[Any] = Field(default_factory=list)
    # Optional validators inside processors container (used by integration tests)
    validators: List[Any] = Field(default_factory=list)
