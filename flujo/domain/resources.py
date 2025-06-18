from flujo.domain.models import BaseModel
from pydantic import ConfigDict
from typing import ClassVar


class AppResources(BaseModel):
    """Base class for user-defined resource containers."""

    model_config: ClassVar[ConfigDict] = {"arbitrary_types_allowed": True}
