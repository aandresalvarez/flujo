from pydantic import BaseModel


class AppResources(BaseModel):
    """Base class for user-defined resource containers."""

    model_config = {"arbitrary_types_allowed": True}
