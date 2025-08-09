"""Custom BaseModel for all flujo domain models with unified serialization."""

from typing import Any, ClassVar, Optional
from pydantic import BaseModel as PydanticBaseModel, ConfigDict


class BaseModel(PydanticBaseModel):
    """BaseModel for all flujo domain models with unified serialization.
    
    This model delegates all serialization logic to flujo.utils.serialization
    to maintain a single source of truth for serialization behavior.
    """

    model_config: ClassVar[ConfigDict] = {
        "arbitrary_types_allowed": True,
    }

    def model_dump(
        self, *, mode: str = "default", **kwargs: Any
    ) -> Any:
        """
        Unified model serialization using flujo.utils.serialization.
        
        This method delegates to the centralized serialization system to ensure
        consistent behavior across all domain models.
        
        Args:
            mode: Serialization mode, supports "default" and "cache" modes
            **kwargs: Additional arguments (preserved for Pydantic compatibility)
        
        Returns:
            Serialized representation of the model
        """
        from flujo.utils.serialization import safe_serialize
        
        # Delegate all serialization to the centralized utility.
        # We pass `self` to be serialized.
        return safe_serialize(self, mode=mode)

    def model_dump_json(self, **kwargs: Any) -> str:
        """
        Serialize model to JSON string using unified serialization.
        
        Args:
            **kwargs: Arguments passed to json.dumps
            
        Returns:
            JSON string representation of the model
        """
        from flujo.utils.serialization import serialize_to_json
        
        # Extract mode if present in kwargs, default to "default"
        mode = kwargs.pop("mode", "default")
        data = self.model_dump(mode=mode, **kwargs)
        return serialize_to_json(data, mode=mode, **kwargs)
