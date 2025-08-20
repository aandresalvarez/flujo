from __future__ import annotations

from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, model_validator


class AgentModel(BaseModel):
    model: str

    # FSD-022: Allow externalized prompt via { from_file: "path" }
    class FromFile(BaseModel):
        from_file: str

    system_prompt: Union[str, "AgentModel.FromFile"]
    output_schema: Dict[str, Any]
    # Optional provider-specific controls (e.g., GPT-5: reasoning, text verbosity)
    model_settings: Optional[Dict[str, Any]] = None
    # Optional execution controls
    timeout: Optional[int] = None
    max_retries: Optional[int] = None

    # Validate 'system_prompt' dict form contains only 'from_file'
    @model_validator(mode="before")
    @classmethod
    def _validate_prompt_format(cls, data: Any) -> Any:
        try:
            if isinstance(data, dict):
                prompt = data.get("system_prompt")
                if isinstance(prompt, dict):
                    if "from_file" not in prompt or len(prompt.keys()) != 1:
                        raise ValueError(
                            "system_prompt dictionary must contain only the 'from_file' key"
                        )
        except Exception:
            # Defer detailed type errors to pydantic after-hook
            pass
        return data


__all__ = ["AgentModel"]
