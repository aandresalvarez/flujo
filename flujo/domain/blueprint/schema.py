from __future__ import annotations

from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, model_validator


class AgentModel(BaseModel):
    model: str

    # FSD-022 + Template Variables: Allow externalized prompt via
    # { from_file: "path", variables: { ... } }
    class PromptTemplateSpec(BaseModel):
        from_file: str
        variables: Optional[Dict[str, Any]] = None

    system_prompt: Union[str, "AgentModel.PromptTemplateSpec"]
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
        if isinstance(data, dict):
            prompt = data.get("system_prompt")
            if isinstance(prompt, dict):
                # Allow {from_file} or {from_file, variables}
                allowed = {"from_file", "variables"}
                if "from_file" not in prompt or any(k not in allowed for k in prompt.keys()):
                    # Let ValueError propagate so Pydantic surfaces a clear error
                    raise ValueError(
                        "system_prompt dictionary must include 'from_file' and only optional 'variables'"
                    )
        return data


__all__ = ["AgentModel"]
