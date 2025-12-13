from __future__ import annotations

from typing import Optional, Union

from pydantic import BaseModel, model_validator, Field
from flujo.type_definitions.common import JSONObject


class AgentModel(BaseModel):
    model: str

    # FSD-022 + Template Variables: Allow externalized prompt via
    # { from_file: "path", variables: { ... } }
    class PromptTemplateSpec(BaseModel):
        from_file: str
        variables: Optional[JSONObject] = None

    system_prompt: Union[str, "AgentModel.PromptTemplateSpec"]
    output_schema: JSONObject
    # Optional provider-specific controls (e.g., GPT-5: reasoning, text verbosity)
    model_settings: Optional[JSONObject] = None
    # Optional execution controls
    timeout: Optional[Union[int, str]] = Field(
        default=None,
        description=(
            "Agent call timeout in seconds. Accepts integer (e.g., 30) or a numeric string"
            " (e.g., '30')."
        ),
    )
    max_retries: Optional[Union[int, str]] = Field(
        default=None,
        description=("Maximum retry attempts. Accepts integer or numeric string (e.g., '2')."),
    )

    # Validate 'system_prompt' dict form contains only 'from_file'
    @model_validator(mode="before")
    @classmethod
    def _validate_prompt_format(cls, data: object) -> object:
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
