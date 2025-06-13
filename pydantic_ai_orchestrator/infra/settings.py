"""Settings and configuration for pydantic-ai-orchestrator."""

import os
from typing import ClassVar, Literal, Optional

import dotenv
from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..exceptions import SettingsError

dotenv.load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API keys are optional to allow starting without them
    openai_api_key: Optional[SecretStr] = Field(None, validation_alias="orch_openai_api_key")
    google_api_key: Optional[SecretStr] = Field(None, validation_alias="orch_google_api_key")
    anthropic_api_key: Optional[SecretStr] = Field(None, validation_alias="orch_anthropic_api_key")
    logfire_api_key: Optional[SecretStr] = Field(None, validation_alias="orch_logfire_api_key")

    # Feature Toggles
    reflection_enabled: bool = Field(True, validation_alias="orch_reflection_enabled")
    reward_enabled: bool = Field(True, validation_alias="orch_reward_enabled")
    telemetry_export_enabled: bool = Field(False, validation_alias="orch_telemetry_export_enabled")
    otlp_export_enabled: bool = Field(False, validation_alias="orch_otlp_export_enabled")

    # Default models for each agent
    default_solution_model: str = Field(
        "openai:gpt-4o", validation_alias="orch_default_solution_model"
    )
    default_review_model: str = Field(
        "openai:gpt-4o", validation_alias="orch_default_review_model"
    )
    default_validator_model: str = Field(
        "openai:gpt-4o", validation_alias="orch_default_validator_model"
    )
    default_reflection_model: str = Field(
        "openai:gpt-4o", validation_alias="orch_default_reflection_model"
    )

    # Orchestrator Tuning
    max_iters: int = 5
    k_variants: int = 3
    reflection_limit: int = 3
    scorer: Literal["ratio", "weighted", "reward"] = "ratio"
    t_schedule: list[float] = [1.0, 0.8, 0.5, 0.2]
    otlp_endpoint: Optional[str] = None
    agent_timeout: int = Field(60, validation_alias="orch_agent_timeout")  # Timeout in seconds for agent calls

    model_config: ClassVar[SettingsConfigDict] = {
        "env_file": ".env",
        "env_prefix": "orch_",
        "alias_generator": None,
        "populate_by_name": True,
        "extra": "ignore",
    }

    @field_validator("t_schedule")
    def schedule_must_not_be_empty(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("t_schedule must not be empty")
        return v


# Singleton instance, fail fast if critical vars missing
try:
    settings = Settings()  # type: ignore[call-arg]
except ValidationError as e:
    # Use custom exception for better error handling downstream
    raise SettingsError(f"Invalid or missing environment variables for Settings:\n{e}")

# Ensure OpenAI library can find the API key if provided
if settings.openai_api_key:
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key.get_secret_value())
