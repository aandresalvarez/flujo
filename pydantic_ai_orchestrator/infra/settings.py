"""Settings and configuration for pydantic-ai-orchestrator."""

from pydantic_settings import BaseSettings
from pydantic import ValidationError, SecretStr, field_validator, ConfigDict, Field
from typing import Optional, Literal
from ..exceptions import SettingsError


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API keys are optional to allow starting without them
    openai_api_key: Optional[SecretStr] = None
    google_api_key: Optional[SecretStr] = None
    anthropic_api_key: Optional[SecretStr] = None
    logfire_api_key: Optional[SecretStr] = None

    # Feature Toggles
    reflection_enabled: bool = Field(True)
    reward_enabled: bool = Field(True, alias="reward_enabled")
    telemetry_export_enabled: bool = False
    otlp_export_enabled: bool = False

    # Default models for each agent
    default_solution_model: str = "openai:gpt-4o"
    default_review_model: str = "openai:gpt-4o"
    default_validator_model: str = "openai:gpt-4o"
    default_reflection_model: str = "openai:gpt-4o"

    # Orchestrator Tuning
    max_iters: int = 5
    k_variants: int = 3
    reflection_limit: int = 3
    scorer: Literal["ratio", "weighted", "reward"] = "ratio"
    t_schedule: list[float] = [1.0, 0.8, 0.5, 0.2]
    otlp_endpoint: Optional[str] = None
    agent_timeout: int = 60  # Timeout in seconds for agent calls

    model_config = ConfigDict(
        env_file=".env",
        env_prefix="",
        alias_generator=None,
        populate_by_name=True,
    )

    @field_validator("t_schedule")
    def schedule_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("t_schedule must not be empty")
        return v


# Singleton instance, fail fast if critical vars missing
try:
    settings = Settings()
except ValidationError as e:
    # Use custom exception for better error handling downstream
    raise SettingsError(f"Invalid or missing environment variables for Settings:\n{e}")
