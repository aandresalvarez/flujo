"""Settings and configuration for flujo."""

import os
from typing import Callable, ClassVar, Dict, Literal, Optional, cast

import dotenv
from pydantic import (
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    AliasChoices,
    model_validator,
    BaseModel,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..exceptions import SettingsError

dotenv.load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables. Standard names are preferred."""

    # --- API Keys (with backward compatibility) ---
    openai_api_key: Optional[SecretStr] = Field(
        None,
        validation_alias=AliasChoices(
            "OPENAI_API_KEY",
            "ORCH_OPENAI_API_KEY",
            "orch_openai_api_key",
        ),
    )
    google_api_key: Optional[SecretStr] = Field(
        None,
        validation_alias=AliasChoices(
            "GOOGLE_API_KEY",
            "ORCH_GOOGLE_API_KEY",
            "orch_google_api_key",
        ),
    )
    anthropic_api_key: Optional[SecretStr] = Field(
        None,
        validation_alias=AliasChoices(
            "ANTHROPIC_API_KEY",
            "ORCH_ANTHROPIC_API_KEY",
            "orch_anthropic_api_key",
        ),
    )
    logfire_api_key: Optional[SecretStr] = Field(
        None,
        validation_alias=AliasChoices(
            "LOGFIRE_API_KEY",
            "ORCH_LOGFIRE_API_KEY",
            "orch_logfire_api_key",
        ),
    )

    # --- Dynamic dictionary for other provider keys ---
    provider_api_keys: Dict[str, SecretStr] = Field(default_factory=dict)

    # --- Feature Toggles ---
    reflection_enabled: bool = True
    reward_enabled: bool = True
    telemetry_export_enabled: bool = False
    otlp_export_enabled: bool = False

    # --- Default models for each agent ---
    default_solution_model: str = "openai:gpt-4o"
    default_review_model: str = "openai:gpt-4o"
    default_validator_model: str = "openai:gpt-4o"
    default_reflection_model: str = "openai:gpt-4o"
    default_self_improvement_model: str = Field(
        "openai:gpt-4o",
        description="Default model to use for the SelfImprovementAgent.",
    )
    default_repair_model: str = Field(
        "openai:gpt-4o",
        description="Default model used for the internal repair agent.",
    )

    # --- Orchestrator Tuning ---
    max_iters: int = 5
    k_variants: int = 3
    reflection_limit: int = 3
    scorer: Literal["ratio", "weighted", "reward"] = "ratio"
    t_schedule: list[float] = [1.0, 0.8, 0.5, 0.2]
    otlp_endpoint: Optional[str] = None
    agent_timeout: int = 60  # Timeout in seconds for agent calls

    model_config: ClassVar[SettingsConfigDict] = {
        "env_file": ".env",
        "populate_by_name": True,
        "extra": "ignore",
    }

    @model_validator(mode="after")
    def load_dynamic_api_keys(self) -> "Settings":
        """Load any additional *_API_KEY variables from the environment."""
        handled_keys: set[str] = set()
        for field in self.__class__.model_fields.values():
            alias = field.validation_alias
            if isinstance(alias, AliasChoices):
                handled_keys.update(a.upper() for a in alias.choices if isinstance(a, str))
            elif isinstance(alias, str):
                handled_keys.add(alias.upper())
        for key, value in os.environ.items():
            upper_key = key.upper()
            if upper_key.endswith("_API_KEY") and upper_key not in handled_keys:
                provider_name = upper_key.removesuffix("_API_KEY").lower()
                if value:
                    self.provider_api_keys[provider_name] = SecretStr(value)
        return self

    @field_validator("t_schedule")
    def schedule_must_not_be_empty(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("t_schedule must not be empty")
        return v


class ExecutionConfig(BaseModel):
    """Configuration for step execution optimization."""

    # Iterative executor settings
    use_iterative_executor: bool = Field(
        default=False, description="Use the new iterative step executor for better performance"
    )

    # Memoization settings
    enable_memoization: bool = Field(default=True, description="Enable step result memoization")
    cache_size: int = Field(default=1000, description="Maximum number of cached step results")
    cache_ttl_seconds: int = Field(
        default=3600, description="Time-to-live for cached results in seconds"
    )

    # Parallel processing settings
    enable_parallel_validation: bool = Field(default=True, description="Run validators in parallel")
    max_parallel_validators: int = Field(
        default=10, description="Maximum number of validators to run in parallel"
    )

    # Context optimization settings
    enable_context_optimization: bool = Field(
        default=True, description="Enable context copying optimizations"
    )
    lazy_context_copying: bool = Field(
        default=True, description="Only copy context when actually needed"
    )

    # Resource management settings
    enable_resource_pooling: bool = Field(
        default=False, description="Enable resource pooling for expensive operations"
    )
    max_concurrent_steps: int = Field(
        default=50, description="Maximum number of concurrent step executions"
    )


# Singleton instance, fail fast if critical vars missing
# Note: This will be overridden by the configuration manager when available
try:
    settings = cast(Callable[[], Settings], Settings)()
except ValidationError as e:
    # Use custom exception for better error handling downstream
    raise SettingsError(f"Invalid or missing environment variables for Settings:\n{e}")

# Ensure OpenAI library can find the API key if provided
if settings.openai_api_key:
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key.get_secret_value())


def get_settings() -> Settings:
    """Get the current settings instance.

    This function provides a way to get settings that may be overridden
    by configuration files. It will use the configuration manager if available,
    otherwise fall back to the default settings.
    """
    try:
        from .config_manager import load_settings

        return load_settings()
    except ImportError:
        # Fall back to default settings if config manager is not available
        return settings
