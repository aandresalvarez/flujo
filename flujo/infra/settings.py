"""Settings and configuration for flujo."""

import os
from typing import Any, Callable, ClassVar, Dict, Literal, Optional

import dotenv
from pydantic import (
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    AliasChoices,
    model_validator,
    BaseModel,
    ConfigDict,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..exceptions import SettingsError

dotenv.load_dotenv()


class BackgroundTaskSettings(BaseModel):
    """Settings for background task management."""

    enable_state_tracking: bool = True
    enable_resumability: bool = True
    enable_quota: bool = True
    max_cost_per_task: float = 1.0
    max_tokens_per_task: int = 10000
    default_retry_limit: int = 3
    state_retention_days: int = 30
    stale_task_timeout_hours: int = 24


class ShadowEvalSettings(BaseModel):
    """Settings for shadow evaluations (LLM judge)."""

    enabled: bool = Field(
        default=False,
        description="Enable shadow evaluations (async judge scoring on sampled runs).",
    )
    sample_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of runs to sample for shadow evaluation (0-1).",
    )
    timeout_seconds: int = Field(
        default=30,
        description="Timeout in seconds for a single shadow evaluation task.",
        ge=1,
    )
    judge_model: Optional[str] = Field(
        default=None,
        description="LLM model/tool identifier for judge; if None, rely on default evaluator.",
    )
    sink: Literal["telemetry", "database"] = Field(
        default="telemetry",
        description="Where to record eval results. telemetry or database.",
    )


class SandboxSettings(BaseModel):
    """Settings for sandboxed code execution."""

    model_config = ConfigDict(populate_by_name=True)

    mode: Literal["null", "remote", "docker"] = Field(
        default="null",
        validation_alias=AliasChoices("FLUJO_SANDBOX_MODE", "flujo_sandbox_mode"),
        description="Sandbox provider to use: null (disabled), remote, or docker.",
    )
    api_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("FLUJO_SANDBOX_API_URL", "flujo_sandbox_api_url"),
        description="Base URL for remote sandbox API (required for remote mode).",
    )
    api_key: Optional[SecretStr] = Field(
        default=None,
        validation_alias=AliasChoices("FLUJO_SANDBOX_API_KEY", "flujo_sandbox_api_key"),
        description="API key for remote sandbox (optional).",
    )
    timeout_seconds: int = Field(
        default=60,
        validation_alias=AliasChoices("FLUJO_SANDBOX_TIMEOUT_S", "flujo_sandbox_timeout_s"),
        description="Request timeout in seconds for sandbox executions.",
        ge=1,
    )
    verify_ssl: bool = Field(
        default=True,
        validation_alias=AliasChoices("FLUJO_SANDBOX_VERIFY_SSL", "flujo_sandbox_verify_ssl"),
        description="Whether to verify TLS certificates for remote sandbox.",
    )
    docker_image: str = Field(
        default="python:3.13-slim",
        validation_alias=AliasChoices("FLUJO_SANDBOX_DOCKER_IMAGE", "flujo_sandbox_docker_image"),
        description="Docker image to use for docker sandbox executions.",
    )
    docker_pull: bool = Field(
        default=True,
        validation_alias=AliasChoices("FLUJO_SANDBOX_DOCKER_PULL", "flujo_sandbox_docker_pull"),
        description="Pull the docker image if not present locally.",
    )
    docker_mem_limit: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "FLUJO_SANDBOX_DOCKER_MEM_LIMIT", "flujo_sandbox_docker_mem_limit"
        ),
        description="Memory limit for docker sandbox containers (e.g., '512m', '1g').",
    )
    docker_pids_limit: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices(
            "FLUJO_SANDBOX_DOCKER_PIDS_LIMIT", "flujo_sandbox_docker_pids_limit"
        ),
        description="Maximum number of PIDs for docker sandbox containers.",
        ge=1,
    )
    docker_network_mode: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "FLUJO_SANDBOX_DOCKER_NETWORK_MODE", "flujo_sandbox_docker_network_mode"
        ),
        description="Docker network mode for sandbox containers (e.g., 'none', 'bridge').",
    )


class Settings(BaseSettings):
    """Application settings loaded from environment variables. Standard names are preferred."""

    # --- Test/CI toggles ---
    # Exposed so orchestration layers can cheaply short‑circuit expensive persistence
    # and monitoring paths during unit tests and CI perf suites.
    test_mode: bool = Field(
        False,
        validation_alias=AliasChoices("FLUJO_TEST_MODE", "flujo_test_mode"),
        description="Enable low‑overhead test mode (skip persistence, monitors, etc.)",
    )

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
    state_backend_span_export_enabled: Optional[bool] = Field(
        default=None,
        description=(
            "Export OpenTelemetry spans to the configured state backend. "
            "None enables auto mode (SQLite only)."
        ),
    )
    shadow_eval: ShadowEvalSettings = ShadowEvalSettings()
    tree_search_discovery_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "FLUJO_TREE_SEARCH_DISCOVERY_ENABLED",
            "flujo_tree_search_discovery_enabled",
        ),
        description="Enable TreeSearchStep discovery agent for deducing invariants.",
    )

    # --- Core strictness toggles ---
    # Strict mode is the default; opt-out is not supported in CI.
    strict_dsl: bool = Field(
        default=True,
        validation_alias=AliasChoices("FLUJO_STRICT_DSL", "flujo_strict_dsl"),
        description="Enable strict DSL/type enforcement (no loose Any/object flows).",
    )
    # Enforce strict context isolation and merging.
    strict_context_isolation: bool = True
    strict_context_merge: bool = True
    enforce_typed_context: bool = Field(
        default=True,
        validation_alias=AliasChoices("FLUJO_ENFORCE_TYPED_CONTEXT", "flujo_enforce_typed_context"),
        description="[DEPRECATED] Always True - strict mode enforced in executor_helpers.py (env flag ignored).",
    )
    governance_policy_module: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "FLUJO_GOVERNANCE_POLICY_MODULE",
            "flujo_governance_policy_module",
        ),
        description="Optional module path (pkg.mod:Class) to load a GovernancePolicy implementation.",
    )
    governance_tool_allowlist: str = Field(
        default="",
        validation_alias=AliasChoices(
            "FLUJO_GOVERNANCE_TOOL_ALLOWLIST",
            "flujo_governance_tool_allowlist",
        ),
        description=(
            "Comma-separated allowlist of tool/skill IDs allowed under governance; "
            "empty means allow all."
        ),
    )
    sandbox: SandboxSettings = SandboxSettings()
    memory_indexing_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "FLUJO_MEMORY_INDEXING_ENABLED", "flujo_memory_indexing_enabled"
        ),
        description="Enable indexing of successful step outputs into the configured vector store.",
    )
    memory_embedding_model: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "FLUJO_MEMORY_EMBEDDING_MODEL", "flujo_memory_embedding_model"
        ),
        description="Embedding model id (e.g., openai:text-embedding-3-small) used for memory indexing.",
    )
    memory_embedding_dimensions: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices(
            "FLUJO_MEMORY_EMBEDDING_DIMENSIONS", "flujo_memory_embedding_dimensions"
        ),
        description=(
            "Embedding vector dimensions for memory indexing. "
            "If unset, Flujo will infer known model sizes when possible."
        ),
    )

    # --- Background task management ---
    background_tasks: BackgroundTaskSettings = BackgroundTaskSettings()

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
    max_iters: int = Field(5, validation_alias="MAX_ITERS")
    k_variants: int = Field(3, validation_alias="K_VARIANTS")
    reflection_limit: int = Field(3, validation_alias="REFLECTION_LIMIT")
    scorer: Literal["ratio", "weighted", "reward"] = Field("ratio", validation_alias="SCORER")
    t_schedule: list[float] = Field([1.0, 0.8, 0.5, 0.2], validation_alias="T_SCHEDULE")
    otlp_endpoint: Optional[str] = Field(None, validation_alias="OTLP_ENDPOINT")
    agent_timeout: int = Field(
        60, validation_alias="AGENT_TIMEOUT"
    )  # Timeout in seconds for agent calls

    # --- Postgres state backend ---
    postgres_pool_min: int = Field(
        1,
        description="Minimum connection pool size for the Postgres state backend.",
        ge=1,
    )
    postgres_pool_max: int = Field(
        10,
        description="Maximum connection pool size for the Postgres state backend.",
        ge=1,
    )

    model_config: ClassVar[SettingsConfigDict] = {
        "env_file": ".env",
        "populate_by_name": True,
        "extra": "ignore",
    }

    @model_validator(mode="after")
    def load_dynamic_api_keys(self: "Settings") -> "Settings":
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
        if self.postgres_pool_max < self.postgres_pool_min:
            raise ValueError("postgres_pool_max must be >= postgres_pool_min")
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
        default=False,
        description="Use the new iterative step executor for better performance",
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
    settings = Settings()
except ValidationError as e:
    # Use custom exception for better error handling downstream
    raise SettingsError(f"Invalid or missing environment variables for Settings:\n{e}")

# Ensure OpenAI library can find the API key if provided
if settings.openai_api_key:
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key.get_secret_value())


# Note: The global get_settings() function now delegates to ConfigManager
# which handles caching and proper settings precedence internally


def get_settings() -> Settings:
    """Get the current settings instance.

    This function delegates to ConfigManager.get_settings() which implements
    the proper precedence: Defaults < TOML File < Environment Variables.
    """
    from .config_manager import get_config_manager

    return get_config_manager().get_settings()


# Wire domain-level settings provider interface to avoid direct infra imports in domain logic
set_default_settings_provider_fn: Optional[Callable[[Any], None]]
try:  # pragma: no cover - import guard
    from flujo.domain.interfaces import (
        set_default_settings_provider as _set_default_settings_provider_fn,
    )

    set_default_settings_provider_fn = _set_default_settings_provider_fn
except Exception:  # pragma: no cover - defensive fallback
    set_default_settings_provider_fn = None


class _SettingsProviderAdapter:
    def get_settings(self) -> Settings:  # pragma: no cover - simple delegation
        return get_settings()


if set_default_settings_provider_fn is not None:  # pragma: no cover - simple wiring
    try:
        set_default_settings_provider_fn(_SettingsProviderAdapter())
    except Exception:
        pass
