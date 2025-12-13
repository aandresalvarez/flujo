from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal, Optional

GovernanceMode = Literal["allow_all", "deny_all"]
SandboxMode = Literal["null", "remote", "docker"]


@dataclass
class Settings:
    """Process-level settings for Flujo.

    Sources:
    - Environment variables (authoritative for runtime toggles)
    - Future: project config (flujo.toml) via a dedicated loader
    """

    # FSD-009: Pure quota mode is now the only supported mode
    pure_quota_mode: bool = True
    test_mode: bool = False
    warn_legacy: bool = False
    # NOTE: enforce_typed_context removed - strict mode is always on (executor_helpers.py:136-148)
    memory_indexing_enabled: bool = False
    memory_embedding_model: str | None = None
    governance_mode: GovernanceMode = "allow_all"
    governance_policy_module: str | None = None
    shadow_eval_enabled: bool = False
    shadow_eval_sample_rate: float = 0.0
    shadow_eval_timeout_s: float = 30.0
    shadow_eval_judge_model: str = "openai:gpt-4o-mini"
    shadow_eval_sink: str = "telemetry"
    sandbox_mode: SandboxMode = "null"
    sandbox_api_url: str | None = None
    sandbox_api_key: str | None = None
    sandbox_timeout_s: float = 60.0
    sandbox_verify_ssl: bool = True
    sandbox_docker_image: str = "python:3.11-slim"
    sandbox_docker_pull: bool = True


_CACHED_SETTINGS: Optional[Settings] = None


def _load_from_env() -> Settings:
    def _flag(name: str) -> bool:
        v = os.getenv(name, "")
        return v.lower() in {"1", "true", "yes"}

    def _mode(name: str, default: GovernanceMode) -> GovernanceMode:
        value = os.getenv(name, default)
        normalized = value.lower().strip()
        if normalized == "allow_all":
            return "allow_all"
        if normalized == "deny_all":
            return "deny_all"
        return default

    raw_sandbox_mode = os.getenv("FLUJO_SANDBOX_MODE", "null").lower().strip() or "null"
    sandbox_mode_map: dict[str, SandboxMode] = {
        "null": "null",
        "remote": "remote",
        "docker": "docker",
    }
    sandbox_mode = sandbox_mode_map.get(raw_sandbox_mode, "null")

    return Settings(
        # Always use pure quota mode - legacy system removed
        pure_quota_mode=True,
        test_mode=_flag("FLUJO_TEST_MODE"),
        warn_legacy=_flag("FLUJO_WARN_LEGACY"),
        # NOTE: enforce_typed_context removed - strict mode is always on
        memory_indexing_enabled=_flag("FLUJO_MEMORY_INDEXING_ENABLED"),
        memory_embedding_model=os.getenv("FLUJO_MEMORY_EMBEDDING_MODEL"),
        governance_mode=_mode("FLUJO_GOVERNANCE_MODE", "allow_all"),
        governance_policy_module=os.getenv("FLUJO_GOVERNANCE_POLICY_MODULE"),
        shadow_eval_enabled=_flag("FLUJO_SHADOW_EVAL_ENABLED"),
        shadow_eval_sample_rate=float(os.getenv("FLUJO_SHADOW_EVAL_SAMPLE_RATE", "0") or "0"),
        shadow_eval_timeout_s=float(os.getenv("FLUJO_SHADOW_EVAL_TIMEOUT_S", "30") or "30"),
        shadow_eval_judge_model=os.getenv("FLUJO_SHADOW_EVAL_JUDGE_MODEL", "openai:gpt-4o-mini"),
        shadow_eval_sink=os.getenv("FLUJO_SHADOW_EVAL_SINK", "telemetry"),
        sandbox_mode=sandbox_mode,
        sandbox_api_url=os.getenv("FLUJO_SANDBOX_API_URL"),
        sandbox_api_key=os.getenv("FLUJO_SANDBOX_API_KEY"),
        sandbox_timeout_s=float(os.getenv("FLUJO_SANDBOX_TIMEOUT_S", "60") or "60"),
        sandbox_verify_ssl=_flag("FLUJO_SANDBOX_VERIFY_SSL")
        if "FLUJO_SANDBOX_VERIFY_SSL" in os.environ
        else True,
        sandbox_docker_image=os.getenv("FLUJO_SANDBOX_DOCKER_IMAGE", "python:3.11-slim"),
        sandbox_docker_pull=_flag("FLUJO_SANDBOX_DOCKER_PULL")
        if "FLUJO_SANDBOX_DOCKER_PULL" in os.environ
        else True,
    )


def get_settings() -> Settings:
    """Return Settings derived from environment, with test-friendly refresh.

    The environment often changes during tests (monkeypatch). To avoid stale
    values causing flaky behavior, we rebuild the Settings object on every call.
    The overhead is negligible compared to I/O and execution.
    """
    return _load_from_env()


class ConfigManager:
    """Compatibility façade for team guideline naming."""

    @staticmethod
    def get_settings() -> Settings:  # pragma: no cover - simple passthrough
        return get_settings()
