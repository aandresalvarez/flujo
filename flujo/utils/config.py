from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal, Optional


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
    governance_mode: Literal["allow_all", "deny_all"] = "allow_all"


_CACHED_SETTINGS: Optional[Settings] = None


def _load_from_env() -> Settings:
    def _flag(name: str) -> bool:
        v = os.getenv(name, "")
        return v.lower() in {"1", "true", "yes"}

    def _mode(
        name: str, default: Literal["allow_all", "deny_all"]
    ) -> Literal["allow_all", "deny_all"]:
        value = os.getenv(name, default)
        normalized = value.lower().strip()
        if normalized not in {"allow_all", "deny_all"}:
            return default
        return normalized  # type: ignore[return-value]

    return Settings(
        # Always use pure quota mode - legacy system removed
        pure_quota_mode=True,
        test_mode=_flag("FLUJO_TEST_MODE"),
        warn_legacy=_flag("FLUJO_WARN_LEGACY"),
        governance_mode=_mode("FLUJO_GOVERNANCE_MODE", "allow_all"),
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
