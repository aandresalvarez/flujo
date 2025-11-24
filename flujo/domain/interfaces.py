from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol


class SkillResolver(Protocol):
    """Resolve skills/agents by identifier."""

    def get(self, skill_id: str, *, scope: Optional[str] = None) -> Optional[dict[str, Any]]: ...


class TelemetrySink(Protocol):
    """Minimal telemetry surface used by domain components."""

    def info(self, message: str, *args: Any, **kwargs: Any) -> None: ...

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None: ...

    def error(self, message: str, *args: Any, **kwargs: Any) -> None: ...

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None: ...

    def span(self, name: str, *args: Any, **kwargs: Any) -> Any: ...


class ConfigProvider(Protocol):
    """Provide access to configuration (e.g., flujo.toml) without hard infra deps."""

    def load_config(self) -> Any: ...


class SettingsProvider(Protocol):
    """Provide access to runtime settings (e.g., env-backed feature toggles)."""

    def get_settings(self) -> Any: ...


class SkillsDiscovery(Protocol):
    """Load skills/entry points from a given base directory or environment."""

    def load_catalog(self, base_dir: str) -> None: ...

    def load_entry_points(self) -> None: ...


class SkillRegistry(Protocol):
    """Registry Protocol to allow registration and resolution of skills."""

    _entries: dict[str, dict[str, dict[str, Any]]]

    def register(
        self,
        id: str,
        factory: Any,
        *,
        scope: str | None = None,
        description: Optional[str] = None,
        input_schema: Optional[dict[str, Any]] = None,
        output_schema: Optional[dict[str, Any]] = None,
        capabilities: Optional[list[str]] = None,
        safety_level: Optional[str] = None,
        auth_required: Optional[bool] = None,
        auth_scope: Optional[str] = None,
        side_effects: Optional[bool] = None,
        arg_schema: Optional[dict[str, Any]] = None,
        version: str | None = None,
    ) -> None: ...

    def get(
        self, id: str, *, scope: str | None = None, version: str | None = None
    ) -> Optional[dict[str, Any]]: ...


class SkillRegistryProvider(Protocol):
    """Provide scoped skill registries."""

    def get_registry(self, *, scope: str | None = None) -> SkillRegistry: ...


class StateProvider(Protocol):
    """Protocol for external state providers that manage data persistence."""

    async def load(self, key: str) -> Any:
        """Fetch data from external storage."""
        ...

    async def save(self, key: str, data: Any) -> None:
        """Commit data to external storage."""
        ...


@dataclass
class _NullTelemetrySpan:
    def __enter__(self) -> "_NullTelemetrySpan":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:  # pragma: no cover - trivial
        return None


class _NullSkillResolver:
    def get(
        self, skill_id: str, *, scope: Optional[str] = None
    ) -> Optional[dict[str, Any]]:  # pragma: no cover - trivial
        return None


class _NullTelemetrySink:
    def info(self, message: str, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - trivial
        return None

    def warning(
        self, message: str, *args: Any, **kwargs: Any
    ) -> None:  # pragma: no cover - trivial
        return None

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - trivial
        return None

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - trivial
        return None

    def span(self, name: str, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - trivial
        return _NullTelemetrySpan()


class _NullConfigProvider:
    def load_config(self) -> Any:  # pragma: no cover - trivial
        return None


class _NullSettingsProvider:
    def get_settings(self) -> Any:  # pragma: no cover - trivial
        return None


class _NullSkillsDiscovery:
    def load_catalog(self, base_dir: str) -> None:  # pragma: no cover - trivial
        return None

    def load_entry_points(self) -> None:  # pragma: no cover - trivial
        return None


class _NullSkillRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, dict[str, dict[str, Any]]] = {}

    def register(
        self,
        id: str,
        factory: Any,
        *,
        scope: str | None = None,
        description: Optional[str] = None,
        input_schema: Optional[dict[str, Any]] = None,
        output_schema: Optional[dict[str, Any]] = None,
        capabilities: Optional[list[str]] = None,
        safety_level: Optional[str] = None,
        auth_required: Optional[bool] = None,
        auth_scope: Optional[str] = None,
        side_effects: Optional[bool] = None,
        arg_schema: Optional[dict[str, Any]] = None,
        version: str | None = None,
    ) -> None:  # pragma: no cover - trivial
        return None

    def get(
        self, id: str, *, scope: str | None = None, version: str | None = None
    ) -> Optional[dict[str, Any]]:
        return None


class _NullSkillRegistryProvider:
    def __init__(self) -> None:
        self._default_registry: SkillRegistry = _NullSkillRegistry()

    def get_registry(self, *, scope: str | None = None) -> SkillRegistry:
        return self._default_registry


_DEFAULT_SKILL_RESOLVER: SkillResolver = _NullSkillResolver()
_DEFAULT_TELEMETRY_SINK: TelemetrySink = _NullTelemetrySink()
_DEFAULT_CONFIG_PROVIDER: ConfigProvider = _NullConfigProvider()
_DEFAULT_SETTINGS_PROVIDER: SettingsProvider = _NullSettingsProvider()
_DEFAULT_SKILLS_DISCOVERY: SkillsDiscovery = _NullSkillsDiscovery()
_DEFAULT_SKILL_REGISTRY_PROVIDER: SkillRegistryProvider = _NullSkillRegistryProvider()


def set_default_skill_resolver(resolver: SkillResolver) -> None:
    """Register the process-wide SkillResolver."""

    global _DEFAULT_SKILL_RESOLVER
    _DEFAULT_SKILL_RESOLVER = resolver


def get_skill_resolver() -> SkillResolver:
    """Return the configured SkillResolver (null object if unset)."""

    global _DEFAULT_SKILL_RESOLVER
    if isinstance(_DEFAULT_SKILL_RESOLVER, _NullSkillResolver):
        # Lazy-load infra adapter if available to preserve default wiring without hard dependency
        try:  # pragma: no cover - best-effort wiring
            import flujo.infra.skill_registry as _  # noqa: F401
        except Exception:
            pass
    return _DEFAULT_SKILL_RESOLVER


def set_default_telemetry_sink(sink: TelemetrySink) -> None:
    """Register the process-wide TelemetrySink."""

    global _DEFAULT_TELEMETRY_SINK
    _DEFAULT_TELEMETRY_SINK = sink


def get_telemetry_sink() -> TelemetrySink:
    """Return the configured TelemetrySink (null object if unset)."""

    return _DEFAULT_TELEMETRY_SINK


def set_default_config_provider(provider: ConfigProvider) -> None:
    """Register the process-wide ConfigProvider."""

    global _DEFAULT_CONFIG_PROVIDER
    _DEFAULT_CONFIG_PROVIDER = provider


def get_config_provider() -> ConfigProvider:
    """Return the configured ConfigProvider (null object if unset)."""

    global _DEFAULT_CONFIG_PROVIDER
    if isinstance(_DEFAULT_CONFIG_PROVIDER, _NullConfigProvider):
        try:  # pragma: no cover - best-effort wiring
            import flujo.infra.config_manager as _  # noqa: F401
        except Exception:
            pass
    return _DEFAULT_CONFIG_PROVIDER


def set_default_settings_provider(provider: SettingsProvider) -> None:
    """Register the process-wide SettingsProvider."""

    global _DEFAULT_SETTINGS_PROVIDER
    _DEFAULT_SETTINGS_PROVIDER = provider


def get_settings_provider() -> SettingsProvider:
    """Return the configured SettingsProvider (null object if unset)."""

    global _DEFAULT_SETTINGS_PROVIDER
    if isinstance(_DEFAULT_SETTINGS_PROVIDER, _NullSettingsProvider):
        try:  # pragma: no cover - best-effort wiring
            import flujo.infra.settings as _  # noqa: F401
        except Exception:
            pass
    return _DEFAULT_SETTINGS_PROVIDER


def set_default_skills_discovery(discovery: SkillsDiscovery) -> None:
    """Register the process-wide SkillsDiscovery helper."""

    global _DEFAULT_SKILLS_DISCOVERY
    _DEFAULT_SKILLS_DISCOVERY = discovery


def get_skills_discovery() -> SkillsDiscovery:
    """Return the configured SkillsDiscovery helper (null object if unset)."""

    global _DEFAULT_SKILLS_DISCOVERY
    if isinstance(_DEFAULT_SKILLS_DISCOVERY, _NullSkillsDiscovery):
        try:  # pragma: no cover - best-effort wiring
            import flujo.infra.skills_catalog as _  # noqa: F401
        except Exception:
            pass
    return _DEFAULT_SKILLS_DISCOVERY


def set_default_skill_registry_provider(provider: SkillRegistryProvider) -> None:
    """Register the process-wide SkillRegistry provider."""

    global _DEFAULT_SKILL_REGISTRY_PROVIDER
    _DEFAULT_SKILL_REGISTRY_PROVIDER = provider


def get_skill_registry_provider() -> SkillRegistryProvider:
    """Return the configured SkillRegistry provider (null object if unset)."""

    global _DEFAULT_SKILL_REGISTRY_PROVIDER
    if isinstance(_DEFAULT_SKILL_REGISTRY_PROVIDER, _NullSkillRegistryProvider):
        try:  # pragma: no cover - best-effort wiring
            import flujo.infra.skill_registry as _  # noqa: F401
        except Exception:
            pass
    return _DEFAULT_SKILL_REGISTRY_PROVIDER


__all__ = [
    "SkillResolver",
    "TelemetrySink",
    "ConfigProvider",
    "SettingsProvider",
    "SkillsDiscovery",
    "SkillRegistry",
    "SkillRegistryProvider",
    "get_skill_resolver",
    "set_default_skill_resolver",
    "get_telemetry_sink",
    "set_default_telemetry_sink",
    "get_config_provider",
    "set_default_config_provider",
    "get_settings_provider",
    "set_default_settings_provider",
    "get_skills_discovery",
    "set_default_skills_discovery",
    "get_skill_registry_provider",
    "set_default_skill_registry_provider",
    "StateProvider",
]
