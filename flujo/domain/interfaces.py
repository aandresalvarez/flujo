"""Lightweight Protocol interfaces to decouple core execution from concrete DSL types."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Callable, Optional, Protocol, TypeAlias, TypeVar, runtime_checkable

from flujo.type_definitions.common import JSONObject


@runtime_checkable
class PipelineContextLike(Protocol):
    """Context contract required by execution/CLI."""

    def model_dump(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class RunnerLike(Protocol):
    """Abstract interface for a runner (Flujo)."""

    def as_step(
        self,
        name: str,
        *,
        inherit_context: bool = True,
        **kwargs: Any,
    ) -> Any: ...


RunnerFactory: TypeAlias = Callable[[Any], RunnerLike]
_default_runner_factory: RunnerFactory | None = None


def set_default_runner_factory(factory: RunnerFactory) -> None:
    global _default_runner_factory
    _default_runner_factory = factory


def get_runner_factory() -> RunnerFactory | None:
    return _default_runner_factory


StepLike: TypeAlias = Any
PipelineLike: TypeAlias = Any


__all__ = [
    "PipelineContextLike",
    "RunnerLike",
    "RunnerFactory",
    "set_default_runner_factory",
    "get_runner_factory",
    "StepLike",
    "PipelineLike",
]

# ---- Providers / infra-light contracts ----

TState = TypeVar("TState")


class StateProvider(Protocol[TState]):
    """Minimal async storage interface used for ContextReference hydration."""

    async def load(self, key: str) -> TState | None: ...

    async def save(self, key: str, value: TState) -> None: ...


__all__ += ["StateProvider"]

# ---- Telemetry and settings provider hooks (lightweight, overridable) ----


class TelemetrySink(Protocol):
    def info(self, message: str, *args: Any, **kwargs: Any) -> None: ...

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None: ...

    def error(self, message: str, *args: Any, **kwargs: Any) -> None: ...

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None: ...

    def span(self, name: str, *args: Any, **kwargs: Any) -> Any: ...


class SettingsProvider(Protocol):
    def get_settings(self) -> Any: ...


_default_telemetry_sink: TelemetrySink | None = None
_default_settings_provider: SettingsProvider | None = None
# Backward-compatibility aliases expected by tests
_DEFAULT_TELEMETRY_SINK: TelemetrySink | None = None
_DEFAULT_SETTINGS_PROVIDER: SettingsProvider | None = None


def set_default_telemetry_sink(sink: TelemetrySink) -> None:
    global _default_telemetry_sink
    _default_telemetry_sink = sink
    globals()["_DEFAULT_TELEMETRY_SINK"] = sink


def get_telemetry_sink() -> TelemetrySink:
    sink = globals().get("_DEFAULT_TELEMETRY_SINK") or _default_telemetry_sink
    if sink is not None:
        return sink
    # Fallback logger-based sink
    logger = logging.getLogger("flujo")

    class _LoggerSink:
        def info(self, message: str, *args: Any, **kwargs: Any) -> None:
            logger.info(message, *args, **kwargs)

        def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
            logger.warning(message, *args, **kwargs)

        def error(self, message: str, *args: Any, **kwargs: Any) -> None:
            logger.error(message, *args, **kwargs)

        def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
            logger.debug(message, *args, **kwargs)

        def span(self, name: str, *args: Any, **kwargs: Any) -> Any:
            return nullcontext(self)  # no-op span surrogate as context manager

    return _LoggerSink()


def set_default_settings_provider(provider: SettingsProvider) -> None:
    global _default_settings_provider
    _default_settings_provider = provider
    globals()["_DEFAULT_SETTINGS_PROVIDER"] = provider


def get_settings_provider() -> SettingsProvider:
    provider = globals().get("_DEFAULT_SETTINGS_PROVIDER") or _default_settings_provider
    if provider is not None:
        return provider

    from flujo.infra.settings import get_settings as _get_settings

    class _Provider:
        def get_settings(self) -> Any:
            return _get_settings()

    return _Provider()


__all__ += [
    "TelemetrySink",
    "SettingsProvider",
    "set_default_telemetry_sink",
    "get_telemetry_sink",
    "set_default_settings_provider",
    "get_settings_provider",
]

# ---- Skill registry hooks ----


SkillFactory: TypeAlias = Callable[..., Any]


class SkillResolver(Protocol):
    def get(
        self, skill_id: str, *, scope: Optional[str] = None, version: str | None = None
    ) -> Optional[JSONObject]: ...


class SkillRegistry(SkillResolver, Protocol):
    def register(
        self,
        id: str,
        factory: Any,
        *,
        scope: str | None = None,
        description: Optional[str] = None,
        input_schema: Optional[JSONObject] = None,
        output_schema: Optional[JSONObject] = None,
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
    ) -> Optional[JSONObject]: ...


class SkillRegistryProvider(Protocol):
    def get_registry(self) -> SkillRegistry: ...


_default_skill_registry_provider: SkillRegistryProvider | None = None
_DEFAULT_SKILL_RESOLVER: SkillResolver | None = None


def set_default_skill_registry_provider(provider: SkillRegistryProvider) -> None:
    global _default_skill_registry_provider
    _default_skill_registry_provider = provider


def get_skill_registry_provider() -> SkillRegistryProvider:
    if _default_skill_registry_provider is not None:
        return _default_skill_registry_provider
    # Use shared infra provider to keep registry entries consistent
    from flujo.infra.skill_registry import get_skill_registry_provider as _infra_provider

    return _infra_provider()


def get_skill_resolver() -> SkillResolver:
    if _DEFAULT_SKILL_RESOLVER is not None:
        return _DEFAULT_SKILL_RESOLVER
    provider = get_skill_registry_provider()
    return provider.get_registry()


def set_default_skill_resolver(resolver: SkillResolver) -> None:
    global _DEFAULT_SKILL_RESOLVER
    _DEFAULT_SKILL_RESOLVER = resolver
    globals()["_DEFAULT_SKILL_RESOLVER"] = resolver


__all__ += [
    "SkillResolver",
    "SkillRegistry",
    "SkillRegistryProvider",
    "set_default_skill_registry_provider",
    "get_skill_registry_provider",
    "get_skill_resolver",
    "set_default_skill_resolver",
    "SkillFactory",
]


# ---- Skills discovery / config providers ----
class SkillsDiscovery(Protocol):
    def load_catalog(self, base_dir: str) -> None: ...

    def load_entry_points(self) -> None: ...


_default_skills_discovery: SkillsDiscovery | None = None


def set_default_skills_discovery(discovery: SkillsDiscovery) -> None:
    global _default_skills_discovery
    _default_skills_discovery = discovery


def get_skills_discovery() -> SkillsDiscovery:
    if _default_skills_discovery is not None:
        return _default_skills_discovery

    class _NoopDiscovery:
        def load_catalog(self, base_dir: str) -> None:
            return None

        def load_entry_points(self) -> None:
            return None

    return _NoopDiscovery()


class ConfigProvider(Protocol):
    def load_config(self) -> Any: ...


_default_config_provider: ConfigProvider | None = None


def set_default_config_provider(provider: ConfigProvider) -> None:
    global _default_config_provider
    _default_config_provider = provider


def get_config_provider() -> ConfigProvider:
    if _default_config_provider is not None:
        return _default_config_provider

    from flujo.infra.config_manager import ConfigManager

    class _Provider(ConfigProvider):
        def __init__(self) -> None:
            self._manager = ConfigManager()

        def load_config(self) -> Any:
            return self._manager.load_config()

    return _Provider()


__all__ += [
    "SkillsDiscovery",
    "set_default_skills_discovery",
    "get_skills_discovery",
    "ConfigProvider",
    "set_default_config_provider",
    "get_config_provider",
]


# ---- Parameter introspection utilities ----
# These allow DSL modules to introspect callables without importing from core.


def accepts_param(func: Any, param_name: str) -> bool:
    """Check if a callable accepts a parameter with the given name.

    This utility enables DSL modules to check parameter signatures without
    importing from flujo.application.core, helping decouple DSL from execution.

    Args:
        func: The callable to inspect.
        param_name: The parameter name to check for.

    Returns:
        True if the callable accepts the parameter, False otherwise.
    """
    import inspect

    try:
        sig = inspect.signature(func)
        return param_name in sig.parameters
    except (ValueError, TypeError):
        # Cannot inspect (e.g., built-in, C extension)
        return False


__all__ += ["accepts_param"]
