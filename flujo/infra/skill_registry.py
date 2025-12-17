from __future__ import annotations

from typing import Any, Callable, List, Optional, TYPE_CHECKING
import inspect
import functools

from flujo.type_definitions.common import JSONObject
from flujo.exceptions import ConfigurationError
from flujo.infra.settings import get_settings

# Domain interface adapter to avoid leaking infra into domain logic
if TYPE_CHECKING:
    from flujo.domain.interfaces import (
        SkillRegistry as SkillRegistryProtocol,
        SkillRegistryProvider as SkillRegistryProviderProtocol,
        SkillFactory,
        set_default_skill_registry_provider as set_default_skill_registry_provider_fn,
        set_default_skill_resolver as set_default_skill_resolver_fn,
    )
else:  # pragma: no cover - runtime import guard
    try:
        from flujo.domain.interfaces import (
            SkillRegistry as SkillRegistryProtocol,
            SkillRegistryProvider as SkillRegistryProviderProtocol,
            SkillFactory,
            set_default_skill_registry_provider as set_default_skill_registry_provider_fn,
            set_default_skill_resolver as set_default_skill_resolver_fn,
        )
    except Exception:
        SkillRegistryProtocol = object  # type: ignore[assignment]
        SkillRegistryProviderProtocol = object  # type: ignore[assignment]
        SkillFactory = Callable[..., Any]
        set_default_skill_registry_provider_fn = None
        set_default_skill_resolver_fn = None


class SkillRegistry(SkillRegistryProtocol):
    """Versioned, scoped registry for resolving skills/agents by ID."""

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, JSONObject]] = {}

    @staticmethod
    def _wrap_callable_for_governance(skill_id: str, obj: object) -> object:
        """Attach skill metadata and enforce tool allowlist at call time.

        Notes:
        - This wrapper preserves the original signature via __signature__ to avoid
          breaking tool schema generation in pydantic-ai.
        - Enforcement is configured via `Settings.governance_tool_allowlist` (TOML/env).
        """
        if not callable(obj):
            return obj
        try:
            setattr(obj, "__flujo_skill_id__", skill_id)
        except Exception:
            pass

        allowlist_raw = getattr(get_settings(), "governance_tool_allowlist", ())
        if not allowlist_raw:
            return obj
        if isinstance(allowlist_raw, str):
            allowed = {p.strip() for p in allowlist_raw.split(",") if p.strip()}
        else:
            allowed = {str(p).strip() for p in allowlist_raw if str(p).strip()}
        if not allowed:
            return obj

        try:
            sig = inspect.signature(obj)
        except Exception:
            sig = None

        if inspect.iscoroutinefunction(obj):

            @functools.wraps(obj)
            async def _wrapped(*args: object, **kwargs: object) -> object:
                if skill_id not in allowed:
                    raise ConfigurationError(f"tool_not_allowed:{skill_id}")
                return await obj(*args, **kwargs)

        else:

            @functools.wraps(obj)
            def _wrapped(*args: object, **kwargs: object) -> object:
                if skill_id not in allowed:
                    raise ConfigurationError(f"tool_not_allowed:{skill_id}")
                return obj(*args, **kwargs)

        try:
            setattr(_wrapped, "__flujo_skill_id__", skill_id)
        except Exception:
            pass
        if sig is not None:
            try:
                setattr(_wrapped, "__signature__", sig)
            except Exception:
                pass
        return _wrapped

    @classmethod
    def _wrap_factory_for_governance(cls, skill_id: str, factory: object) -> object:
        if not callable(factory):
            return factory
        try:
            sig = inspect.signature(factory)
        except Exception:
            sig = None

        @functools.wraps(factory)
        def _wrapped_factory(*args: object, **kwargs: object) -> object:
            produced = factory(*args, **kwargs)
            return cls._wrap_callable_for_governance(skill_id, produced)

        if sig is not None:
            try:
                setattr(_wrapped_factory, "__signature__", sig)
            except Exception:
                pass
        return _wrapped_factory

    def register(
        self,
        id: str,
        factory: SkillFactory | Any,
        *,
        scope: str | None = None,
        description: Optional[str] = None,
        input_schema: Optional[JSONObject] = None,
        # FSD-020 naming alias: arg_schema is accepted as alias of input_schema
        output_schema: Optional[JSONObject] = None,
        capabilities: Optional[List[str]] = None,
        safety_level: Optional[str] = None,
        auth_required: Optional[bool] = None,
        auth_scope: Optional[str] = None,
        side_effects: Optional[bool] = None,
        arg_schema: Optional[JSONObject] = None,
        version: str | None = None,
    ) -> None:
        # Prefer explicit input_schema; fall back to arg_schema for compatibility with FSD specs
        effective_input_schema: Optional[dict[str, Any]] = (
            input_schema if input_schema is not None else arg_schema
        )
        wrapped_factory = self._wrap_factory_for_governance(id, factory)
        scope_key = scope or "default"
        versions = self._entries.setdefault(scope_key, {})
        version_key = version or "latest"
        scoped = versions.setdefault(id, {})

        scoped[version_key] = {
            "factory": wrapped_factory,
            "description": description,
            "input_schema": effective_input_schema,
            "output_schema": output_schema,
            "capabilities": capabilities or [],
            "safety_level": safety_level or "none",
            "auth_required": bool(auth_required) if auth_required is not None else False,
            "auth_scope": auth_scope,
            "side_effects": bool(side_effects) if side_effects is not None else False,
            "version": version_key,
            "scope": scope_key,
        }

    def get(
        self, id: str, *, scope: str | None = None, version: str | None = None
    ) -> Optional[JSONObject]:
        scope_key = scope or "default"
        scoped = self._entries.get(scope_key, {})
        versions = scoped.get(id)
        if versions is None and id.startswith("flujo.builtins."):
            # Force a fresh registration for any builtin miss to avoid flakiness.
            try:
                from flujo.builtins import _register_builtins as _reg

                # Reset only the default scope to keep tenant scopes isolated
                self._entries["default"] = {}
                _reg()
                scoped_default = self._entries.get("default", {})
                versions = scoped_default.get(id) if scope_key == "default" else scoped.get(id)
            except Exception:
                versions = None
        if versions is None:
            return None
        if version is None or version == "latest":
            # Prefer an explicitly registered "latest" entry when present to avoid
            # parsing arbitrary version strings.
            if "latest" in versions:
                latest_entry = versions.get("latest")
                if isinstance(latest_entry, dict):
                    return latest_entry
            # Return the latest registered version by lexical order
            try:
                from packaging.version import Version

                candidates: list[tuple[Version, str]] = []
                for key in versions.keys():
                    try:
                        candidates.append((Version(key), key))
                    except Exception:
                        continue
                if candidates:
                    candidates.sort()
                    latest_key = candidates[-1][1]
                else:
                    latest_key = max(versions.keys())
            except Exception:
                latest_key = max(versions.keys())
            return versions.get(latest_key)
        return versions.get(version)


def get_skill_registry(scope: str | None = None) -> SkillRegistryProtocol:
    """Legacy accessor; returns the default-scope registry from the provider."""

    # Use provider to ensure consistent scoping behavior
    return get_skill_registry_provider().get_registry(scope=scope)


class _SkillRegistryResolver:
    """Adapter exposing SkillRegistry through the domain SkillResolver protocol."""

    def get(
        self, skill_id: str, *, scope: str | None = None, version: str | None = None
    ) -> Optional[dict[str, Any]]:
        return get_skill_registry().get(skill_id, scope=scope, version=version)


class SkillRegistryProvider(SkillRegistryProviderProtocol):
    """Provide scoped registries (per scope name)."""

    def __init__(self) -> None:
        self._registries: dict[str, SkillRegistry] = {}

    def get_registry(self, *, scope: str | None = None) -> SkillRegistryProtocol:
        scope_key = scope or "default"
        reg = self._registries.get(scope_key)
        if reg is None:
            reg = SkillRegistry()
            self._registries[scope_key] = reg
        return reg


_GLOBAL_PROVIDER: Optional[SkillRegistryProvider] = None


def get_skill_registry_provider() -> SkillRegistryProvider:
    global _GLOBAL_PROVIDER
    if _GLOBAL_PROVIDER is None:
        _GLOBAL_PROVIDER = SkillRegistryProvider()
    return _GLOBAL_PROVIDER


def reset_skill_registry_provider() -> None:
    """Reset the global registry provider (CLI/test hygiene).

    The CLI can be invoked multiple times within a single Python process during tests
    (e.g., via Typer's CliRunner). Pipelines/plugins may register skills dynamically,
    so resetting the provider prevents cross-test contamination without relying on
    broad process-wide cleanup.
    """
    global _GLOBAL_PROVIDER
    _GLOBAL_PROVIDER = SkillRegistryProvider()
    try:
        if set_default_skill_registry_provider_fn is not None:  # pragma: no cover - wiring
            set_default_skill_registry_provider_fn(_GLOBAL_PROVIDER)
    except Exception:
        pass
    try:
        from flujo.builtins import _register_builtins as _reg

        _reg()
    except Exception:
        pass


# Register default provider/resolver for domain consumers while keeping dependency direction infra->domain
if set_default_skill_registry_provider_fn is not None:  # pragma: no cover - simple wiring
    try:
        set_default_skill_registry_provider_fn(get_skill_registry_provider())
    except Exception:
        pass

if set_default_skill_resolver_fn is not None:  # pragma: no cover - simple wiring
    try:
        set_default_skill_resolver_fn(_SkillRegistryResolver())
    except Exception:
        pass


__all__ = [
    "SkillRegistry",
    "SkillRegistryProvider",
    "get_skill_registry",
    "get_skill_registry_provider",
    "reset_skill_registry_provider",
]
