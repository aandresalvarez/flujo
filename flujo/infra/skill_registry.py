from __future__ import annotations

from typing import Any, Callable, Dict, Optional, List, TYPE_CHECKING

# Domain interface adapter to avoid leaking infra into domain logic
if TYPE_CHECKING:
    from flujo.domain.interfaces import (
        SkillRegistry as SkillRegistryProtocol,
        SkillRegistryProvider as SkillRegistryProviderProtocol,
        set_default_skill_registry_provider as set_default_skill_registry_provider_fn,
        set_default_skill_resolver as set_default_skill_resolver_fn,
    )
else:  # pragma: no cover - runtime import guard
    try:
        from flujo.domain.interfaces import (
            SkillRegistry as SkillRegistryProtocol,
            SkillRegistryProvider as SkillRegistryProviderProtocol,
            set_default_skill_registry_provider as set_default_skill_registry_provider_fn,
            set_default_skill_resolver as set_default_skill_resolver_fn,
        )
    except Exception:
        SkillRegistryProtocol = object  # type: ignore[assignment]
        SkillRegistryProviderProtocol = object  # type: ignore[assignment]
        set_default_skill_registry_provider_fn = None
        set_default_skill_resolver_fn = None


class SkillRegistry(SkillRegistryProtocol):
    """Versioned, scoped registry for resolving skills/agents by ID."""

    def __init__(self) -> None:
        self._entries: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def register(
        self,
        id: str,
        factory: Callable[..., Any] | Any,
        *,
        scope: str | None = None,
        description: Optional[str] = None,
        input_schema: Optional[dict[str, Any]] = None,
        # FSD-020 naming alias: arg_schema is accepted as alias of input_schema
        output_schema: Optional[dict[str, Any]] = None,
        capabilities: Optional[List[str]] = None,
        safety_level: Optional[str] = None,
        auth_required: Optional[bool] = None,
        auth_scope: Optional[str] = None,
        side_effects: Optional[bool] = None,
        arg_schema: Optional[dict[str, Any]] = None,
        version: str | None = None,
    ) -> None:
        # Prefer explicit input_schema; fall back to arg_schema for compatibility with FSD specs
        effective_input_schema: Optional[dict[str, Any]] = (
            input_schema if input_schema is not None else arg_schema
        )
        scope_key = scope or "default"
        versions = self._entries.setdefault(scope_key, {})
        version_key = version or "latest"
        scoped = versions.setdefault(id, {})

        scoped[version_key] = {
            "factory": factory,
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
    ) -> Optional[dict[str, Any]]:
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
            # Return the latest registered version by lexical order
            try:
                from packaging.version import Version

                latest_key = max(versions.keys(), key=Version)
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

    def get(self, skill_id: str, *, scope: str | None = None) -> Optional[dict[str, Any]]:
        return get_skill_registry().get(skill_id, scope=scope)


class SkillRegistryProvider(SkillRegistryProviderProtocol):
    """Provide scoped registries (per scope name)."""

    def __init__(self) -> None:
        self._registries: Dict[str, SkillRegistry] = {}

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
]
