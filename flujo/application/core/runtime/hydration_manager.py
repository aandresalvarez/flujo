"""StateProvider hydration and persistence management."""

from __future__ import annotations
import asyncio
from pydantic import BaseModel

from ....domain.models import ContextReference
from ....domain.interfaces import StateProvider


class HydrationManager:
    """Manages hydration and persistence of ContextReference fields using StateProviders."""

    def __init__(self, state_providers: dict[str, StateProvider[object]] | None = None) -> None:
        self._state_providers: dict[str, StateProvider[object]] = state_providers or {}
        self._telemetry: object | None = None

    def set_telemetry(self, telemetry: object) -> None:
        """Set telemetry instance for logging warnings."""
        self._telemetry = telemetry

    def add_state_provider(self, provider_id: str, provider: StateProvider[object]) -> None:
        """Add a state provider."""
        self._state_providers[provider_id] = provider

    def get_state_provider(self, provider_id: str) -> StateProvider[object] | None:
        """Get a state provider by ID."""
        return self._state_providers.get(provider_id)

    def remove_state_provider(self, provider_id: str) -> None:
        """Remove a state provider."""
        self._state_providers.pop(provider_id, None)

    async def hydrate_context(self, context: object | None) -> None:
        """Hydrate ContextReference fields in the context using registered providers."""
        if context is None or not self._state_providers:
            return

        # Iterate over fields to find ContextReference
        # We check top-level fields of Pydantic models
        if isinstance(context, BaseModel):
            tasks = []
            refs: list[tuple[str, ContextReference[object], StateProvider[object]]] = []
            for field_name, field_value in context:
                if isinstance(field_value, ContextReference):
                    provider = self._state_providers.get(field_value.provider_id)
                    if provider:
                        refs.append((field_name, field_value, provider))
            for field_name, ref, provider in refs:

                async def _load_and_set(
                    fn: str, r: ContextReference[object], p: StateProvider[object]
                ) -> None:
                    try:
                        data = await p.load(r.key)
                        if data is not None:
                            r.set(data)
                    except Exception as e:
                        if self._telemetry:
                            warning_fn = getattr(self._telemetry, "warning", None)
                            if callable(warning_fn):
                                warning_fn(f"Failed to hydrate reference {fn}: {e}")

                tasks.append(_load_and_set(field_name, ref, provider))
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def persist_context(self, context: object | None) -> None:
        """Persist ContextReference fields in the context using registered providers."""
        if context is None or not self._state_providers:
            return

        if isinstance(context, BaseModel):
            for field_name, field_value in context:
                if isinstance(field_value, ContextReference):
                    # Only save if value is present (hydrated/modified)
                    # Accessing _value directly to avoid ValueError from get()
                    if field_value._value is not None:
                        provider = self._state_providers.get(field_value.provider_id)
                        if provider:
                            try:
                                await provider.save(field_value.key, field_value._value)
                            except Exception as e:
                                if self._telemetry:
                                    warning_fn = getattr(self._telemetry, "warning", None)
                                    if callable(warning_fn):
                                        warning_fn(f"Failed to persist reference {field_name}: {e}")

    def get_state_providers(self) -> dict[str, StateProvider[object]]:
        """Get all registered state providers."""
        return self._state_providers.copy()

    def clear_state_providers(self) -> None:
        """Clear all registered state providers."""
        self._state_providers.clear()
