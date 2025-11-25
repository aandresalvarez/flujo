"""
Tests for state_providers integration with Flujo runner.

This module verifies that:
1. Flujo accepts state_providers parameter
2. state_providers are passed through the factory chain
3. ContextReference is properly hydrated and persisted
4. Backward compatibility is maintained
"""

from __future__ import annotations

import pytest
from typing import Any, List

from pydantic import Field

from flujo import Flujo, Step
from flujo.domain.models import PipelineContext, ContextReference
from flujo.domain.interfaces import StateProvider
from flujo.application.core.factories import ExecutorFactory, BackendFactory


# --- Test Fixtures ---


class MockStateProvider(StateProvider):
    """Mock StateProvider for testing."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self.load_calls: list[str] = []
        self.save_calls: list[tuple[str, Any]] = []

    async def load(self, key: str) -> Any:
        self.load_calls.append(key)
        return self._data.get(key, [])

    async def save(self, key: str, data: Any) -> None:
        self.save_calls.append((key, data))
        self._data[key] = data


class ContextWithReference(PipelineContext):
    """Test context with a ContextReference field."""

    ref: ContextReference[List[int]] = Field(
        default_factory=lambda: ContextReference(provider_id="test_provider", key="test_key")
    )


# --- Unit Tests: Factory Chain ---


def test_executor_factory_accepts_state_providers() -> None:
    """Verify ExecutorFactory accepts state_providers in __init__."""
    provider = MockStateProvider()
    factory = ExecutorFactory(state_providers={"test_provider": provider})
    assert factory._state_providers == {"test_provider": provider}


def test_executor_factory_defaults_to_empty_dict() -> None:
    """Verify ExecutorFactory defaults to empty dict when no providers given."""
    factory = ExecutorFactory()
    assert factory._state_providers == {}


def test_executor_factory_passes_providers_to_executor() -> None:
    """Verify ExecutorFactory passes state_providers to ExecutorCore."""
    provider = MockStateProvider()
    factory = ExecutorFactory(state_providers={"test_provider": provider})
    executor = factory.create_executor()
    assert executor._state_providers == {"test_provider": provider}


def test_backend_factory_uses_executor_with_providers() -> None:
    """Verify BackendFactory creates backend with providers from ExecutorFactory."""
    provider = MockStateProvider()
    executor_factory = ExecutorFactory(state_providers={"test_provider": provider})
    backend_factory = BackendFactory(executor_factory)
    backend = backend_factory.create_execution_backend()
    assert backend._executor._state_providers == {"test_provider": provider}


# --- Unit Tests: Flujo Runner ---


def test_flujo_accepts_state_providers_parameter() -> None:
    """Verify Flujo accepts state_providers in __init__."""
    provider = MockStateProvider()
    step = Step.from_callable(lambda x: x, name="test_step")

    runner = Flujo(
        pipeline=step,
        context_model=ContextWithReference,
        state_providers={"test_provider": provider},
    )

    assert runner._state_providers == {"test_provider": provider}


def test_flujo_defaults_to_empty_providers() -> None:
    """Verify Flujo defaults to empty dict when no providers given."""
    step = Step.from_callable(lambda x: x, name="test_step")

    runner = Flujo(pipeline=step, context_model=PipelineContext)

    assert runner._state_providers == {}


def test_flujo_passes_providers_to_executor_factory() -> None:
    """Verify Flujo passes state_providers to ExecutorFactory."""
    provider = MockStateProvider()
    step = Step.from_callable(lambda x: x, name="test_step")

    runner = Flujo(
        pipeline=step,
        context_model=ContextWithReference,
        state_providers={"test_provider": provider},
    )

    # Check the executor factory received the providers
    assert runner._executor_factory._state_providers == {"test_provider": provider}


def test_flujo_respects_custom_executor_factory() -> None:
    """Verify Flujo uses custom executor_factory when provided."""
    provider = MockStateProvider()
    custom_factory = ExecutorFactory(state_providers={"custom_provider": provider})
    step = Step.from_callable(lambda x: x, name="test_step")

    runner = Flujo(
        pipeline=step,
        context_model=PipelineContext,
        executor_factory=custom_factory,
        state_providers={"ignored_provider": MockStateProvider()},  # Should be ignored
    )

    # Custom factory should be used
    assert runner._executor_factory is custom_factory
    assert runner._executor_factory._state_providers == {"custom_provider": provider}


def test_flujo_warns_when_executor_factory_and_state_providers() -> None:
    """Verify Flujo warns when both executor_factory and state_providers are provided."""
    import warnings

    provider = MockStateProvider()
    custom_factory = ExecutorFactory()  # No state_providers
    step = Step.from_callable(lambda x: x, name="test_step")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Flujo(
            pipeline=step,
            context_model=PipelineContext,
            executor_factory=custom_factory,
            state_providers={"provider": provider},  # Should trigger warning
        )

        # Check warning was issued
        assert len(w) == 1
        assert "state_providers is ignored when executor_factory is provided" in str(w[0].message)


def test_flujo_propagates_providers_with_custom_backend_factory() -> None:
    """Verify state_providers are propagated even when custom backend_factory is provided."""
    provider = MockStateProvider()
    custom_backend_factory = BackendFactory()  # Default factory, no executor
    step = Step.from_callable(lambda x: x, name="test_step")

    runner = Flujo(
        pipeline=step,
        context_model=ContextWithReference,
        backend_factory=custom_backend_factory,
        state_providers={"test_provider": provider},
    )

    # Our executor factory should have the state_providers
    assert runner._executor_factory._state_providers == {"test_provider": provider}

    # The backend should use our executor (with state_providers)
    # This is verified by checking the executor in the backend
    executor = runner.backend._executor
    assert executor._state_providers == {"test_provider": provider}


# --- Integration Tests: Context Hydration ---


@pytest.mark.asyncio
async def test_context_reference_hydration() -> None:
    """Verify ContextReference is hydrated from state_providers during execution."""
    provider = MockStateProvider()
    provider._data["test_key"] = [1, 2, 3]

    captured_value: List[int] = []

    async def check_hydration(data: str, *, context: ContextWithReference) -> str:
        # The context reference should be hydrated
        captured_value.extend(context.ref.get())
        return "done"

    step = Step.from_callable(check_hydration, name="check_step")

    runner = Flujo(
        pipeline=step,
        context_model=ContextWithReference,
        state_providers={"test_provider": provider},
    )

    async with runner:
        result = await runner.run_async("test_input")

    assert result.success
    assert captured_value == [1, 2, 3]
    assert "test_key" in provider.load_calls


@pytest.mark.asyncio
async def test_context_reference_persistence() -> None:
    """Verify ContextReference changes are persisted via state_providers."""
    provider = MockStateProvider()
    provider._data["test_key"] = [1, 2, 3]

    async def modify_context(data: str, *, context: ContextWithReference) -> str:
        current = context.ref.get()
        current.append(4)
        context.ref.set(current)
        return "modified"

    step = Step.from_callable(modify_context, name="modify_step")

    runner = Flujo(
        pipeline=step,
        context_model=ContextWithReference,
        state_providers={"test_provider": provider},
    )

    async with runner:
        result = await runner.run_async("test_input")

    assert result.success
    # Check that save was called
    assert len(provider.save_calls) > 0
    # The last saved value should include the new element
    saved_values = [call[1] for call in provider.save_calls if call[0] == "test_key"]
    assert any(4 in val for val in saved_values if isinstance(val, list))


# --- Backward Compatibility Tests ---


@pytest.mark.asyncio
async def test_backward_compatibility_no_providers() -> None:
    """Verify existing code without state_providers still works."""

    async def simple_step(data: str) -> str:
        return f"processed: {data}"

    step = Step.from_callable(simple_step, name="simple_step")

    runner = Flujo(pipeline=step, context_model=PipelineContext)

    async with runner:
        result = await runner.run_async("input")

    assert result.success
    assert result.output == "processed: input"


@pytest.mark.asyncio
async def test_backward_compatibility_with_resources() -> None:
    """Verify state_providers works alongside resources (doesn't break when both are provided)."""
    from flujo.domain.resources import AppResources

    class MyTestResources(AppResources):
        value: int = 42

    provider = MockStateProvider()
    resources = MyTestResources()

    async def simple_step(data: str) -> str:
        return f"processed: {data}"

    step = Step.from_callable(simple_step, name="resource_step")

    # Verify runner accepts both resources and state_providers
    runner = Flujo(
        pipeline=step,
        context_model=PipelineContext,
        resources=resources,
        state_providers={"test_provider": provider},
    )

    async with runner:
        result = await runner.run_async("input")

    assert result.success
    assert result.output == "processed: input"
    # Both are stored
    assert runner.resources is resources
    assert runner._state_providers == {"test_provider": provider}


# --- Edge Cases ---


def test_multiple_providers() -> None:
    """Verify multiple providers can be registered."""
    provider1 = MockStateProvider()
    provider2 = MockStateProvider()
    step = Step.from_callable(lambda x: x, name="test_step")

    runner = Flujo(
        pipeline=step,
        context_model=PipelineContext,
        state_providers={"provider1": provider1, "provider2": provider2},
    )

    assert len(runner._state_providers) == 2
    assert runner._state_providers["provider1"] is provider1
    assert runner._state_providers["provider2"] is provider2


def test_none_state_providers_treated_as_empty() -> None:
    """Verify None state_providers is treated as empty dict."""
    step = Step.from_callable(lambda x: x, name="test_step")

    runner = Flujo(pipeline=step, context_model=PipelineContext, state_providers=None)

    assert runner._state_providers == {}
