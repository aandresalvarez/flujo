"""Tests for mock output handling in pipelines.

After the mock detection logic was purged (commit b9f65b73), these tests
verify that agents can return various output types without restriction.
Mock objects are now allowed as they may be valid in testing scenarios.
"""

import pytest
from unittest.mock import AsyncMock, Mock

from flujo.domain.dsl import Step
from flujo.testing.utils import StubAgent, gather_result
from tests.conftest import create_test_flujo


@pytest.fixture
def disable_executor_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force disable cache to avoid serializing Mock objects."""
    import flujo.application.core.executor_core as exec_core

    original_init = exec_core.ExecutorCore.__init__

    def mocked_init(self, *args, **kwargs):
        # Force cache disabled in constructor
        kwargs["enable_cache"] = False
        original_init(self, *args, **kwargs)
        # Ensure the attribute is set to False (in case original_init logic overrides it)
        self._enable_cache = False
        # Also disable on cache manager if present
        if hasattr(self, "_cache_manager"):
            try:
                self._cache_manager._enabled = False
            except Exception:
                pass

    monkeypatch.setattr(exec_core.ExecutorCore, "__init__", mocked_init)


@pytest.mark.asyncio
async def test_concrete_value_passes() -> None:
    """Concrete values pass through normally."""
    step = Step.model_validate({"name": "s", "agent": StubAgent(["ok"])})
    runner = create_test_flujo(step)
    result = await gather_result(runner, "in")
    history = result.step_history[0]
    assert history.success is True
    assert history.output == "ok"


@pytest.mark.asyncio
async def test_mock_output_allowed(disable_executor_cache) -> None:
    """Mock objects are allowed as output (post mock-detection removal).

    After commit b9f65b73, mock detection was removed because:
    - It was overly restrictive in testing scenarios
    - Mock objects can be valid sentinel values
    - Users can use type validation if they need strict output checking
    """

    class MockReturningAgent:
        async def run(self, *_args, **_kwargs):
            return Mock()

    agent = MockReturningAgent()
    step = Step.model_validate({"name": "s", "agent": agent})
    # Disable persistence to avoid serialization errors with Mock objects
    runner = create_test_flujo(step, persist_state=False)

    # Should complete without error - mock detection no longer active
    result = await gather_result(runner, "in")
    history = result.step_history[0]
    assert history.success is True
    assert isinstance(history.output, Mock)


@pytest.mark.asyncio
async def test_nested_mock_in_dict_allowed(disable_executor_cache) -> None:
    """Mocks nested in data structures are allowed."""
    nested = Mock()

    class NestedAgent:
        async def run(self, *_args, **_kwargs):
            return {"data": nested}

    agent = NestedAgent()
    step = Step.model_validate({"name": "s", "agent": agent})
    # Disable persistence to avoid serialization errors with Mock objects
    runner = create_test_flujo(step, persist_state=False)
    result = await gather_result(runner, "in")
    history = result.step_history[0]
    assert history.success is True
    assert history.output["data"] is nested


@pytest.mark.asyncio
async def test_pipeline_continues_with_mock_output(disable_executor_cache) -> None:
    """Pipeline continues execution even when steps return Mock objects."""
    good_agent = StubAgent(["ok"])

    class MockAgent:
        def __init__(self):
            self.run = AsyncMock(return_value=Mock(value="test"))

    mock_agent = MockAgent()
    final_agent = StubAgent(["end"])

    step1 = Step.model_validate({"name": "a", "agent": good_agent})
    step2 = Step.model_validate({"name": "b", "agent": mock_agent})
    step3 = Step.model_validate({"name": "c", "agent": final_agent})
    pipeline = step1 >> step2 >> step3
    # Disable persistence to avoid serialization errors with Mock objects
    runner = create_test_flujo(pipeline, persist_state=False)

    # Pipeline completes - no mock detection
    result = await gather_result(runner, "start")

    assert len(result.step_history) == 3
    assert good_agent.call_count == 1
    assert mock_agent.run.call_count == 1
    assert final_agent.call_count == 1
