import pytest
from typing import Any
from flujo.domain.models import ContextReference, PipelineContext, StepResult
from flujo.application.core.executor_core import ExecutorCore
from flujo.domain.dsl.step import Step


class MockStateProvider:
    def __init__(self):
        self.store = {"key1": "initial_value"}
        self.load_calls = 0
        self.save_calls = 0

    async def load(self, key: str) -> Any:
        self.load_calls += 1
        return self.store.get(key)

    async def save(self, key: str, data: Any) -> None:
        self.save_calls += 1
        self.store[key] = data


class MyContext(PipelineContext):
    ref: ContextReference[str]


class MyStep(Step):
    name: str = "my_step"

    async def execute(self, context: MyContext, **kwargs) -> StepResult:
        # Verify data is hydrated
        val = context.ref.get()
        assert val == "initial_value"
        # Modify data
        context.ref.set("modified_value")
        return StepResult(name=self.name, success=True, output="done")


@pytest.mark.asyncio
async def test_managed_state_lifecycle():
    provider = MockStateProvider()
    providers = {"mock_provider": provider}

    executor = ExecutorCore(state_providers=providers)

    # Initialize context with reference
    ref = ContextReference[str](provider_id="mock_provider", key="key1")
    context = MyContext(ref=ref)

    # Verify serialization works (no private attr)
    json_str = context.model_dump_json()
    assert "initial_value" not in json_str
    assert "mock_provider" in json_str

    step = MyStep()

    # Register policy for MyStep
    async def my_policy(frame):
        return await frame.step.execute(frame.context)

    executor.policy_registry.register(MyStep, my_policy)

    # Run execution
    result = await executor.execute(step=step, data=None, context=context)

    assert result.success
    assert provider.load_calls >= 1
    assert provider.save_calls >= 1
    assert provider.store["key1"] == "modified_value"

    # Verify context reference has new value in memory
    assert context.ref.get() == "modified_value"


@pytest.mark.asyncio
async def test_managed_state_missing_provider():
    executor = ExecutorCore(state_providers={})
    ref = ContextReference[str](provider_id="missing", key="key1")
    context = MyContext(ref=ref)
    step = MyStep()

    # Register policy for MyStep
    async def my_policy(frame):
        return await frame.step.execute(frame.context)

    executor.policy_registry.register(MyStep, my_policy)

    # Should fail inside step because get() raises ValueError
    result = await executor.execute(step=step, data=None, context=context)

    assert not result.success
    assert "State not hydrated" in (result.feedback or str(result.error))
