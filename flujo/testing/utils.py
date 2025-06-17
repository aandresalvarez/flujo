from __future__ import annotations

from typing import Any, List, AsyncIterator, Dict, Optional
import asyncio

from ..domain.plugins import PluginOutcome


class StubAgent:
    """Simple agent for testing that returns preset outputs."""

    def __init__(self, outputs: List[Any]):
        self.outputs = outputs
        self.call_count = 0
        self.inputs: List[Any] = []

    async def run(self, input_data: Any = None, **_: Any) -> Any:
        self.inputs.append(input_data)
        idx = min(self.call_count, len(self.outputs) - 1)
        self.call_count += 1
        return self.outputs[idx]


class DummyPlugin:
    """A validation plugin used for testing."""

    def __init__(self, outcomes: List[PluginOutcome]):
        self.outcomes = outcomes
        self.call_count = 0

    async def validate(self, data: dict[str, Any]) -> PluginOutcome:
        idx = min(self.call_count, len(self.outcomes) - 1)
        self.call_count += 1
        return self.outcomes[idx]


async def gather_result(runner: Any, data: Any, **kwargs: Any) -> Any:
    """Consume a streaming run and return the final result."""
    result = None
    has_items = False
    async for item in runner.run_async(data, **kwargs):
        result = item
        has_items = True
    if not has_items:
        raise ValueError("runner.run_async did not yield any items.")
    return result


class FailingStreamAgent:
    """Agent that yields a few chunks then raises an exception."""

    def __init__(self, chunks: List[str], exc: Exception) -> None:
        self.chunks = chunks
        self.exc = exc

    async def stream(self, data: Any, **kwargs: Any) -> AsyncIterator[str]:
        for ch in self.chunks:
            await asyncio.sleep(0)
            yield ch
        raise self.exc


from ..domain.backends import ExecutionBackend, StepExecutionRequest
from ..domain.agent_protocol import AsyncAgentProtocol
from ..infra.backends import LocalBackend
from ..domain.models import StepResult


class DummyRemoteBackend(ExecutionBackend):
    """Mock backend that simulates remote execution."""

    def __init__(self, agent_registry: Dict[str, AsyncAgentProtocol[Any, Any]] | None = None) -> None:
        self.agent_registry = agent_registry or {}
        self.call_counter = 0
        self.recorded_requests: List[StepExecutionRequest] = []
        self.local = LocalBackend(agent_registry=self.agent_registry)

    async def execute_step(self, request: StepExecutionRequest) -> StepResult:
        self.call_counter += 1
        self.recorded_requests.append(request)
        data_dict = request.model_dump()
        roundtrip = StepExecutionRequest.model_validate(data_dict)
        # Preserve original step object to avoid pydantic resetting config
        roundtrip.step = request.step
        # Ensure usage_limits propagate to the underlying LocalBackend
        roundtrip.usage_limits = request.usage_limits
        return await self.local.execute_step(roundtrip)
