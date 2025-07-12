from __future__ import annotations

from typing import Any, List, AsyncIterator, Dict, Iterator
import asyncio
import json
from pydantic import BaseModel
from contextlib import contextmanager

from ..domain.plugins import PluginOutcome
from ..domain.backends import ExecutionBackend, StepExecutionRequest
from ..domain.agent_protocol import AsyncAgentProtocol
from ..infra.backends import LocalBackend
from ..domain.resources import AppResources
from ..domain.models import StepResult, UsageLimits, BaseModel as FlujoBaseModel


class StubAgent:
    """Simple agent for testing that returns preset outputs."""

    def __init__(self, outputs: List[Any]):
        self.outputs = outputs
        self.call_count = 0
        self.inputs: List[Any] = []

    async def run(self, data: Any = None, **_: Any) -> Any:
        self.inputs.append(data)
        idx = min(self.call_count, len(self.outputs) - 1)
        self.call_count += 1
        return self.outputs[idx]

    async def run_async(self, data: Any = None, **kwargs: Any) -> Any:
        return await self.run(data, **kwargs)


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
    """Gather all results from a runner into a single result."""
    results = []
    async for item in runner.run_async(data, **kwargs):
        results.append(item)
    return results[-1] if results else None


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


class DummyRemoteBackend(ExecutionBackend):
    """Mock backend that simulates remote execution."""

    def __init__(
        self, agent_registry: Dict[str, AsyncAgentProtocol[Any, Any]] | None = None
    ) -> None:
        self.agent_registry = agent_registry or {}
        self.call_counter = 0
        self.recorded_requests: List[StepExecutionRequest] = []
        self.local = LocalBackend(agent_registry=self.agent_registry)

    async def execute_step(self, request: StepExecutionRequest) -> StepResult:
        self.call_counter += 1
        self.recorded_requests.append(request)

        original_step = request.step

        payload = {
            "input_data": request.input_data,
            "context": request.context,
            "resources": request.resources,
            "context_model_defined": request.context_model_defined,
            "usage_limits": request.usage_limits,
            "stream": request.stream,
        }

        # Use robust serialization for nested structures
        def robust_serialize(obj: Any) -> Any:
            if obj is None:
                return None
            if isinstance(obj, BaseModel):
                return {k: robust_serialize(v) for k, v in obj.model_dump(mode="json").items()}
            if isinstance(obj, dict):
                return {k: robust_serialize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [robust_serialize(v) for v in obj]
            if isinstance(obj, (str, int, float, bool)):
                return obj
            return str(obj)  # fallback for unknown types

        serialized = robust_serialize(payload)
        data = json.loads(json.dumps(serialized))

        def reconstruct(original: Any, value: Any) -> Any:
            """Rebuild a value using the type of ``original``."""
            if original is None:
                return None
            if isinstance(original, BaseModel):
                if isinstance(value, dict):
                    fixed_value = {
                        k: reconstruct(getattr(original, k, None), v) for k, v in value.items()
                    }
                    return type(original).model_validate(fixed_value)
                else:
                    return type(original).model_validate(value)
            elif isinstance(original, (list, tuple)):
                if isinstance(value, (list, tuple)):
                    if not original:
                        return list(value)
                    return type(original)(reconstruct(original[0], v) for v in value)
                else:
                    return original
            elif isinstance(original, dict):
                if isinstance(value, dict):
                    return {k: reconstruct(original.get(k), v) for k, v in value.items()}
                else:
                    return original
            else:
                return value

        # Reconstruct the payload with proper types
        reconstructed_payload = {}
        for key, original_value in payload.items():
            if key in data:
                if key == "context" and isinstance(original_value, BaseModel):
                    reconstructed_payload[key] = original_value
                else:
                    reconstructed_payload[key] = reconstruct(original_value, data[key])
            else:
                reconstructed_payload[key] = original_value

        # Create a new request with reconstructed data
        reconstructed_request = StepExecutionRequest(
            step=original_step,
            input_data=reconstructed_payload["input_data"],
            context=reconstructed_payload["context"]
            if isinstance(reconstructed_payload["context"], FlujoBaseModel)
            or reconstructed_payload["context"] is None
            else None,
            resources=reconstructed_payload["resources"]
            if isinstance(reconstructed_payload["resources"], AppResources)
            or reconstructed_payload["resources"] is None
            else None,
            context_model_defined=bool(reconstructed_payload["context_model_defined"]),
            usage_limits=reconstructed_payload["usage_limits"]
            if isinstance(reconstructed_payload["usage_limits"], UsageLimits)
            or reconstructed_payload["usage_limits"] is None
            else None,
            stream=bool(reconstructed_payload["stream"]),
        )

        return await self.local.execute_step(reconstructed_request)


@contextmanager
def override_agent(step: Any, new_agent: Any) -> Iterator[None]:
    """Temporarily override the agent of a Step within a context."""
    original_agent = getattr(step, "agent", None)
    step.agent = new_agent
    try:
        yield
    finally:
        step.agent = original_agent
