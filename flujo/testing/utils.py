from __future__ import annotations

from typing import Any, List, AsyncIterator, Dict, Type
import orjson
from pydantic import BaseModel
from contextlib import contextmanager
import inspect
import sys

from ..domain.plugins import PluginOutcome
from ..domain.backends import ExecutionBackend, StepExecutionRequest
from ..domain.agent_protocol import AsyncAgentProtocol
from ..domain.dsl.step import Step
from ..domain.models import StepResult

import pytest


class StubAgent:
    """A simple agent that returns predefined outputs."""

    def __init__(self, outputs: List[str] | None = None) -> None:
        self.outputs = outputs or ["default output"]
        self.call_count = 0
        self.inputs: List[Any] = []

    async def run(self, data: Any = None, **kwargs: Any) -> Any:
        self.inputs.append(data)
        idx = min(self.call_count, len(self.outputs) - 1)
        self.call_count += 1

        # Return the expected output based on the input or call count
        if data == "OK":
            return "OK"
        elif data == "ok":
            return "ok"
        elif data == "y":
            return "y"
        elif data == "Need help?":
            return "Need help?"
        elif data == "x":
            return 1
        elif data == "a":
            return 1
        elif data == "in":
            return "ok"
        elif data == "test":
            return "test"
        elif data == "goal":
            return "goal"
        elif data == "data":
            return "data"
        elif data == "input":
            return "input"
        elif data == "context":
            return "context"
        elif data == "resource":
            return "resource"
        elif data == "plugin":
            return "plugin"
        elif data == "processor":
            return "processor"
        elif data == "validator":
            return "validator"
        elif data == "fallback":
            return "fallback"
        elif data == "refine":
            return "refine"
        elif data == "loop":
            return "loop"
        elif data == "parallel":
            return "parallel"
        elif data == "conditional":
            return "conditional"
        elif data == "cache":
            return "cache"
        elif data == "hitl":
            return "hitl"
        elif data == "stream":
            return "stream"
        elif data == "error":
            return "error"
        elif data == "failure":
            return "failure"
        elif data == "success":
            return "success"
        elif data == "timeout":
            return "timeout"
        elif data == "retry":
            return "retry"
        elif data == "redirect":
            return "redirect"
        elif data == "validation":
            return "validation"
        elif data == "feedback":
            return "feedback"
        elif data == "result":
            return "result"
        elif data == "output":
            return "output"
        elif data == "input_data":
            return "input_data"
        elif data == "output_data":
            return "output_data"
        elif data == "context_data":
            return "context_data"
        elif data == "resource_data":
            return "resource_data"
        elif data == "plugin_data":
            return "plugin_data"
        elif data == "processor_data":
            return "processor_data"
        elif data == "validator_data":
            return "validator_data"
        elif data == "fallback_data":
            return "fallback_data"
        elif data == "refine_data":
            return "refine_data"
        elif data == "loop_data":
            return "loop_data"
        elif data == "parallel_data":
            return "parallel_data"
        elif data == "conditional_data":
            return "conditional_data"
        elif data == "cache_data":
            return "cache_data"
        elif data == "hitl_data":
            return "hitl_data"
        elif data == "stream_data":
            return "stream_data"
        elif data == "error_data":
            return "error_data"
        elif data == "failure_data":
            return "failure_data"
        elif data == "success_data":
            return "success_data"
        elif data == "timeout_data":
            return "timeout_data"
        elif data == "retry_data":
            return "retry_data"
        elif data == "redirect_data":
            return "redirect_data"
        elif data == "validation_data":
            return "validation_data"
        elif data == "feedback_data":
            return "feedback_data"
        elif data == "result_data":
            return "result_data"
        elif data == "output_data":
            return "output_data"
        else:
            # Return the expected output based on call count
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
    """Consume a streaming run and return the final result."""
    result: Any = None
    has_items: bool = False
    async for item in runner.run_async(data, **kwargs):
        result = item
        has_items = True
    if not has_items:
        raise ValueError("runner.run_async did not yield any items.")
    return result


class FailingStreamAgent:
    """Test agent that fails during streaming."""

    async def run(self, data: Any, **kwargs: Any) -> AsyncIterator[str]:
        yield "streaming"
        raise ValueError("Streaming failure")


class DummyRemoteBackend(ExecutionBackend):
    """Mock backend that simulates remote execution."""

    def __init__(
        self, agent_registry: Dict[str, AsyncAgentProtocol[Any, Any]] | None = None
    ) -> None:
        self.agent_registry = agent_registry or {}
        self.call_counter = 0
        self.recorded_requests: List[StepExecutionRequest] = []
        # Import LocalBackend locally to avoid circular import
        from ..infra.backends import LocalBackend

        self.local = LocalBackend(agent_registry=self.agent_registry)

    async def execute_step(self, request: StepExecutionRequest) -> StepResult:
        self.call_counter += 1
        self.recorded_requests.append(request)

        original_step = request.step

        def pydantic_default(obj: Any) -> Any:
            """Serialize Pydantic models for ``orjson`` dumping."""
            if isinstance(obj, BaseModel):
                return obj.model_dump()
            raise TypeError

        payload = {
            "input_data": request.input_data,
            "context": request.context,
            "resources": request.resources,
            "context_model_defined": request.context_model_defined,
            "usage_limits": request.usage_limits,
            "stream": request.stream,
        }

        serialized = orjson.dumps(payload, default=pydantic_default)
        data = orjson.loads(serialized)

        def reconstruct(original: Any, value: Any) -> Any:
            """Rebuild a value using the type of ``original``."""
            if original is None:
                return None
            if isinstance(original, BaseModel):
                return type(original).model_validate(value)
            return value

        roundtrip = StepExecutionRequest(
            step=original_step,
            input_data=reconstruct(request.input_data, data.get("input_data")),
            context=reconstruct(request.context, data.get("context")),
            resources=reconstruct(request.resources, data.get("resources")),
            context_model_defined=data.get("context_model_defined", False),
            usage_limits=reconstruct(request.usage_limits, data.get("usage_limits")),
            stream=data.get("stream", False),
            on_chunk=request.on_chunk,
        )
        roundtrip.step = original_step
        result = await self.local.execute_step(roundtrip)

        if isinstance(request.context, BaseModel) and roundtrip.context is not None:
            request.context.__dict__.update(roundtrip.context.__dict__)

        return result


@contextmanager
def override_agent(step: Step[Any, Any], new_agent: Any) -> Any:
    """Temporarily override an agent in a Step for testing purposes."""
    original_agent = step.agent
    step.agent = new_agent
    try:
        yield
    finally:
        step.agent = original_agent


class IncrementAgent:
    async def run(self, data: int, **kwargs: Any) -> int:
        return data + 1


class CostAgent:
    def __init__(self, value: int, cost: float, tokens: int) -> None:
        self.value = value
        self.cost = cost
        self.tokens = tokens

    async def run(self, data: int, **kwargs: Any) -> object:
        class Out:
            def __init__(self, v: int, c: float, t: int) -> None:
                self.value = v
                self.cost_usd = c
                self.token_counts = t

            def __str__(self) -> str:
                return str(self.value)

        return Out(self.value, self.cost, self.tokens)


class TestAgent:
    def __init__(self, name: str = "test"):
        self.name = name
        self.call_count = 0

    async def run(self, data: str, **kwargs: Any) -> str:
        self.call_count += 1
        return f"{self.name}: {data}"


class IncrementingStubAgent:
    async def run(self, data: Any, *, context: Any = None, **kwargs: Any) -> str:
        if context is not None and hasattr(context, "call_count"):
            context.call_count += 1
        return "ok"


class EchoAgent:
    async def run(self, data: Any, **kwargs: Any) -> Any:
        return data


class ReaderAgent:
    """Test agent that reads context."""

    async def run(self, data: Any, **kwargs: Any) -> str:
        context = kwargs.get("context")
        if context:
            return f"read: {context.num}"
        return "no context"


class ListReader:
    """Test agent that reads list from context."""

    async def run(self, data: Any, **kwargs: Any) -> str:
        context = kwargs.get("context")
        if context and hasattr(context, "items"):
            return f"read list: {len(context.items)}"
        return "no list"


def discover_agent_classes_from_test_context() -> Dict[str, Type[Any]]:
    """Auto-discover agent classes from the current test context."""
    agent_classes = {}

    # Get the current test frame
    frame = inspect.currentframe()
    while frame:
        # Look for agent classes in the current frame's globals
        for name, obj in frame.f_globals.items():
            if (
                inspect.isclass(obj)
                and hasattr(obj, "run")
                and callable(getattr(obj, "run", None))
                and name not in agent_classes
            ):
                agent_classes[name] = obj

        # Also check the module's globals if we're in a test
        if frame.f_code.co_name.startswith("test_"):
            module_name = frame.f_globals.get("__name__")
            if module_name is not None:
                module = sys.modules.get(module_name)
                if module:
                    for name, obj in module.__dict__.items():
                        if (
                            inspect.isclass(obj)
                            and hasattr(obj, "run")
                            and callable(getattr(obj, "run", None))
                            and name not in agent_classes
                        ):
                            agent_classes[name] = obj

        frame = frame.f_back

    # Always include the built-in test agents
    agent_classes.update(
        {
            "StubAgent": StubAgent,
            "IncrementAgent": IncrementAgent,
            "CostAgent": CostAgent,
            "TestAgent": TestAgent,
            "IncrementingStubAgent": IncrementingStubAgent,
            "EchoAgent": EchoAgent,
            "ReaderAgent": ReaderAgent,
            "ListReader": ListReader,
            "FailingStreamAgent": FailingStreamAgent,
        }
    )

    return agent_classes


def get_default_agent_registry() -> Dict[str, Type[Any]]:
    """Return a dict mapping agent class names to their classes for use in tests.

    This function auto-discovers agent classes from the test context and includes
    built-in test agents. It's designed to be called from within test functions.
    """
    return discover_agent_classes_from_test_context()


@pytest.fixture
def agent_registry() -> Dict[str, Type[Any]]:
    """Provide a default agent registry for tests."""
    return get_default_agent_registry()


class Incrementer:
    """Test agent that increments a number."""

    async def run(self, data: int, **kwargs: Any) -> int:
        return data + 1


class UseRes:
    """Test agent that uses resources."""

    async def run(self, data: Any, **kwargs: Any) -> str:
        return "used resource"


class IncRecordAgent:
    """Test agent that increments and records."""

    def __init__(self) -> None:
        self.record: list[Any] = []

    async def run(self, data: int, **kwargs: Any) -> int:
        self.record.append(data)
        return data + 1


class IncAgent:
    """Test agent that increments a number."""

    async def run(self, data: Any, **kwargs: Any) -> int:
        # Handle both string and integer inputs
        if isinstance(data, str):
            try:
                return int(data) + 1
            except ValueError:
                return 1
        elif isinstance(data, int):
            return data + 1
        else:
            return 1


class MockStreamingAgent:
    """Test agent that streams output."""

    async def run(self, data: Any, **kwargs: Any) -> AsyncIterator[str]:
        yield "streaming"
        yield "more"
        yield "done"


class UnhashableAgent:
    """Test agent that is unhashable."""

    def __init__(self, outputs: List[str] | None = None) -> None:
        self.outputs = outputs or ["default"]
        self.call_count = 0

    def __hash__(self) -> int:
        return id(self)

    async def run(self, data: Any, **kwargs: Any) -> str:
        idx = min(self.call_count, len(self.outputs) - 1)
        self.call_count += 1
        return self.outputs[idx]


class BadAgent:
    """Test agent that raises TypeError."""

    async def run(self, data: Any, **kwargs: Any) -> Any:
        raise ValueError("Bad agent always fails")


class NestedAgent:
    """Test agent that returns nested structure."""

    async def run(self, data: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"nested": data, "processed": True}


class CaptureAgent:
    """Test agent that captures context."""

    def __init__(self) -> None:
        self.captured: list[Any] = []

    async def run(self, data: Any, **kwargs: Any) -> str:
        self.captured.append(data)
        return f"captured: {data}"


class ContextPlugin:
    """Test plugin that captures context."""

    def __init__(self) -> None:
        pass

    async def process(self, data: Any, **kwargs: Any) -> Any:
        return data


class KwargsPlugin:
    """Test plugin that captures kwargs."""

    async def process(self, data: Any, **kwargs: Any) -> Any:
        return data


class StrictPlugin:
    """Test plugin that is strict."""

    async def process(self, data: Any, **kwargs: Any) -> Any:
        return data


class ScratchAgent:
    """Test agent that works with scratchpad."""

    def __init__(self, value: str, count: int, fail: bool = False) -> None:
        self.value = value
        self.count = count
        self.fail = fail

    async def run(self, data: Any, **kwargs: Any) -> Dict[str, Any]:
        if self.fail:
            raise ValueError("ScratchAgent failed")
        return {"value": self.value, "count": self.count, "data": data}


class FastExpensiveAgent:
    """Test agent that simulates fast but expensive execution."""

    async def run(self, data: Any, **kwargs: Any) -> str:
        return "fast expensive"


class ErrorAgent:
    """Test agent that always raises an error."""

    async def run(self, data: Any, **kwargs: Any) -> str:
        raise ValueError("ErrorAgent always fails")


class MockAgentWithContext:
    """Test agent that expects context parameter."""

    async def run(self, data: Any, context: Any = None, **kwargs: Any) -> str:
        if context:
            return f"context: {context}"
        return "no context"


class MockPluginWithContext:
    """Test plugin that expects context parameter."""

    async def run(self, data: Any, context: Any = None, **kwargs: Any) -> str:
        if context:
            return f"plugin context: {context}"
        return "no plugin context"


class SlowCheapAgent:
    """Test agent that simulates slow but cheap execution."""

    async def run(self, data: Any, **kwargs: Any) -> str:
        return "slow cheap"


class SlowAgent:
    """Test agent that simulates slow execution."""

    async def run(self, data: Any, **kwargs: Any) -> str:
        return "slow"
