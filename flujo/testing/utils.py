"""Testing utilities for Flujo."""

from __future__ import annotations
import asyncio
import json
from typing import Any, Dict, List, AsyncIterator, Iterator, Optional
from contextlib import contextmanager
import ast

from flujo.domain.plugins import PluginOutcome, ValidationPlugin
from flujo.domain.backends import ExecutionBackend, StepExecutionRequest
from flujo.domain.agent_protocol import AsyncAgentProtocol
from flujo.infra.backends import LocalBackend
from flujo.domain.resources import AppResources
from flujo.domain.models import StepResult, UsageLimits, BaseModel as FlujoBaseModel
from flujo.domain.models import StepOutcome, Success, Failure, Paused, PipelineResult
from flujo.type_definitions.common import JSONObject
from flujo.exceptions import PausedException


def _serialize_for_test(obj: Any) -> Any:
    """Serialize an object for testing purposes using native Pydantic/JSON.

    Uses model_dump(mode=\"json\") for Pydantic models and handles primitives natively.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except TypeError:
            return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _serialize_for_test(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_for_test(item) for item in obj]
    # Fallback for unknown types
    try:
        # Test if JSON-serializable
        json.dumps(obj)
    except (TypeError, ValueError):
        return str(obj)
    else:
        return obj


def assert_pipeline_result(result: Any, expected_output: Optional[Any] = None) -> None:
    """Assert that a pipeline result was successful and optionally check output.

    Args:
        result: The pipeline result to check
        expected_output: Optional expected output value to verify

    Raises:
        AssertionError: If the pipeline failed or output doesn't match
    """
    # Check if all steps were successful
    if not hasattr(result, "step_history") or not result.step_history:
        raise AssertionError("Pipeline result has no step history")
    for step in result.step_history:
        if not getattr(step, "success", True):
            raise AssertionError(f"Step {getattr(step, 'name', '?')} failed")
    # Check output if provided
    if expected_output is not None:
        # Use the output of the last step as the final output
        final_output = result.step_history[-1].output
        if final_output != expected_output:
            raise AssertionError(f"Expected output {expected_output}, got {final_output}")


class StubAgent:
    """Simple agent for testing that returns preset outputs."""

    def __init__(self, outputs: List[Any]):
        if not isinstance(outputs, list):
            raise TypeError("outputs must be a list")
        self.outputs = outputs
        self.call_count = 0
        self.inputs: List[Any] = []

    async def run(self, data: Any = None, **_: Any) -> Any:
        self.inputs.append(data)
        self.call_count += 1  # Always increment call_count
        if self.call_count > len(self.outputs):
            raise IndexError("No more outputs available")
        return self.outputs[self.call_count - 1]

    async def run_async(self, data: Any = None, **kwargs: Any) -> Any:
        return await self.run(data, **kwargs)


class DummyPlugin(ValidationPlugin):
    """A validation plugin used for testing."""

    def __init__(self, outcomes: List[PluginOutcome]):
        self.outcomes = outcomes
        self.call_count = 0

    async def validate(self, data: dict[str, Any], *, context: Any = None) -> PluginOutcome:
        idx = min(self.call_count, len(self.outcomes) - 1)
        self.call_count += 1
        return self.outcomes[idx]


async def gather_result(runner: Any, data: Any, **kwargs: Any) -> Any:
    """Gather all results from a runner into a single result."""
    # Always use run_async for pipeline runners and return the last StepOutcome
    results: List[Any] = []
    async for result in runner.run_async(data, **kwargs):
        results.append(result)
    if not results:
        return Success(step_result=StepResult(name="<no-steps>"))
    last = results[-1]
    if isinstance(last, StepOutcome):
        return last
    # Pipeline runs often yield a final PipelineResult; return it directly
    if isinstance(last, PipelineResult):
        return last
    # Legacy StepResult: wrap into Success
    return Success(step_result=last)


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


class DummyRemoteBackend:
    """Mock backend that simulates remote execution."""

    def __init__(
        self, agent_registry: Dict[str, AsyncAgentProtocol[Any, Any]] | None = None
    ) -> None:
        self.agent_registry = agent_registry or {}
        self.call_counter = 0
        self.recorded_requests: List[StepExecutionRequest] = []

        # ✅ Create ExecutorCore and inject into LocalBackend
        from ..application.core.executor_core import ExecutorCore

        executor: ExecutorCore[Any] = ExecutorCore()
        self.local: LocalBackend[Any] = LocalBackend(
            executor=executor, agent_registry=self.agent_registry
        )

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
        serialized = _serialize_for_test(payload)
        data = json.loads(json.dumps(serialized))

        # Don't apply _parse_dict_like_strings to the entire data object
        # as it can cause issues with complex nested structures
        # Instead, handle string parsing in the reconstruct function where needed

        def _ensure_string_fields_are_strings(obj: Any, original: Any = None) -> Any:
            """Ensure that string fields in Pydantic models are actually strings."""
            if isinstance(obj, dict):
                cleaned: JSONObject = {}
                for key, value in obj.items():
                    orig_val = None
                    if original and isinstance(original, dict):
                        orig_val = original.get(key)
                    if isinstance(value, str):
                        cleaned[key] = value
                    elif isinstance(value, dict):
                        cleaned[key] = _ensure_string_fields_are_strings(value, orig_val)
                    elif isinstance(value, list):
                        cleaned[key] = [
                            _ensure_string_fields_are_strings(
                                item,
                                (
                                    orig_val[0]
                                    if orig_val and isinstance(orig_val, list) and orig_val
                                    else None
                                ),
                            )
                            for item in value
                        ]
                    elif hasattr(value, "model_dump"):  # Pydantic v2 models
                        cleaned[key] = _ensure_string_fields_are_strings(
                            value.model_dump(), orig_val
                        )
                    elif hasattr(value, "dict"):  # Pydantic v1 models
                        cleaned[key] = _ensure_string_fields_are_strings(value.dict(), orig_val)
                    elif isinstance(value, (int, float, bool)):
                        if orig_val is not None and isinstance(orig_val, str):
                            cleaned[key] = str(value)
                        else:
                            cleaned[key] = value
                    else:
                        cleaned[key] = (
                            str(value)
                            if (orig_val is not None and isinstance(orig_val, str))
                            else value
                        )
                return cleaned
            elif isinstance(obj, list):
                orig_elem = (
                    original[0] if original and isinstance(original, list) and original else None
                )
                return [_ensure_string_fields_are_strings(item, orig_elem) for item in obj]
            else:
                if original is not None and isinstance(original, str) and not isinstance(obj, str):
                    return str(obj)
                return obj

        def reconstruct(original: Any, value: Any) -> Any:
            """Rebuild a value using the type of ``original``."""
            if original is None or value is None:
                return None
            if isinstance(original, FlujoBaseModel):
                if isinstance(value, dict):
                    fixed_value = {}
                    for k, v in value.items():
                        original_field_value = getattr(original, k, None)
                        reconstructed_value = reconstruct(original_field_value, v)
                        # Skip None values to avoid Pydantic validation errors for non-optional fields
                        if reconstructed_value is not None:
                            fixed_value[k] = reconstructed_value
                    # Apply enhanced cleaning to ensure all string fields are strings
                    fixed_value = _ensure_string_fields_are_strings(fixed_value, original)
                    return type(original).model_validate(fixed_value)
                elif isinstance(value, str):
                    # Handle case where Pydantic model was serialized as string representation
                    # Try to parse it back to a dict
                    try:
                        if (
                            "=" in value
                            and not value.strip().startswith("{")
                            and not value.strip().startswith("[")
                        ):
                            # Convert foo='bar' bar=42 to {'foo': 'bar', 'bar': 42}
                            items = value.split()
                            d = {}
                            for item in items:
                                if "=" in item:
                                    k, v = item.split("=", 1)
                                    v = v.strip()
                                    if v.startswith("'") and v.endswith("'"):
                                        v = v[1:-1]
                                    elif v.startswith('"') and v.endswith('"'):
                                        v = v[1:-1]
                                    else:
                                        try:
                                            v_eval = ast.literal_eval(v)
                                            # If v_eval is not a string, convert to string
                                            v = (
                                                str(v_eval)
                                                if not isinstance(v_eval, str)
                                                else v_eval
                                            )
                                        except Exception:
                                            pass
                                    d[k] = v
                            fixed_value = {
                                k: reconstruct(getattr(original, k, None), v) for k, v in d.items()
                            }
                            fixed_value = _ensure_string_fields_are_strings(fixed_value, original)
                            return type(original).model_validate(fixed_value)
                    except Exception:
                        pass
                    # If parsing fails, return the original
                    return original
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
                    reconstructed_dict = {}
                    for k, v in value.items():
                        # For dict reconstruction, preserve the value even if original doesn't have that key
                        original_value = original.get(k)
                        # Enhanced: Handle None values more gracefully in reconstruction
                        if original_value is not None:
                            reconstructed_dict[k] = reconstruct(original_value, v)
                        else:
                            reconstructed_dict[k] = v
                    return reconstructed_dict
                else:
                    return original
            else:
                return value

        # Reconstruct the payload with proper types
        reconstructed_payload = {}
        for key, original_value in payload.items():
            if key in data:
                if key == "context" and isinstance(original_value, FlujoBaseModel):
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

        # Delegate to local backend which may return typed outcomes
        outcome = await self.local.execute_step(reconstructed_request)
        if isinstance(outcome, StepOutcome):
            if isinstance(outcome, Paused):
                raise PausedException(outcome.message)
            if isinstance(outcome, Success):
                return outcome.step_result
            if isinstance(outcome, Failure):
                return outcome.step_result or StepResult(
                    name=getattr(reconstructed_request.step, "name", "<unnamed>"),
                    success=False,
                    feedback=outcome.feedback,
                )
        # Legacy path already returns StepResult
        if isinstance(outcome, StepResult):
            return outcome
        raise TypeError(f"Unexpected outcome type: {type(outcome)}")


@contextmanager
def override_agent(step: Any, new_agent: Any) -> Iterator[None]:
    """Temporarily override the agent of a Step within a context."""
    original_agent = getattr(step, "agent", None)
    step.agent = new_agent
    try:
        yield
    finally:
        step.agent = original_agent


@contextmanager
def override_agent_direct(original_agent: Any, replacement_agent: Any) -> Iterator[None]:
    """Temporarily delegate all calls from original_agent to replacement_agent, without mutating original_agent's state."""
    if original_agent is None or replacement_agent is None or original_agent is replacement_agent:
        yield
        return

    # Store original methods
    original_run = getattr(original_agent, "run", None)
    original_run_async = getattr(original_agent, "run_async", None)

    # Create wrappers that delegate to the replacement agent
    async def run_wrapper(data: Any = None, **kwargs: Any) -> Any:
        return await replacement_agent.run(data, **kwargs)

    async def run_async_wrapper(data: Any = None, **kwargs: Any) -> Any:
        return await replacement_agent.run_async(data, **kwargs)

    # Replace methods
    original_agent.run = run_wrapper
    original_agent.run_async = run_async_wrapper

    try:
        yield
    finally:
        # Restore original methods
        if original_run is not None:
            original_agent.run = original_run
        if original_run_async is not None:
            original_agent.run_async = original_run_async


# Remove the test_validator_failed function entirely from this file
# It should be moved to a separate utility module or made private


class SimpleDummyRemoteBackend(ExecutionBackend):
    """A simple dummy remote backend for testing purposes."""

    def __init__(self) -> None:
        self.storage: dict[str, Any] = {}
        self.call_count = 0

    async def execute_step(self, request: StepExecutionRequest) -> StepOutcome[StepResult]:
        """Execute a step by storing and retrieving the result."""
        self.call_count += 1

        # Store the input data
        key = f"step_{request.step.name}_{self.call_count}"
        self.store(key, request.input_data)

        # Actually execute the agent to get the result
        agent = request.step.agent
        if hasattr(agent, "run"):
            run_method = getattr(agent, "run")
            if hasattr(run_method, "__call__"):
                # Check if it's an async method
                import inspect

                if inspect.iscoroutinefunction(run_method):
                    # Use async run method
                    result_data = await run_method(request.input_data, context=request.context)
                elif getattr(agent, "run_async", None) is not None:
                    # Use async run method if available
                    run_async_method = getattr(agent, "run_async")
                    if run_async_method is not None:
                        result_data = await run_async_method(
                            request.input_data, context=request.context
                        )
                    else:
                        # Fallback to sync run method
                        result_data = run_method(request.input_data, context=request.context)
            else:
                # Fallback to stored data if no run method
                result_data = self.retrieve(key)
        else:
            # Fallback to stored data if no run method
            result_data = self.retrieve(key)

        return Success(
            step_result=StepResult(
                name=request.step.name,
                output=result_data,
            )
        )

    def store(self, key: str, value: Any) -> None:
        """Store a value with robust serialization."""
        self.call_count += 1
        # Use the Pydantic-native serialization pattern
        serialized = _serialize_for_test(value)
        self.storage[key] = serialized

    def retrieve(self, key: str) -> Any:
        """Retrieve and reconstruct a value."""
        if key not in self.storage:
            raise KeyError(f"Key {key} not found in storage")

        serialized_value = self.storage[key]
        return self._reconstruct(serialized_value)

    def _reconstruct(self, serialized_value: Any) -> Any:
        """
        Reconstruct an object from its serialized form.

        This method handles the reconstruction of:
        - Primitives (str, int, float, bool, None)
        - Lists, tuples, sets
        - Dictionaries
        - Pydantic models (basic reconstruction)
        - Special float values (inf, -inf, nan)

        Args:
            serialized_value: The serialized value to reconstruct

        Returns:
            The reconstructed object
        """
        if serialized_value is None:
            return None

        # Handle primitives
        if isinstance(serialized_value, (str, int, bool)):
            return serialized_value

        # Handle special float values
        if isinstance(serialized_value, str):
            if serialized_value == "nan":
                return float("nan")
            if serialized_value == "inf":
                return float("inf")
            if serialized_value == "-inf":
                return float("-inf")

        # Handle float
        if isinstance(serialized_value, float):
            return serialized_value

        # Handle lists
        if isinstance(serialized_value, list):
            return [self._reconstruct(item) for item in serialized_value]

        # Handle dictionaries
        if isinstance(serialized_value, dict):
            reconstructed = {}
            for k, v in serialized_value.items():
                # Reconstruct key and value
                reconstructed_key = self._reconstruct(k)
                reconstructed_value = self._reconstruct(v)
                reconstructed[reconstructed_key] = reconstructed_value
            return reconstructed

        # For any other type, return as-is
        return serialized_value

    def clear(self) -> None:
        """Clear all stored data."""
        self.storage.clear()
        self.call_count = 0

    def get_call_count(self) -> int:
        """Get the number of store calls made."""
        return self.call_count

    def get_storage_keys(self) -> List[str]:
        """Get all storage keys."""
        return list(self.storage.keys())

    def get_storage_size(self) -> int:
        """Get the number of items in storage."""
        return len(self.storage)
