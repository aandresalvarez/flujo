import os
from typing import Any, Optional, Dict
from flujo import Flujo
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl import Step
from flujo.state.backends.base import StateBackend

# Set test mode environment variable
os.environ["FLUJO_TEST_MODE"] = "1"


def create_test_flujo(
    pipeline: Pipeline[Any, Any] | Step[Any, Any],
    *,
    pipeline_name: Optional[str] = None,
    pipeline_id: Optional[str] = None,
    **kwargs: Any,
) -> Flujo[Any, Any, Any]:
    """Create a Flujo instance with proper test names and IDs.

    This utility function provides meaningful pipeline names and IDs for tests
    while ensuring the warnings are suppressed in test environments.

    Parameters
    ----------
    pipeline : Pipeline | Step
        The pipeline or step to run
    pipeline_name : str, optional
        Custom pipeline name. If not provided, generates one based on test function name.
    pipeline_id : str, optional
        Custom pipeline ID. If not provided, generates a unique test ID.
    **kwargs : Any
        Additional arguments to pass to Flujo constructor

    Returns
    -------
    Flujo
        Configured Flujo instance with proper test identifiers
    """
    if pipeline_name is None:
        # Generate a descriptive name based on the test function
        import inspect

        frame = inspect.currentframe()
        while frame and not frame.f_code.co_name.startswith("test_"):
            frame = frame.f_back
        if frame:
            pipeline_name = f"test_{frame.f_code.co_name}"
        else:
            pipeline_name = "test_pipeline"

    if pipeline_id is None:
        import uuid

        pipeline_id = f"test_{uuid.uuid4().hex[:8]}"

    return Flujo(pipeline, pipeline_name=pipeline_name, pipeline_id=pipeline_id, **kwargs)


class NoOpStateBackend(StateBackend):
    """A state backend that does nothing - used to disable state persistence in tests."""

    async def save_state(self, run_id: str, state: Dict[str, Any]) -> None:
        # Do nothing - no state persistence
        pass

    async def load_state(self, run_id: str) -> Optional[Dict[str, Any]]:
        # Return None - no state to load
        return None

    async def delete_state(self, run_id: str) -> None:
        # Do nothing
        pass

    async def get_trace(self, run_id: str) -> Any:
        # Return None - no trace data
        return None

    async def save_trace(self, run_id: str, trace: Any) -> None:
        # Do nothing - no trace persistence
        pass
