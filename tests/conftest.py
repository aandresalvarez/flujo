import os
from typing import Any, Optional
from flujo import Flujo
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl import Step

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
