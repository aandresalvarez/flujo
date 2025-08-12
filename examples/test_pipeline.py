"""Example pipeline for testing the flujo run command."""

from flujo import step
from flujo.domain.models import PipelineContext
from pydantic import Field


class ExamplePipelineContext(PipelineContext):
    """Example context for the test pipeline."""

    __test__ = False  # Tell pytest this is not a test class

    counter: int = Field(default=0, description="Counter for tracking steps")
    messages: list[str] = Field(default_factory=list, description="List of processed messages")


@step
async def echo_step(data: str, *, context: ExamplePipelineContext | None = None) -> str:
    """Simple echo step that adds to context."""
    if context:
        context.counter += 1
        context.messages.append(f"Echoed: {data}")
    return f"Echo: {data}"


@step
async def transform_step(data: str, *, context: ExamplePipelineContext | None = None) -> str:
    """Transform step that modifies the data."""
    if context:
        context.counter += 1
        context.messages.append(f"Transformed: {data}")
    return f"Transformed: {data.upper()}"


@step
async def finalize_step(data: str, *, context: ExamplePipelineContext | None = None) -> str:
    """Final step that adds a summary."""
    if context:
        context.counter += 1
        context.messages.append(f"Finalized: {data}")
    return f"Final result: {data} (processed in {context.counter if context else 0} steps)"


# Create the pipeline
pipeline = echo_step >> transform_step >> finalize_step

TestContext = ExamplePipelineContext
