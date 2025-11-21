import pytest

from flujo import Flujo, Pipeline, Step
from flujo.domain.models import PipelineResult
from flujo.infra.registry import PipelineRegistry
from flujo.testing.utils import StubAgent


@pytest.mark.asyncio
async def test_run_session_resolves_pipeline_from_registry() -> None:
    """Smoke test: Flujo runs with registry-backed pipeline outside the CLI."""
    registry = PipelineRegistry()
    pipeline = Pipeline.from_step(Step.model_validate({"name": "s1", "agent": StubAgent(["ok"])}))
    registry.register(pipeline, name="smoke", version="1.0.0")

    runner = Flujo(
        pipeline=None,
        registry=registry,
        pipeline_name="smoke",
        pipeline_version="latest",
    )

    final_result: PipelineResult | None = None
    async for item in runner.run_async("ping"):
        final_result = item

    assert isinstance(final_result, PipelineResult)
    assert final_result.step_history[-1].output == "ok"
    # Ensure resolver updated to the latest registered version
    assert runner.pipeline_version == "1.0.0"
