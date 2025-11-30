import pytest

from flujo import Flujo, Pipeline, Step
from flujo.application.run_plan_resolver import RunPlanResolver
from flujo.domain.dsl.step import HumanInTheLoopStep
from flujo.domain.models import PipelineResult
from flujo.exceptions import HitlPolicyError
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


def test_run_plan_resolver_enforces_hitl_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Policy should prevent HITL pipelines when disallowed by env."""
    monkeypatch.setenv("FLUJO_ALLOW_HITL", "0")
    pipeline = Pipeline.from_step(HumanInTheLoopStep(name="hitl", message_for_user="Need approval"))
    resolver = RunPlanResolver(
        pipeline=pipeline,
        registry=None,
        pipeline_name="hitl_test",
        pipeline_version="1.0.0",
    )

    with pytest.raises(HitlPolicyError):
        resolver.ensure_pipeline()
