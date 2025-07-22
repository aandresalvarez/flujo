from flujo.domain import Step
from flujo.testing.utils import StubAgent, DummyRemoteBackend
from tests.conftest import create_test_flujo


def test_pipeline_runs_correctly_with_custom_backend() -> None:
    backend = DummyRemoteBackend()
    pipeline = Step.model_validate({"name": "a", "agent": StubAgent(["x"])}) >> Step.model_validate(
        {"name": "b", "agent": StubAgent(["y"])}
    )
    runner = create_test_flujo(pipeline, backend=backend)

    result = runner.run("start")

    assert backend.call_counter == 2
    assert len(result.step_history) == 2
    assert all(sr.success for sr in result.step_history)
    assert result.step_history[-1].output == "y"

    Step.model_validate({"name": "s", "agent": StubAgent(["ok"])})
