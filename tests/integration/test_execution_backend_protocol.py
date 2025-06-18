from flujo.application.flujo_engine import Flujo
from flujo.domain import Step
from flujo.testing.utils import StubAgent, DummyRemoteBackend


def test_pipeline_runs_correctly_with_custom_backend() -> None:
    backend = DummyRemoteBackend()
    pipeline = Step("a", StubAgent(["x"])) >> Step("b", StubAgent(["y"]))
    runner = Flujo(pipeline, backend=backend)

    result = runner.run("start")

    assert backend.call_counter == 2
    assert len(result.step_history) == 2
    assert all(sr.success for sr in result.step_history)
    assert result.step_history[-1].output == "y"
