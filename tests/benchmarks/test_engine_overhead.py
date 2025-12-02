import pytest
from flujo.domain import Step
from flujo.testing.utils import StubAgent
from tests.conftest import create_test_flujo

pytest.importorskip("pytest_benchmark")


@pytest.mark.benchmark(group="engine-overhead")
def test_pipeline_runner_overhead(benchmark):
    """Measures the execution time of the Flujo engine's orchestration logic,
    minimizing agent execution time by using a fast stub."""
    agent = StubAgent(["output"] * 10000)
    pipeline = (
        Step.model_validate({"name": "s1", "agent": agent})
        >> Step.model_validate({"name": "s2", "agent": agent})
        >> Step.model_validate({"name": "s3", "agent": agent})
        >> Step.model_validate({"name": "s4", "agent": agent})
    )
    runner = create_test_flujo(pipeline, persist_state=False)
    runner.disable_tracing()

    @benchmark
    def run_pipeline():
        runner.run("initial input")
