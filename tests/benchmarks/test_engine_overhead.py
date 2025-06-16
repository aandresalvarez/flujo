import pytest
from flujo.domain import Step
from flujo.application.flujo_engine import Flujo
from flujo.testing.utils import StubAgent

pytest.importorskip("pytest_benchmark")


@pytest.mark.benchmark(group="engine-overhead")
def test_pipeline_runner_overhead(benchmark):
    """Measures the execution time of the Flujo engine's orchestration logic,
    minimizing agent execution time by using a fast stub."""
    agent = StubAgent(["output"])
    pipeline = Step("s1", agent) >> Step("s2", agent) >> Step("s3", agent) >> Step("s4", agent)
    runner = Flujo(pipeline)

    @benchmark
    def run_pipeline():
        runner.run("initial input")
