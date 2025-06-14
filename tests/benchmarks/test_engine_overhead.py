import pytest
from pydantic_ai_orchestrator.domain import Step
from pydantic_ai_orchestrator.application.pipeline_runner import PipelineRunner
from pydantic_ai_orchestrator.testing.utils import StubAgent

pytest.importorskip("pytest_benchmark")


@pytest.mark.benchmark(group="engine-overhead")
def test_pipeline_runner_overhead(benchmark):
    """Measures the execution time of the runner minus agent time."""
    agent = StubAgent(["output"])
    pipeline = Step("s1", agent) >> Step("s2", agent) >> Step("s3", agent) >> Step("s4", agent)
    runner = PipelineRunner(pipeline)

    @benchmark
    def run_pipeline():
        runner.run("initial input")
