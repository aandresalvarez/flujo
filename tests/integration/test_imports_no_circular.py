def test_import_core_and_dsl_without_cycles():
    # Smoke import test to catch circular-import regressions in core/dsl wiring.
    from flujo.application.core.executor_core import ExecutorCore  # noqa: F401
    from flujo.domain.dsl.step import Step  # noqa: F401
    from flujo.domain.dsl.pipeline import Pipeline  # noqa: F401


def test_construct_basic_pipeline():
    from flujo.domain.dsl.step import Step
    from flujo.domain.dsl.pipeline import Pipeline
    from typing import Any

    class DummyAgent:
        async def run(self, data, **kwargs):
            return data

    s1 = Step[Any, Any](name="s1", agent=DummyAgent())
    s2 = Step[Any, Any](name="s2", agent=DummyAgent())
    pipe = Pipeline.from_step(s1) >> s2
    assert len(pipe.steps) == 2
