from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.step import Step


def _agent(x: object) -> object:
    return x


def test_step_io_keys_validation_passes_when_produced() -> None:
    s1 = Step(name="first", agent=_agent, output_keys=["summary"])
    s2 = Step(name="second", agent=_agent, input_keys=["summary"])

    pipeline = Pipeline.model_construct(steps=[s1, s2], hooks=[], on_finish=[])
    report = pipeline.validate_graph()

    assert not report.errors


def test_step_io_keys_validation_errors_when_missing() -> None:
    s1 = Step(name="first", agent=_agent, output_keys=["unrelated"])
    s2 = Step(name="second", agent=_agent, input_keys=["summary"])

    pipeline = Pipeline.model_construct(steps=[s1, s2], hooks=[], on_finish=[])
    report = pipeline.validate_graph()

    assert any(f.rule_id == "V-CTX1" for f in report.errors)


def test_step_io_keys_warns_when_only_root_available() -> None:
    s1 = Step(name="first", agent=_agent, output_keys=["scratchpad"])
    s2 = Step(name="second", agent=_agent, input_keys=["scratchpad.summary"])

    pipeline = Pipeline.model_construct(steps=[s1, s2], hooks=[], on_finish=[])
    report = pipeline.validate_graph()

    assert not report.errors
    assert any(f.rule_id == "V-CTX2" for f in report.warnings)
