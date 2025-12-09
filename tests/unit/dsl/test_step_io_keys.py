from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.conditional import ConditionalStep
from flujo.domain.dsl.parallel import ParallelStep
from flujo.domain.dsl.step import Step


def _agent(x: dict[str, object]) -> dict[str, object]:
    return x


def test_step_io_keys_validation_passes_when_produced() -> None:
    s1 = Step(name="first", agent=_agent, output_keys=["summary"])
    s1.__step_output_type__ = dict[str, object]
    s2 = Step(name="second", agent=_agent, input_keys=["summary"])
    s2.__step_input_type__ = dict[str, object]

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
    s1.__step_output_type__ = dict[str, object]
    s2 = Step(name="second", agent=_agent, input_keys=["scratchpad.summary"])
    s2.__step_input_type__ = dict[str, object]

    pipeline = Pipeline.model_construct(steps=[s1, s2], hooks=[], on_finish=[])
    report = pipeline.validate_graph()

    assert not report.errors
    assert any(f.rule_id == "V-CTX2" for f in report.warnings)


def test_step_io_keys_follow_branch_union_for_conditional() -> None:
    branch_step = Step(name="branch-a-step", agent=_agent, output_keys=["branch_value"])
    branch_step.__step_output_type__ = dict[str, object]
    branch_a = Pipeline.model_construct(steps=[branch_step], hooks=[], on_finish=[])
    cond = ConditionalStep(
        name="choose",
        agent=_agent,
        branches={"a": branch_a},
        condition_callable=lambda *_: "a",
    )
    consumer = Step(name="after-cond", agent=_agent, input_keys=["branch_value"])
    consumer.__step_input_type__ = dict[str, object]

    pipeline = Pipeline.model_construct(steps=[cond, consumer], hooks=[], on_finish=[])
    report = pipeline.validate_graph()

    assert not report.errors


def test_step_io_keys_require_branch_outputs_when_missing() -> None:
    empty_branch = Pipeline.model_construct(
        steps=[Step(name="branch-a-step", agent=_agent)],
        hooks=[],
        on_finish=[],
    )
    cond = ConditionalStep(
        name="choose",
        agent=_agent,
        branches={"a": empty_branch},
        condition_callable=lambda *_: "a",
    )
    consumer = Step(name="after-cond", agent=_agent, input_keys=["branch_value"])

    pipeline = Pipeline.model_construct(steps=[cond, consumer], hooks=[], on_finish=[])
    report = pipeline.validate_graph()

    assert any(f.rule_id == "V-CTX1" for f in report.errors)


def test_step_io_keys_union_parallel_branch_outputs() -> None:
    branch_one = Pipeline.model_construct(
        steps=[Step(name="branch-one", agent=_agent, output_keys=["branch_value"])],
        hooks=[],
        on_finish=[],
    )
    branch_two = Pipeline.model_construct(
        steps=[Step(name="branch-two", agent=_agent, output_keys=["other_value"])],
        hooks=[],
        on_finish=[],
    )
    parallel = ParallelStep(
        name="parallel",
        agent=_agent,
        branches={"one": branch_one, "two": branch_two},
    )
    consumer = Step(
        name="after-parallel",
        agent=_agent,
        input_keys=["branch_value", "other_value"],
    )

    pipeline = Pipeline.model_construct(steps=[parallel, consumer], hooks=[], on_finish=[])
    report = pipeline.validate_graph()

    assert not report.errors
