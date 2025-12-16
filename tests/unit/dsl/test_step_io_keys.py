from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.conditional import ConditionalStep
from flujo.domain.dsl.import_step import ImportStep, OutputMapping
from flujo.domain.dsl.parallel import ParallelStep
from flujo.domain.dsl.step import MergeStrategy, Step


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
    s1 = Step(name="first", agent=_agent, output_keys=["import_artifacts"])
    s1.__step_output_type__ = dict[str, object]
    s2 = Step(name="second", agent=_agent, input_keys=["import_artifacts.summary"])
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
    cond.__step_output_type__ = dict[str, object]
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
    branch_one_step = Step(name="branch-one", agent=_agent, output_keys=["branch_value"])
    branch_one_step.__step_output_type__ = dict[str, object]
    branch_one = Pipeline.model_construct(steps=[branch_one_step], hooks=[], on_finish=[])
    branch_two_step = Step(name="branch-two", agent=_agent, output_keys=["other_value"])
    branch_two_step.__step_output_type__ = dict[str, object]
    branch_two = Pipeline.model_construct(steps=[branch_two_step], hooks=[], on_finish=[])
    parallel = ParallelStep(
        name="parallel",
        agent=_agent,
        branches={"one": branch_one, "two": branch_two},
    )
    parallel.__step_output_type__ = dict[str, object]
    consumer = Step(
        name="after-parallel",
        agent=_agent,
        input_keys=["branch_value", "other_value"],
    )
    consumer.__step_input_type__ = dict[str, object]

    pipeline = Pipeline.model_construct(steps=[parallel, consumer], hooks=[], on_finish=[])
    report = pipeline.validate_graph()

    assert not report.errors


def test_import_step_outputs_mapping_produces_parent_paths() -> None:
    child_step = Step(
        name="child", agent=_agent, updates_context=True, output_keys=["custom.child"]
    )
    child_step.__step_output_type__ = dict[str, object]
    child_pipeline = Pipeline.model_construct(steps=[child_step], hooks=[], on_finish=[])

    imp = ImportStep(
        name="imp",
        pipeline=child_pipeline,
        updates_context=True,
        outputs=[OutputMapping(child="custom.child", parent="custom.parent")],
    )
    imp.__step_output_type__ = dict[str, object]

    consumer = Step(name="after-import", agent=_agent, input_keys=["custom.parent"])
    consumer.__step_input_type__ = dict[str, object]

    pipeline = Pipeline.model_construct(steps=[imp, consumer], hooks=[], on_finish=[])
    report = pipeline.validate_graph()

    assert not any(f.rule_id == "CTX-OUTPUT-KEYS" for f in report.errors)
    assert not any(f.rule_id == "V-CTX1" for f in report.errors)


def test_import_step_outputs_empty_does_not_propagate_child_paths() -> None:
    child_step = Step(
        name="child", agent=_agent, updates_context=True, output_keys=["custom.child"]
    )
    child_step.__step_output_type__ = dict[str, object]
    child_pipeline = Pipeline.model_construct(steps=[child_step], hooks=[], on_finish=[])

    imp = ImportStep(name="imp", pipeline=child_pipeline, updates_context=True, outputs=[])
    imp.__step_output_type__ = dict[str, object]

    consumer = Step(name="after-import", agent=_agent, input_keys=["custom.child"])
    consumer.__step_input_type__ = dict[str, object]

    pipeline = Pipeline.model_construct(steps=[imp, consumer], hooks=[], on_finish=[])
    report = pipeline.validate_graph()

    assert any(f.rule_id == "V-CTX1" for f in report.errors)


def test_import_step_inherit_context_false_validates_child_without_parent_keys() -> None:
    parent_producer = Step(
        name="produce-parent",
        agent=_agent,
        updates_context=True,
        output_keys=["custom.parent"],
    )
    parent_producer.__step_output_type__ = dict[str, object]

    child_consumer = Step(name="child-consumer", agent=_agent, input_keys=["custom.parent"])
    child_consumer.__step_input_type__ = dict[str, object]
    child_pipeline = Pipeline.model_construct(steps=[child_consumer], hooks=[], on_finish=[])

    imp = ImportStep(
        name="imp", pipeline=child_pipeline, updates_context=False, inherit_context=False
    )
    imp.__step_output_type__ = dict[str, object]

    pipeline = Pipeline.model_construct(steps=[parent_producer, imp], hooks=[], on_finish=[])
    report = pipeline.validate_graph()

    assert any(f.rule_id == "V-CTX1" for f in report.errors)


def test_parallel_no_merge_does_not_propagate_branch_context_keys() -> None:
    branch_one_step = Step(
        name="branch-one",
        agent=_agent,
        updates_context=True,
        output_keys=["custom.branch_value"],
    )
    branch_one_step.__step_output_type__ = dict[str, object]
    branch_one = Pipeline.model_construct(steps=[branch_one_step], hooks=[], on_finish=[])

    parallel = ParallelStep(
        name="parallel",
        agent=_agent,
        branches={"one": branch_one},
        merge_strategy=MergeStrategy.NO_MERGE,
    )
    parallel.__step_output_type__ = dict[str, object]

    consumer = Step(name="after-parallel", agent=_agent, input_keys=["custom.branch_value"])
    consumer.__step_input_type__ = dict[str, object]

    pipeline = Pipeline.model_construct(steps=[parallel, consumer], hooks=[], on_finish=[])
    report = pipeline.validate_graph()

    assert any(f.rule_id == "V-CTX1" for f in report.errors)


def test_parallel_field_mapping_empty_list_merges_nothing_for_branch() -> None:
    branch_one_step = Step(
        name="branch-one",
        agent=_agent,
        updates_context=True,
        output_keys=["a_root.a"],
    )
    branch_one_step.__step_output_type__ = dict[str, object]
    branch_one = Pipeline.model_construct(steps=[branch_one_step], hooks=[], on_finish=[])

    branch_two_step = Step(
        name="branch-two",
        agent=_agent,
        updates_context=True,
        output_keys=["b_root.b"],
    )
    branch_two_step.__step_output_type__ = dict[str, object]
    branch_two = Pipeline.model_construct(steps=[branch_two_step], hooks=[], on_finish=[])

    parallel = ParallelStep(
        name="parallel",
        agent=_agent,
        branches={"a": branch_one, "b": branch_two},
        merge_strategy=MergeStrategy.CONTEXT_UPDATE,
        field_mapping={"a": ["a_root.a"], "b": []},
    )
    parallel.__step_output_type__ = dict[str, object]

    consumer = Step(name="after-parallel", agent=_agent, input_keys=["b_root.b"])
    consumer.__step_input_type__ = dict[str, object]

    pipeline = Pipeline.model_construct(steps=[parallel, consumer], hooks=[], on_finish=[])
    report = pipeline.validate_graph()

    assert any(f.rule_id == "V-CTX1" for f in report.errors)


def test_conditional_only_guarantees_common_branch_keys() -> None:
    branch_a_step = Step(
        name="branch-a-step",
        agent=_agent,
        updates_context=True,
        output_keys=["only_a_root.only_a", "common_root.common"],
    )
    branch_a_step.__step_output_type__ = dict[str, object]
    branch_a = Pipeline.model_construct(steps=[branch_a_step], hooks=[], on_finish=[])

    branch_b_step = Step(
        name="branch-b-step",
        agent=_agent,
        updates_context=True,
        output_keys=["only_b_root.only_b", "common_root.common"],
    )
    branch_b_step.__step_output_type__ = dict[str, object]
    branch_b = Pipeline.model_construct(steps=[branch_b_step], hooks=[], on_finish=[])

    cond = ConditionalStep(
        name="choose",
        agent=_agent,
        branches={"a": branch_a, "b": branch_b},
        condition_callable=lambda *_: "a",
    )
    cond.__step_output_type__ = dict[str, object]

    consumer = Step(name="after-cond", agent=_agent, input_keys=["only_a_root.only_a"])
    consumer.__step_input_type__ = dict[str, object]

    pipeline = Pipeline.model_construct(steps=[cond, consumer], hooks=[], on_finish=[])
    report = pipeline.validate_graph()

    assert any(f.rule_id == "V-CTX1" for f in report.errors)
