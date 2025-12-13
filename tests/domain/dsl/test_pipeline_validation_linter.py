from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel

from flujo.domain.dsl import Step, Pipeline
from flujo.domain.dsl.loop import LoopStep
from flujo.domain.dsl.step import MergeStrategy
from flujo.domain.dsl.parallel import ParallelStep


async def _id(x: Any) -> Any:
    return x


def _lenient_pipeline(steps: list[Step[Any, Any]]) -> Pipeline[Any, Any]:
    prev_strict = os.environ.get("FLUJO_STRICT_DSL")
    os.environ["FLUJO_STRICT_DSL"] = "0"
    try:
        return Pipeline.model_construct(steps=steps)
    finally:
        if prev_strict is None:
            os.environ.pop("FLUJO_STRICT_DSL", None)
        else:
            os.environ["FLUJO_STRICT_DSL"] = prev_strict


def test_validate_graph_detects_merge_conflict_with_field_mapping() -> None:
    p = ParallelStep(
        name="P",
        branches={
            "a": Pipeline.from_step(Step.from_callable(_id, name="a")),
            "b": Pipeline.from_step(Step.from_callable(_id, name="b")),
        },
        merge_strategy=MergeStrategy.CONTEXT_UPDATE,
        field_mapping={"a": ["value"], "b": ["value"]},
    )
    report = Pipeline.model_validate({"steps": [p]}).validate_graph()
    assert any(f.rule_id == "V-P1" for f in report.errors)


def test_validate_graph_warns_unbound_output_without_updates_context() -> None:
    s1 = Step.from_callable(_id, name="producer")

    async def _noop(_: Any) -> Any:
        return None

    s2 = Step.from_callable(_noop, name="consumer")
    # simulate a consumer that does not take prior output (object input)
    s2.__step_input_type__ = object
    report = (Pipeline.from_step(s1) >> s2).validate_graph()
    assert any(f.rule_id == "V-A5" for f in report.warnings)


def test_validate_graph_incompatible_fallback_signature() -> None:
    async def takes_str(x: str) -> str:
        return x

    async def takes_int(x: int) -> int:
        return x

    primary = Step.from_callable(takes_str, name="primary")
    fb = Step.from_callable(takes_int, name="fallback")
    primary.fallback_step = fb

    report = Pipeline.from_step(primary).validate_graph()
    assert any(f.rule_id == "V-F1" for f in report.errors)


def test_parallel_conflict_ignored_when_ignore_branch_names_true() -> None:
    p = ParallelStep(
        name="P",
        branches={
            "a": Pipeline.from_step(Step.from_callable(_id, name="a")),
            "b": Pipeline.from_step(Step.from_callable(_id, name="b")),
        },
        merge_strategy=MergeStrategy.CONTEXT_UPDATE,
        field_mapping={"a": ["value"], "b": ["value"]},
        ignore_branch_names=True,
    )
    report = Pipeline.model_validate({"steps": [p]}).validate_graph()
    assert not any(f.rule_id == "V-P1" for f in report.errors)


def test_parallel_no_conflict_check_when_not_context_update() -> None:
    p = ParallelStep(
        name="P",
        branches={"a": Pipeline.from_step(Step.from_callable(_id, name="a"))},
        merge_strategy=MergeStrategy.OVERWRITE,
        field_mapping={"a": ["value"]},
    )
    report = Pipeline.model_validate({"steps": [p]}).validate_graph()
    assert not any(f.rule_id.startswith("V-P1") for f in (report.errors + report.warnings))


def test_parallel_conflict_with_include_keys_no_field_mapping() -> None:
    p = ParallelStep(
        name="P",
        branches={
            "a": Pipeline.from_step(Step.from_callable(_id, name="a")),
            "b": Pipeline.from_step(Step.from_callable(_id, name="b")),
        },
        merge_strategy=MergeStrategy.CONTEXT_UPDATE,
        context_include_keys=["value"],
        field_mapping=None,
    )
    report = Pipeline.model_validate({"steps": [p]}).validate_graph()
    assert any(f.rule_id == "V-P1" for f in report.errors)


def test_unbound_output_not_warned_when_updates_context_true() -> None:
    async def a(x: int) -> int:  # type: ignore[override]
        return x

    async def b(_: object) -> None:  # type: ignore[override]
        return None

    s1 = Step.from_callable(a, name="a", updates_context=True)
    s2 = Step.from_callable(b, name="b")
    s2.__step_input_type__ = object
    p = Pipeline.model_construct(steps=[s1, s2])
    report = p.validate_graph()
    assert not any(f.rule_id == "V-A5" for f in report.warnings)


def test_fallback_signature_compatible_ok() -> None:
    async def f(x: int) -> int:  # type: ignore[override]
        return x

    primary = Step.from_callable(f, name="p")
    fb = Step.from_callable(f, name="fb")
    primary.fallback_step = fb
    report = Pipeline.from_step(primary).validate_graph()
    assert not any(f.rule_id == "V-F1" for f in report.errors)


def test_type_mismatch_between_steps_reports_V_A2() -> None:
    async def a(x: int) -> int:  # type: ignore[override]
        return x

    async def b(x: str) -> str:  # type: ignore[override]
        return x

    s1 = Step.from_callable(a, name="a")
    s2 = Step.from_callable(b, name="b")
    p = _lenient_pipeline([s1, s2])
    report = p.validate_graph()
    assert any(f.rule_id == "V-A2" for f in report.errors)


def test_type_mismatch_reports_strict_rule_V_A2_TYPE() -> None:
    async def a(x: int) -> int:  # type: ignore[override]
        return x

    async def b(x: str) -> str:  # type: ignore[override]
        return x

    s1 = Step.from_callable(a, name="a")
    s2 = Step.from_callable(b, name="b")
    p = _lenient_pipeline([s1, s2])
    report = p.validate_graph()
    assert any(f.rule_id == "V-A2-TYPE" for f in report.errors)


def test_adapter_step_skips_type_errors_when_allowlisted() -> None:
    async def a(x: int) -> int:  # type: ignore[override]
        return x

    async def b(x: object) -> object:  # type: ignore[override]
        return x

    producer = Step.from_callable(a, name="a")
    adapter = Step.from_callable(
        b,
        name="adapt",
        is_adapter=True,
        adapter_id="generic-adapter",
        adapter_allow="generic",
    )
    report = (Pipeline.from_step(producer) >> adapter).validate_graph()
    assert not any(f.rule_id in {"V-A2", "V-A2-STRICT", "V-A2-TYPE"} for f in report.errors)


def test_missing_agent_simple_step_reports_V_A1() -> None:
    naked = Step(name="naked")  # agent=None by default
    report = Pipeline.from_step(naked).validate_graph()
    assert any(f.rule_id == "V-A1" for f in report.errors)


def test_no_missing_agent_error_for_hitl_step() -> None:
    from flujo.domain.dsl.step import HumanInTheLoopStep

    hitl = HumanInTheLoopStep(name="hitl")
    report = Pipeline.from_step(hitl).validate_graph()
    assert not any(f.rule_id == "V-A1" for f in report.errors)


def test_removed_root_sink_to_errors() -> None:
    async def a(x: int) -> int:  # type: ignore[override]
        return x

    removed_root = "scrat" + "chpad"
    s1 = Step.from_callable(a, name="a", updates_context=True, sink_to=f"{removed_root}.value")
    report = Pipeline.from_step(s1).validate_graph()
    expected = "CTX-" + removed_root.upper()
    assert any(f.rule_id == expected for f in report.errors)


def test_updates_context_without_outputs_errors() -> None:
    async def a(x: int) -> int:  # type: ignore[override]
        return x

    s1 = Step.from_callable(a, name="a", updates_context=True)
    report = Pipeline.from_step(s1).validate_graph()
    assert any(f.rule_id == "CTX-OUTPUT-KEYS" for f in report.errors)


def test_generic_input_without_adapter_errors() -> None:
    async def a(x: int) -> int:  # type: ignore[override]
        return x

    async def b(x: object) -> object:  # type: ignore[override]
        return x

    s1 = Step.from_callable(a, name="a")
    s2 = Step.from_callable(b, name="b")
    p = Pipeline.model_construct(steps=[s1, s2])
    report = p.validate_graph()
    assert any(f.rule_id == "V-A2-STRICT" for f in report.errors)


def test_adapter_requires_allowlist_token() -> None:
    async def a(x: int) -> int:  # type: ignore[override]
        return x

    s1 = Step.from_callable(a, name="a")
    s2 = Step.from_callable(
        a,
        name="adapt",
        is_adapter=True,
        adapter_id="generic-adapter",
        adapter_allow="generic",
    )
    # Remove token to trigger failure
    s2.meta["adapter_allow"] = "wrong"
    report = (Pipeline.from_step(s1) >> s2).validate_graph()
    assert any(f.rule_id == "V-ADAPT-ALLOW" for f in report.errors)


def test_concrete_type_mismatch_reports_v_a2_type() -> None:
    async def a(x: int) -> int:  # type: ignore[override]
        return x

    async def b(x: float) -> float:  # type: ignore[override]
        return x

    s1 = Step.from_callable(a, name="a")
    s2 = Step.from_callable(b, name="b")
    p = _lenient_pipeline([s1, s2])
    report = p.validate_graph()

    assert any(f.rule_id == "V-A2-TYPE" for f in report.errors)


def test_parallel_branch_requires_adapter_for_generic_consumer() -> None:
    async def a(x: int) -> int:  # type: ignore[override]
        return x

    async def b(x: object) -> object:  # type: ignore[override]
        return x

    branch_a = Pipeline.from_step(Step.from_callable(a, name="a"))
    branch_b = Pipeline.from_step(
        Step.from_callable(
            b,
            name="b",
            is_adapter=True,
            adapter_id="generic-adapter",
            adapter_allow="generic",
        ).model_copy(
            update={
                "meta": {
                    "is_adapter": True,
                    "adapter_id": "generic-adapter",
                    "adapter_allow": "generic",
                }
            }
        )
    )
    p = ParallelStep(
        name="P",
        branches={"a": branch_a, "b": branch_b},
        merge_strategy=MergeStrategy.CONTEXT_UPDATE,
    )
    report = Pipeline.from_step(p).validate_graph()
    assert not any(f.rule_id == "V-A2-STRICT" for f in report.errors)


def test_parallel_merge_removed_root_rejected() -> None:
    branches = {
        "a": Pipeline.from_step(Step.from_callable(_id, name="a")),
        "b": Pipeline.from_step(Step.from_callable(_id, name="b")),
    }
    removed_root = "scrat" + "chpad"
    p = ParallelStep.model_construct(
        name="P",
        branches=branches,
        merge_strategy="merge_" + removed_root,
    )
    report = Pipeline.from_step(p).validate_graph()
    expected = "V-P-" + removed_root.upper()
    assert any(f.rule_id == expected for f in report.errors)


def test_pydantic_to_dict_requires_adapter() -> None:
    class Model(BaseModel):
        x: int

    async def a(x: int) -> Model:  # type: ignore[override]
        return Model(x=x)

    async def b(x: dict[str, str]) -> dict[str, str]:  # type: ignore[override]
        return x

    s1 = Step.from_callable(a, name="a")
    s2 = Step.from_callable(b, name="b")

    p = _lenient_pipeline([s1, s2])
    report = p.validate_graph()
    assert any(f.rule_id == "V-A2-TYPE" for f in report.errors)


def test_pydantic_to_dict_allowed_via_adapter() -> None:
    class Model(BaseModel):
        x: int

    async def a(x: int) -> Model:  # type: ignore[override]
        return Model(x=x)

    async def b(x: dict[str, str]) -> dict[str, str]:  # type: ignore[override]
        return x

    s1 = Step.from_callable(a, name="a")
    s2 = Step.from_callable(
        b,
        name="adapt",
        is_adapter=True,
        adapter_id="generic-adapter",
        adapter_allow="generic",
    )

    report = (Pipeline.from_step(s1) >> s2).validate_graph()
    assert not any(f.rule_id == "V-A2-TYPE" for f in report.errors)


def test_pipeline_records_head_and_tail_types() -> None:
    async def a(x: int) -> str:  # type: ignore[override]
        return str(x)

    async def b(x: str) -> float:  # type: ignore[override]
        return float(len(x))

    s1 = Step.from_callable(a, name="a")
    s2 = Step.from_callable(b, name="b")

    p1 = Pipeline.from_step(s1)
    assert p1.input_type is int
    assert p1.output_type is str

    p2 = p1 >> s2
    assert p2.input_type is int
    assert p2.output_type is float


def test_pipeline_rshift_with_pipeline_preserves_head_tail() -> None:
    async def a(x: int) -> str:  # type: ignore[override]
        return str(x)

    async def b(x: str) -> bytes:  # type: ignore[override]
        return x.encode()

    async def c(x: bytes) -> bool:  # type: ignore[override]
        return bool(x)

    p_left = Pipeline.from_step(Step.from_callable(a, name="a")) >> Step.from_callable(b, name="b")
    p_right = Pipeline.from_step(Step.from_callable(c, name="c"))

    chained = p_left >> p_right
    assert chained.input_type is int
    assert chained.output_type is bool


def test_reused_step_instance_warns_V_A3() -> None:
    s = Step.from_callable(_id, name="dup")
    p = Pipeline.model_construct(steps=[s, s])
    report = p.validate_graph()
    assert any(f.rule_id == "V-A3" for f in report.warnings)


def test_signature_analysis_failure_warns(monkeypatch) -> None:  # type: ignore[no-redef]
    # Create step before monkeypatch so type inference succeeds
    s = Step.from_callable(_id, name="s")

    import flujo.signature_tools as st

    def _boom(*_args: Any, **_kw: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(st, "analyze_signature", _boom)
    report = Pipeline.from_step(s).validate_graph()
    assert any(f.rule_id == "V-A4-ERR" for f in report.warnings)


def test_suggestions_present_for_rules() -> None:
    async def a(x: int) -> int:  # type: ignore[override]
        return x

    async def b(_: object) -> None:  # type: ignore[override]
        return None

    s1 = Step.from_callable(a, name="a")
    s2 = Step.from_callable(b, name="b")
    s2.__step_input_type__ = object
    p = Pipeline.model_construct(steps=[s1, s2])
    rep = p.validate_graph()
    sug = next(f.suggestion for f in rep.warnings if f.rule_id == "V-A5")
    assert sug is not None and "updates_context" in sug


def test_hitl_directly_in_loop_body_allowed() -> None:
    hitl = Step.human_in_the_loop(name="collect_feedback", message_for_user="Continue?")
    loop_body = Pipeline.from_step(hitl)

    loop = LoopStep(
        name="interactive_loop",
        loop_body_pipeline=loop_body,
        exit_condition_callable=lambda _output, _ctx: True,
        max_retries=1,
    )

    report = Pipeline.from_step(loop).validate_graph()
    assert not any(f.rule_id == "HITL-NESTED-001" for f in report.errors)


def test_hitl_in_conditional_inside_loop_triggers_nested_error() -> None:
    def _needs_more_info(previous: str, _ctx: Any) -> str:
        return "true" if str(previous) else "false"

    async def _finalize_step(x: Any) -> Any:
        return x

    ask_human = Step.human_in_the_loop(name="ask_user", message_for_user="Need details?")
    finalize = Step.from_callable(_finalize_step, name="finalize")

    conditional = Step.branch_on(
        name="maybe_request_info",
        condition_callable=_needs_more_info,
        branches={
            "true": Pipeline.from_step(ask_human),
            "false": Pipeline.from_step(finalize),
        },
    )

    loop = LoopStep(
        name="conditional_loop",
        loop_body_pipeline=Pipeline.from_step(conditional),
        exit_condition_callable=lambda _output, _ctx: True,
        max_retries=1,
    )

    report = Pipeline.from_step(loop).validate_graph()
    assert any(f.rule_id == "HITL-NESTED-001" for f in report.errors)
