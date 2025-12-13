from __future__ import annotations

from flujo.domain.dsl import Pipeline, Step


def test_v_c1_updates_context_nonmergeable_warns_and_escalates() -> None:
    """V-C1: updates_context with non-mergeable output should warn/escalate.

    We construct a typed pipeline so __step_output_type__/__step_input_type__ are known.
    prev step returns int (non-mergeable) and sets updates_context=True.
    next step accepts object (default) and thus does not consume previous output.
    Expect: V-C1 as error (escalated) on the previous step.
    """

    async def produce(_: str) -> int:  # type: ignore[override]
        return 1

    async def consume(_: object) -> None:  # type: ignore[override]
        return None

    s1 = Step.from_callable(produce, name="p1")
    s1.updates_context = True
    s2 = Step.from_callable(consume, name="p2")
    report = (Pipeline.from_step(s1) >> s2).validate_graph()
    # Find V-C1 on p1
    vc1 = [
        f for f in (report.errors + report.warnings) if f.rule_id == "V-C1" and f.step_name == "p1"
    ]
    assert vc1, f"Expected V-C1 finding on p1, got: {report.model_dump()}"
    # Because consume does not use the int, escalation -> error
    assert any(f.severity == "error" for f in vc1)


def test_v_c2_removed_root_shape_conflicts_warns() -> None:
    """V-C2: Mapping to a removed root is now an error."""
    from flujo.domain.dsl.import_step import ImportStep, OutputMapping

    # Child pipeline can be anything; we only check the mapping target
    removed_root = "scrat" + "chpad"
    child = Pipeline.model_validate(
        {
            "steps": [
                Step.model_validate(
                    {
                        "name": "c1",
                        "agent": "flujo.builtins.stringify",
                        "meta": {"templated_input": "x"},
                    }
                )
            ],
        }
    )
    imp = ImportStep.model_validate(
        {
            "name": "RunChild",
            "pipeline": child,
            "updates_context": True,
            "outputs": [OutputMapping(child=f"{removed_root}.value", parent=removed_root)],
        }
    )
    report = Pipeline.model_validate({"steps": [imp]}).validate_graph()
    vc2 = [
        f
        for f in (report.errors + report.warnings)
        if f.rule_id == "V-C2" and f.step_name == "RunChild"
    ]
    assert vc2, report.model_dump()
    assert any(f.severity == "error" for f in vc2)


def test_v_c3_large_literal_templates_todo() -> None:
    """V-C3: Large literal via repetition inside template should warn."""

    async def passthrough(x: str) -> str:  # type: ignore[override]
        return x

    s = Step.from_callable(passthrough, name="s")
    # Large literal without tokens triggers coarse length-based V-C3
    s.meta["templated_input"] = "x" * 60000
    report = Pipeline.from_step(s).validate_graph()
    assert any(w.rule_id == "V-C3" for w in report.warnings), report.model_dump()
