from __future__ import annotations

from flujo.domain.dsl import Pipeline, Step


def test_v_l1_loop_exit_coverage_warns() -> None:
    """V-L1: Loop with side-effect-free body and no mappers should warn."""

    async def body(_: str) -> str:  # type: ignore[override]
        return "keep"

    s_body = Step.from_callable(body, name="body")
    loop_body = Pipeline.model_validate({"steps": [s_body]})

    from flujo.domain.dsl.loop import LoopStep

    # Exit condition depends on output, but body output is constant; statically we warn due to missing mappers/updates
    loop = LoopStep.model_validate(
        {
            "name": "loop",
            "loop_body_pipeline": loop_body,
            "exit_condition_callable": lambda o, _c: (isinstance(o, str) and o == "stop"),
            "max_retries": 5,
        }
    )
    report = Pipeline.model_validate({"steps": [loop]}).validate_graph()
    assert any(w.rule_id == "V-L1" for w in report.warnings), report.model_dump()


def test_v_p2_parallel_explicit_conflicts_detected() -> None:
    """V-P2: Two branches import and map to the same parent path → warn.

    Construct a ParallelStep programmatically with two branches each containing
    an ImportStep that maps a child value to the same parent key.
    """
    from flujo.domain.dsl.import_step import ImportStep, OutputMapping
    from flujo.domain.dsl.parallel import ParallelStep

    # Branch A: child pipeline with a simple step
    a_child = Pipeline.model_validate(
        {
            "steps": [
                Step.model_validate(
                    {
                        "name": "a1",
                        "agent": "flujo.builtins.stringify",
                        "meta": {"templated_input": "A"},
                    }
                )
            ],
        }
    )
    a_import = ImportStep.model_validate(
        {
            "name": "runA",
            "pipeline": a_child,
            "updates_context": True,
            "outputs": [OutputMapping(child="import_artifacts.value", parent="import_artifacts.k")],
        }
    )
    # Branch B
    b_child = Pipeline.model_validate(
        {
            "steps": [
                Step.model_validate(
                    {
                        "name": "b1",
                        "agent": "flujo.builtins.stringify",
                        "meta": {"templated_input": "B"},
                    }
                )
            ],
        }
    )
    b_import = ImportStep.model_validate(
        {
            "name": "runB",
            "pipeline": b_child,
            "updates_context": True,
            "outputs": [OutputMapping(child="import_artifacts.value", parent="import_artifacts.k")],
        }
    )

    p = ParallelStep.model_validate(
        {
            "name": "P",
            "branches": {
                "A": Pipeline.model_validate({"steps": [a_import]}),
                "B": Pipeline.model_validate({"steps": [b_import]}),
            },
        }
    )
    pipeline = Pipeline.model_validate({"steps": [p]})
    report = pipeline.validate_graph()
    assert any(w.rule_id == "V-P2" for w in report.warnings), report.model_dump()
