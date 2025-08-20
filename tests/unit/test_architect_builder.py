from __future__ import annotations


def test_build_architect_pipeline_minimal() -> None:
    from flujo.architect.builder import build_architect_pipeline

    p = build_architect_pipeline()
    assert p is not None
    assert len(p.steps) == 1
    step = p.steps[0]
    assert getattr(step, "name", None) == "GenerateYAML"
