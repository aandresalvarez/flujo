from __future__ import annotations

from flujo.domain.dsl.pipeline import Pipeline


def test_yaml_pipeline_name_is_propagated() -> None:
    yaml_text = (
        "version: '0.1'\n"
        "name: minimal_gpt5_test\n"
        "steps:\n"
        "  - kind: hitl\n"
        "    name: get_name\n"
        "    message: What is your name?\n"
    )
    pipe = Pipeline.from_yaml_text(yaml_text)
    # The loader attaches a dynamic attribute `name` so CLI can pick it up
    assert getattr(pipe, "name", None) == "minimal_gpt5_test"  # noqa: S101
