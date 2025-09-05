from __future__ import annotations

import pytest


@pytest.mark.integration
def test_architect_happy_path_generates_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    # State machine enabled via module conftest fixture

    from flujo.architect.builder import build_architect_pipeline
    from flujo.architect.context import ArchitectContext
    from flujo.cli.helpers import create_flujo_runner, execute_pipeline_with_output_handling

    pipeline = build_architect_pipeline()
    # Create runner with ArchitectContext and run
    initial = {"initial_prompt": "Fetch something", "user_goal": "Fetch a web page"}
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )
    result = execute_pipeline_with_output_handling(
        runner=runner, input_data="Fetch a web page", run_id=None, json_output=False
    )

    # Assert a YAML was produced in context
    ctx = getattr(result, "final_pipeline_context", None)
    assert ctx is not None
    yaml_text = getattr(ctx, "yaml_text", None)
    assert isinstance(yaml_text, str) and "version:" in yaml_text

    # REGRESSION TEST: Ensure context updates are working correctly
    # This test will fail if the context update issue we fixed recurs
    generated_yaml = getattr(ctx, "generated_yaml", None)
    assert isinstance(generated_yaml, str), (
        f"Expected generated_yaml to be a string, got {type(generated_yaml)}"
    )
    assert "version:" in generated_yaml, (
        f"Expected generated_yaml to contain 'version:', got: {generated_yaml[:100]}"
    )
    assert yaml_text == generated_yaml, "yaml_text and generated_yaml should be identical"

    # REGRESSION TEST: Ensure yaml_is_valid is properly set
    yaml_is_valid = getattr(ctx, "yaml_is_valid", None)
    assert yaml_is_valid is not None, "yaml_is_valid should be set during validation"
    assert isinstance(yaml_is_valid, bool), (
        f"yaml_is_valid should be a boolean, got {type(yaml_is_valid)}"
    )

    # REGRESSION TEST: Ensure the pipeline completed without infinite loops
    # The test should complete in reasonable time, not hang indefinitely
    assert len(yaml_text) > 0, "Generated YAML should not be empty"

    # Ensure GenerateYAML step executed at some point (nested step history)
    def _flatten(steps):
        for sr in steps or []:
            yield sr
            for child in getattr(sr, "step_history", []) or []:
                yield from _flatten([child])

    names = [getattr(sr, "name", "") for sr in _flatten(getattr(result, "step_history", []) or [])]
    assert any(n == "GenerateYAML" for n in names)

    # Optional: assert key state order subsequence exists
    want = [
        "GotoGoalClarification",
        "GotoPlanning",
        "MakePlan",
        "PlanApproval",
        "CollectParams",
        "GenerateYAML",
        "SelectYAMLText",
        "ValidateYAML",
        "CaptureReport",
        "ValidationDecision",
    ]
    pos = 0
    for w in want:
        try:
            idx = names.index(w, pos)
        except ValueError:
            idx = -1
        assert idx != -1, f"Missing step in order: {w}"
        pos = idx + 1
