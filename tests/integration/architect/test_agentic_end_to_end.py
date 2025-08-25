from __future__ import annotations

import pytest


@pytest.mark.integration
def test_agentic_end_to_end_with_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure agentic path is enabled
    # Enable full state machine path explicitly
    monkeypatch.setenv("FLUJO_ARCHITECT_STATE_MACHINE", "1")
    # Ensure agentic features are enabled
    monkeypatch.delenv("FLUJO_ARCHITECT_MINIMAL", raising=False)
    monkeypatch.delenv("FLUJO_ARCHITECT_AGENTIC_PLANNER", raising=False)
    monkeypatch.delenv("FLUJO_ARCHITECT_AGENTIC_TOOLMATCHER", raising=False)
    monkeypatch.delenv("FLUJO_ARCHITECT_AGENTIC_YAMLWRITER", raising=False)

    import flujo.builtins as _builtins  # noqa: F401 - ensure stubs registered

    # Use internal state machine builder directly to avoid env coupling
    from flujo.architect.builder import _build_state_machine_pipeline as build_pipe  # type: ignore
    from flujo.architect.context import ArchitectContext
    from flujo.cli.helpers import create_flujo_runner, execute_pipeline_with_output_handling

    pipeline = build_pipe()
    goal = "Search the web and save result to file in parallel"
    initial = {"user_goal": goal, "non_interactive": True}
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )
    result = execute_pipeline_with_output_handling(
        runner=runner, input_data=goal, run_id=None, json_output=False
    )

    ctx = getattr(result, "final_pipeline_context", None)
    assert ctx is not None
    yaml_text = getattr(ctx, "yaml_text", None)
    assert isinstance(yaml_text, str)
    # Keep assertions shape-agnostic to avoid overfitting fallback paths
    assert "version:" in yaml_text and "steps:" in yaml_text
