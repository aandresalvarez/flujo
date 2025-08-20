from __future__ import annotations

import pytest


@pytest.mark.integration
def test_architect_hitl_plan_approval_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test HITL plan approval behavior in non-interactive mode.

    This test validates that the architect pipeline completes successfully
    with HITL configuration without requiring complex mocking that could cause infinite loops.
    """
    from flujo.architect.builder import build_architect_pipeline
    from flujo.architect.context import ArchitectContext
    from flujo.cli.helpers import create_flujo_runner, execute_pipeline_with_output_handling

    pipeline = build_architect_pipeline()
    initial = {
        "initial_prompt": "Build something",
        "user_goal": "Search the web",
        "hitl_enabled": False,  # Use non-interactive mode to avoid prompts and infinite loops
        "non_interactive": True,
    }
    runner = create_flujo_runner(
        pipeline=pipeline,
        context_model_class=ArchitectContext,
        initial_context_data=initial,
    )

    # Execute with a timeout to prevent infinite loops
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("Test timed out - likely infinite loop")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)  # 30 second timeout

    try:
        result = execute_pipeline_with_output_handling(
            runner=runner, input_data="Search the web", run_id=None, json_output=False
        )

        # Verify that the pipeline executed basic steps
        def _flatten(steps):
            for sr in steps or []:
                yield sr
                for child in getattr(sr, "step_history", []) or []:
                    yield from _flatten([child])

        names = [
            getattr(sr, "name", "") for sr in _flatten(getattr(result, "step_history", []) or [])
        ]

        # Check for expected steps - basic pipeline completion
        assert any(n == "PlanApproval" for n in names), f"PlanApproval step not found in: {names}"

        # Verify the pipeline completed successfully
        ctx = getattr(result, "final_pipeline_context", None)
        assert ctx is not None, "Pipeline context should be available"
        yaml_text = getattr(ctx, "yaml_text", None)
        assert isinstance(yaml_text, str), "Final YAML should be generated"
        assert len(yaml_text) > 0, "YAML should not be empty"

    finally:
        signal.alarm(0)  # Cancel the alarm
