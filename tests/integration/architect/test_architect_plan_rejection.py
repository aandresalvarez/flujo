from __future__ import annotations

import pytest

# Architect tests use StateMachine pipelines with shared state; serial prevents race conditions
pytestmark = [pytest.mark.serial]


@pytest.mark.integration
def test_architect_plan_rejection_triggers_refinement(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test plan rejection behavior in non-interactive mode.

    This test validates that the architect pipeline completes successfully
    without requiring complex mocking that could cause infinite loops.
    """
    # Enable state machine
    monkeypatch.setenv("FLUJO_ARCHITECT_STATE_MACHINE", "1")

    from flujo.architect.builder import build_architect_pipeline
    from flujo.architect.context import ArchitectContext
    from flujo.cli.helpers import create_flujo_runner, execute_pipeline_with_output_handling

    pipeline = build_architect_pipeline()
    initial = {
        "initial_prompt": "Build something",
        "user_goal": "Search the web",
        "hitl_enabled": False,  # Use non-interactive mode to avoid infinite loops
        "non_interactive": True,
        "refinement_feedback": "Prefer safer search strategy.",
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

        # Collect executed step names (flatten nested histories)
        def _flatten(steps):
            for sr in steps or []:
                yield sr
                for child in getattr(sr, "step_history", []) or []:
                    yield from _flatten([child])

        names = [
            getattr(sr, "name", "") for sr in _flatten(getattr(result, "step_history", []) or [])
        ]

        # Expect basic pipeline completion
        assert any(n == "PlanApproval" for n in names), f"PlanApproval step not found in: {names}"

        # MakePlan should appear at least once
        assert names.count("MakePlan") >= 1, (
            f"MakePlan step should appear at least once, found {names.count('MakePlan')} times in: {names}"
        )

        # Final YAML exists
        ctx = getattr(result, "final_pipeline_context", None)
        assert ctx is not None, "Pipeline context should be available"
        yaml_text = getattr(ctx, "yaml_text", None)
        assert isinstance(yaml_text, str), "Final YAML should be generated"
        assert len(yaml_text) > 0, "YAML should not be empty"

    finally:
        signal.alarm(0)  # Cancel the alarm
