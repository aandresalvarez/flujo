from typing import Optional, Any
from ..domain.models import PipelineResult
from ..domain.validation import ValidationResult


def assert_validator_failed(
    result: PipelineResult[Any],
    validator_name: str,
    expected_feedback_part: Optional[str] = None,
) -> None:
    """Assert that a specific validator failed during the run."""
    ctx = result.final_pipeline_context
    if ctx is None or not hasattr(ctx, "validation_history"):
        raise AssertionError("validation_history not found in pipeline context")

    history = getattr(ctx, "validation_history")
    found = False
    for item in history:
        if not isinstance(item, ValidationResult):
            continue
        if item.validator_name == validator_name and not item.is_valid:
            found = True
            if expected_feedback_part:
                assert expected_feedback_part in (item.feedback or "")
            break
    assert found, f"Validator '{validator_name}' did not fail as expected."


def assert_context_updated(result: PipelineResult[Any], **expected_updates: Any) -> None:
    """Assert that the final context contains the expected updates."""
    final_context = result.final_pipeline_context
    if final_context is None:
        raise AssertionError("final_pipeline_context not found in result")

    for key, expected_value in expected_updates.items():
        actual_value = getattr(final_context, key, None)
        assert actual_value == expected_value, (
            f"Expected {key} to be {expected_value}, but got {actual_value}"
        )
