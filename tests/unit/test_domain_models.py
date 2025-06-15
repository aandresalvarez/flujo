from flujo.domain.models import (
    ImprovementSuggestion,
    ImprovementReport,
    SuggestionType,
    PromptModificationDetail,
)


def test_improvement_models_round_trip() -> None:
    suggestion = ImprovementSuggestion(
        target_step_name="step",
        suggestion_type=SuggestionType.PROMPT_MODIFICATION,
        failure_pattern_summary="fails",
        detailed_explanation="explain",
        prompt_modification_details=PromptModificationDetail(modification_instruction="Add foo"),
        example_failing_input_snippets=["snippet"],
        estimated_impact="HIGH",
        estimated_effort_to_implement="LOW",
    )
    report = ImprovementReport(suggestions=[suggestion])
    data = report.model_dump()
    loaded = ImprovementReport.model_validate(data)
    assert loaded.suggestions[0].prompt_modification_details is not None


def test_improvement_models_validation() -> None:
    # missing required fields should raise
    try:
        ImprovementSuggestion(suggestion_type=SuggestionType.OTHER)
    except Exception as e:
        assert isinstance(e, Exception)
    else:
        assert False, "Validation should fail"
