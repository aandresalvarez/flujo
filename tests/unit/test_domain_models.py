from pydantic_ai_orchestrator.domain.models import (
    ImprovementReport,
    ImprovementSuggestion,
)


def test_improvement_models_round_trip():
    suggestion = ImprovementSuggestion(
        target_step_name="solve",
        failure_pattern="missing key",
        suggested_change="Add api_key",
        example_failing_cases=["case1"],
        suggested_config_change="temperature=0.5",
        suggested_new_test_case="test_missing_key",
    )
    report = ImprovementReport(suggestions=[suggestion])
    data = report.model_dump()
    loaded = ImprovementReport.model_validate(data)
    assert loaded.suggestions[0].target_step_name == "solve"
