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


def test_improvement_models_config_and_new_case() -> None:
    suggestion = ImprovementSuggestion(
        suggestion_type=SuggestionType.CONFIG_ADJUSTMENT,
        failure_pattern_summary="f",
        detailed_explanation="d",
        config_change_details=[
            {
                "parameter_name": "temperature",
                "suggested_value": "0.1",
                "reasoning": "more deterministic",
            }
        ],
        suggested_new_eval_case_description="Add join query case",
    )
    report = ImprovementReport(suggestions=[suggestion])
    dumped = report.model_dump()
    loaded = ImprovementReport.model_validate(dumped)
    assert loaded.suggestions[0].config_change_details is not None
    assert loaded.suggestions[0].suggested_new_eval_case_description == "Add join query case"


def test_global_custom_serializer_registry():
    """Test that custom serializers work with the new robust serialization approach."""
    from flujo.utils.serialization import register_custom_serializer
    from flujo.domain.models import BaseModel

    class Custom:
        def __init__(self, value):
            self.value = value

    class MyModel(BaseModel):
        foo: Custom
        bar: complex
        model_config = {"arbitrary_types_allowed": True}

    def custom_serializer(obj):
        if isinstance(obj, Custom):
            return f"custom:{obj.value}"
        if isinstance(obj, complex):
            return f"{obj.real}+{obj.imag}j"
        raise TypeError(f"Cannot serialize {type(obj)}")

    # Register the custom serializer globally
    register_custom_serializer(Custom, custom_serializer)
    register_custom_serializer(complex, custom_serializer)

    m = MyModel(foo=Custom(42), bar=3 + 4j)
    # Use model_dump(mode="json", fallback=custom_serializer) to trigger the global registry and fallback
    serialized = m.model_dump(mode="json", fallback=custom_serializer)
    assert serialized["foo"] == "custom:42"
    assert serialized["bar"] == "3+4j"
