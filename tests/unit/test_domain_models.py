from typing import List, Optional

from pydantic import Field

from flujo.domain.models import (
    BaseModel,
    ImprovementSuggestion,
    ImprovementReport,
    SuggestionType,
    PromptModificationDetail,
)


class Node(BaseModel):
    """Simple recursive model for circular reference testing."""

    name: str
    parent: Optional["Node"] = None
    children: List["Node"] = Field(default_factory=list)


Node.model_rebuild()


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
        # model_config inherited from BaseModel

    def custom_serializer(obj):
        if isinstance(obj, Custom):
            return f"custom:{obj.value}"
        if isinstance(obj, complex):
            real = int(obj.real) if obj.real == int(obj.real) else obj.real
            imag = int(obj.imag) if obj.imag == int(obj.imag) else obj.imag
            return f"{real}+{imag}j"
        raise TypeError(f"Cannot serialize {type(obj)}")

    # Register the custom serializer globally
    register_custom_serializer(Custom, custom_serializer)
    register_custom_serializer(complex, custom_serializer)

    m = MyModel(foo=Custom(42), bar=3 + 4j)
    # Use model_dump to trigger the global registry
    serialized = m.model_dump(mode="cache")
    assert serialized["foo"] == "custom:42"
    assert serialized["bar"] == "3+4j"


def test_base_model_circular_reference_serialization() -> None:
    """Ensure BaseModel.model_dump handles circular references correctly."""

    # Scenario A: direct parent-child loop
    parent = Node(name="parent")
    child = Node(name="child", parent=parent)
    parent.children.append(child)

    dumped_default = parent.model_dump(mode="default")
    child_dump_default = dumped_default["children"][0]
    assert child_dump_default["parent"] is None

    dumped_cache = parent.model_dump(mode="cache")
    child_dump_cache = dumped_cache["children"][0]
    assert child_dump_cache["parent"] == "<Node circular>"

    # Scenario B: self reference
    node = Node(name="self")
    node.parent = node

    dumped_default = node.model_dump(mode="default")
    assert dumped_default["parent"] is None

    dumped_cache = node.model_dump(mode="cache")
    assert dumped_cache["parent"] == "<Node circular>"

    # Scenario C: deeply nested loop
    root = Node(name="root")
    child1 = Node(name="child1", parent=root)
    grandchild = Node(name="grandchild", parent=child1)
    grandchild.children.append(root)
    root.children.append(child1)
    child1.children.append(grandchild)

    dumped_default = root.model_dump(mode="default")
    grandchild_default = dumped_default["children"][0]["children"][0]
    assert grandchild_default["children"][0] is None

    dumped_cache = root.model_dump(mode="cache")
    grandchild_cache = dumped_cache["children"][0]["children"][0]
    assert grandchild_cache["children"][0] == "<Node circular>"

    # Scenario D: no circular references
    root_no_loop = Node(name="root")
    child_no_loop = Node(name="child")
    root_no_loop.children.append(child_no_loop)

    default_out = root_no_loop.model_dump(mode="default")
    cache_out = root_no_loop.model_dump(mode="cache")
    assert default_out == cache_out
    assert default_out["children"][0]["parent"] is None
