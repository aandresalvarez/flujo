from typing import Optional
from flujo.domain.models import BaseModel
from pydantic import BaseModel as PydanticModel
from flujo.utils import format_prompt


class Person(BaseModel):
    name: str
    email: Optional[str] = None


def test_baseline_placeholder_and_json() -> None:
    template = "Hello {{ name }}! Data: {{ person }}"
    person = Person(name="Alice", email="a@example.com")
    result = format_prompt(template, name="World", person=person)
    # Derive expected JSON from the actual serialization logic to keep test in sync
    from flujo.utils.serialization import _serialize_for_json
    import json

    expected_json = json.dumps(_serialize_for_json(person))
    assert result == f"Hello World! Data: {expected_json}"


def test_if_block() -> None:
    template = "User query: {{ query }}. {{#if feedback}}Previous feedback: {{ feedback }}{{/if}}"
    assert (
        format_prompt(template, query="a", feedback="It was wrong.")
        == "User query: a. Previous feedback: It was wrong."
    )
    assert format_prompt(template, query="a", feedback=None) == "User query: a. "


def test_each_block() -> None:
    template = "Items:\n{{#each examples}}- {{ this }}\n{{/each}}"
    result = format_prompt(template, examples=["A", "B"])
    assert "- A" in result and "- B" in result
    empty = format_prompt(template, examples=[])
    assert empty == "Items:\n"


def test_nested_placeholders() -> None:
    template = "User: {{ user.name }} ({{ user.email }})"
    data = {"name": "Bob", "email": "b@example.com"}
    assert format_prompt(template, user=data) == "User: Bob (b@example.com)"
    assert format_prompt(template, user={}) == "User:  ()"


def test_escaping() -> None:
    template = r"The syntax is \{{ variable_name }}."
    assert format_prompt(template) == "The syntax is {{ variable_name }}."


def test_prompt_robust_serialization() -> None:
    class Unknown:
        pass

    class Wrapper(PydanticModel):
        data: object

    template = "Value: {{ wrapper }}"
    result = format_prompt(template, wrapper=Wrapper(data=Unknown()))
    assert "<unserializable: Unknown>" in result
