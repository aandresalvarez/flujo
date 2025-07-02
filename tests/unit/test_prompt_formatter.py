from flujo.domain.models import BaseModel
from flujo.utils import format_prompt


class Person(BaseModel):
    name: str
    email: str | None = None


def test_baseline_placeholder_and_json() -> None:
    template = "Hello {{ name }}! Data: {{ person }}"
    person = Person(name="Alice", email="a@example.com")
    result = format_prompt(template, name="World", person=person)
    assert result == f"Hello World! Data: {person.model_dump_json()}"


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
