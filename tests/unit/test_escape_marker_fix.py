"""Comprehensive tests for the escape marker collision fix."""

import os
from flujo.utils import format_prompt
from typing import Optional
from flujo.domain.models import BaseModel


class Person(BaseModel):
    name: str
    email: Optional[str] = None


class TestEscapeMarkerCollisionFix:
    """Test suite for the escape marker collision fix."""

    def test_literal_escape_marker_in_user_input(self):
        """Test that literal '__ESCAPED_OPEN__' in user input is preserved."""
        template = "User input: {{ user_input }}"
        user_input = "__ESCAPED_OPEN__"

        result = format_prompt(template, user_input=user_input)

        # The literal string should be preserved, not converted to {{
        assert result == f"User input: {user_input}"
        assert "__ESCAPED_OPEN__" in result
        assert "{{" not in result

    def test_template_syntax_in_user_input(self):
        """Test that {{ in user input is properly escaped and restored."""
        template = "User input: {{ user_input }}"
        user_input = "Some text with {{ template syntax }}"

        result = format_prompt(template, user_input=user_input)

        # The {{ should be preserved as literal text
        assert result == f"User input: {user_input}"
        assert "{{ template syntax }}" in result

    def test_each_block_with_literal_escape_marker(self):
        """Test each blocks with items containing literal escape marker."""
        template = "Items:\n{{#each items}}- {{ this }}\n{{/each}}"
        items = ["Normal item", "__ESCAPED_OPEN__", "Another item"]

        result = format_prompt(template, items=items)

        # All items should be preserved, including the literal escape marker
        assert "- Normal item" in result
        assert "- __ESCAPED_OPEN__" in result
        assert "- Another item" in result
        assert "__ESCAPED_OPEN__" in result
        # Should not have any unintended {{ conversions
        assert result.count("{{") == 0

    def test_nested_object_with_literal_escape_marker(self):
        """Test nested objects with literal escape marker values."""
        template = "User: {{ user.name }} ({{ user.description }})"
        user = {"name": "Alice", "description": "__ESCAPED_OPEN__"}

        result = format_prompt(template, user=user)

        # The literal escape marker should be preserved
        assert "User: Alice (__ESCAPED_OPEN__)" in result
        assert "__ESCAPED_OPEN__" in result
        # Should not have any unintended {{ conversions
        assert result.count("{{") == 0

    def test_mixed_content_with_literal_and_template_syntax(self):
        """Test mixed content with both literal escape marker and template syntax."""
        template = "Content: {{ content }}"
        content = "Start __ESCAPED_OPEN__ middle {{ variable }} end"

        result = format_prompt(template, content=content)

        # Both the literal escape marker and template syntax should be preserved
        assert "Start __ESCAPED_OPEN__ middle {{ variable }} end" in result
        assert "__ESCAPED_OPEN__" in result
        assert "{{ variable }}" in result

    def test_multiple_literal_escape_markers(self):
        """Test multiple occurrences of literal escape marker."""
        template = "Content: {{ content }}"
        content = "First __ESCAPED_OPEN__ Second __ESCAPED_OPEN__ Third"

        result = format_prompt(template, content=content)

        # All literal escape markers should be preserved
        assert "First __ESCAPED_OPEN__ Second __ESCAPED_OPEN__ Third" in result
        assert result.count("__ESCAPED_OPEN__") == 2
        assert result.count("{{") == 0

    def test_escape_marker_in_complex_objects(self):
        """Test escape marker in complex nested objects."""
        template = "Data: {{ data }}"
        data = {
            "level1": {
                "level2": {
                    "value": "__ESCAPED_OPEN__",
                    "list": ["item1", "__ESCAPED_OPEN__", "item3"],
                }
            }
        }

        result = format_prompt(template, data=data)

        # The escape marker should be preserved in the serialized JSON
        assert "__ESCAPED_OPEN__" in result
        assert result.count("{{") == 0

    def test_escape_marker_in_base_model(self):
        """Test escape marker in Pydantic BaseModel."""
        template = "Person: {{ person }}"
        person = Person(name="John", email="__ESCAPED_OPEN__")

        result = format_prompt(template, person=person)

        # The escape marker should be preserved in the serialized model
        assert "__ESCAPED_OPEN__" in result
        assert result.count("{{") == 0

    def test_conditional_blocks_with_escape_marker(self):
        """Test conditional blocks with escape marker content."""
        template = "{{#if user}}User: {{ user.name }} ({{ user.description }}){{/if}}"
        user = {"name": "Bob", "description": "__ESCAPED_OPEN__"}

        result = format_prompt(template, user=user)

        # The escape marker should be preserved
        assert "User: Bob (__ESCAPED_OPEN__)" in result
        assert "__ESCAPED_OPEN__" in result
        assert result.count("{{") == 0

    def test_nested_conditional_blocks(self):
        """Test nested conditional blocks with escape markers."""
        template = """
        {{#if user}}
        User: {{ user.name }}
        {{#if user.description}}
        Description: {{ user.description }}
        {{/if}}
        {{/if}}
        """
        user = {"name": "Alice", "description": "__ESCAPED_OPEN__"}

        result = format_prompt(template, user=user)

        # The escape marker should be preserved
        assert "User: Alice" in result
        assert "Description: __ESCAPED_OPEN__" in result
        assert "__ESCAPED_OPEN__" in result
        assert result.count("{{") == 0

    def test_each_block_with_nested_objects(self):
        """Test each blocks with nested objects containing escape markers."""
        template = "Users:\n{{#each users}}- {{ this.name }} ({{ this.description }})\n{{/each}}"
        users = [
            {"name": "Alice", "description": "Normal description"},
            {"name": "Bob", "description": "__ESCAPED_OPEN__"},
            {"name": "Charlie", "description": "Another normal one"},
        ]

        result = format_prompt(template, users=users)

        # All descriptions should be preserved correctly
        assert "- Alice (Normal description)" in result
        assert "- Bob (__ESCAPED_OPEN__)" in result
        assert "- Charlie (Another normal one)" in result
        assert "__ESCAPED_OPEN__" in result
        assert result.count("{{") == 0

    def test_escape_marker_edge_cases(self):
        """Test various edge cases with escape markers."""
        test_cases = [
            ("Empty string", ""),
            ("Just escape marker", "__ESCAPED_OPEN__"),
            ("Escape marker at start", "__ESCAPED_OPEN__ text"),
            ("Escape marker at end", "text __ESCAPED_OPEN__"),
            ("Multiple escape markers", "__ESCAPED_OPEN__ middle __ESCAPED_OPEN__"),
            ("Escape marker with template syntax", "__ESCAPED_OPEN__ {{ variable }}"),
            ("Template syntax with escape marker", "{{ variable }} __ESCAPED_OPEN__"),
        ]

        for description, content in test_cases:
            template = "Content: {{ content }}"
            result = format_prompt(template, content=content)

            # The literal escape marker should always be preserved
            if "__ESCAPED_OPEN__" in content:
                assert "__ESCAPED_OPEN__" in result
            # Template syntax should be preserved as literal text
            if "{{ variable }}" in content:
                assert "{{ variable }}" in result

    def test_unique_escape_markers_per_instance(self):
        """Test that each formatter instance uses a unique escape marker."""
        from flujo.utils.prompting import AdvancedPromptFormatter

        formatter1 = AdvancedPromptFormatter("template1")
        formatter2 = AdvancedPromptFormatter("template2")

        # Each instance should have a different escape marker
        assert formatter1._escape_marker != formatter2._escape_marker
        assert "__ESCAPED_TEMPLATE_" in formatter1._escape_marker
        assert "__ESCAPED_TEMPLATE_" in formatter2._escape_marker

    def test_escape_marker_with_special_characters(self):
        """Test escape marker with special characters in user input."""
        template = "Content: {{ content }}"
        content = "Special chars: __ESCAPED_OPEN__ !@#$%^&*()"

        result = format_prompt(template, content=content)

        # The escape marker should be preserved with special characters
        assert "Special chars: __ESCAPED_OPEN__ !@#$%^&*()" in result
        assert "__ESCAPED_OPEN__" in result

    def test_escape_marker_in_json_strings(self):
        """Test escape marker in JSON string values."""
        template = "Data: {{ data }}"
        data = {
            "json_string": '{"key": "__ESCAPED_OPEN__", "value": "test"}',
            "normal_string": "__ESCAPED_OPEN__",
        }

        result = format_prompt(template, data=data)

        # The escape marker should be preserved in JSON strings
        assert "__ESCAPED_OPEN__" in result
        assert result.count("{{") == 0

    def test_escape_marker_with_unicode(self):
        """Test escape marker with unicode characters."""
        template = "Content: {{ content }}"
        content = "Unicode: __ESCAPED_OPEN__ 🚀 中文"

        result = format_prompt(template, content=content)

        # The escape marker should be preserved with unicode
        assert "Unicode: __ESCAPED_OPEN__ 🚀 中文" in result
        assert "__ESCAPED_OPEN__" in result

    def test_escape_marker_in_empty_and_none_values(self):
        """Test escape marker handling with empty and None values."""
        template = "Empty: {{ empty }}, None: {{ none_value }}, Normal: {{ normal }}"

        result = format_prompt(template, empty="", none_value=None, normal="__ESCAPED_OPEN__")

        # Empty and None should be handled correctly
        assert "Empty: " in result
        assert "None: " in result
        assert "Normal: __ESCAPED_OPEN__" in result
        assert "__ESCAPED_OPEN__" in result

    def test_escape_marker_regression_original_functionality(self):
        """Test that original template functionality still works."""
        # Test basic placeholder replacement
        template = "Hello {{ name }}!"
        result = format_prompt(template, name="World")
        assert result == "Hello World!"

        # Test conditional blocks
        template = "{{#if show}}Visible{{/if}}"
        result = format_prompt(template, show=True)
        assert result == "Visible"

        result = format_prompt(template, show=False)
        assert result == ""

        # Test each blocks
        template = "{{#each items}}{{ this }}{{/each}}"
        result = format_prompt(template, items=["a", "b", "c"])
        assert result == "abc"

        # Test escaping literal {{ in templates
        template = r"Literal: \{{ not a placeholder }}"
        result = format_prompt(template)
        assert result == "Literal: {{ not a placeholder }}"

    def test_escape_marker_performance(self):
        """Test that the fix doesn't significantly impact performance."""
        import time

        template = "{{#each items}}- {{ this.name }}: {{ this.description }}{{/each}}"
        items = [{"name": f"User{i}", "description": "__ESCAPED_OPEN__"} for i in range(100)]

        start_time = time.time()
        result = format_prompt(template, items=items)
        end_time = time.time()

        # Should complete in reasonable time (less than 1 second)
        threshold = float(os.getenv("TYPE_RESOLUTION_THRESHOLD", 2.0))  # Default to 2 seconds
        assert end_time - start_time < threshold

        # All escape markers should be preserved
        assert result.count("__ESCAPED_OPEN__") == 100
        assert result.count("{{") == 0
