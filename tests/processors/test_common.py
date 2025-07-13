"""Unit tests for common processors."""

import pytest

from flujo.processors.common import (
    AddContextVariables,
    EnforceJsonResponse,
    StripMarkdownFences,
)


class TestAddContextVariables:
    """Test AddContextVariables processor."""

    @pytest.mark.asyncio
    async def test_add_context_variables_basic(self):
        """Test basic AddContextVariables functionality."""
        processor = AddContextVariables(vars=["key1", "key2"])

        # Test with context that has the variables
        context = type("MockContext", (), {"key1": "value1", "key2": "value2"})()

        result = await processor.process("input", context)

        expected = "--- CONTEXT ---\nkey1: value1\nkey2: value2\n---\ninput"
        assert result == expected

    @pytest.mark.asyncio
    async def test_add_context_variables_no_context(self):
        """Test AddContextVariables when context is None."""
        processor = AddContextVariables(vars=["key1"])

        result = await processor.process("input", None)

        # Should return input unchanged when context is None
        assert result == "input"

    @pytest.mark.asyncio
    async def test_add_context_variables_empty_vars(self):
        """Test AddContextVariables with empty vars list."""
        processor = AddContextVariables(vars=[])

        context = type("MockContext", (), {})()

        result = await processor.process("input", context)

        expected = "--- CONTEXT ---\n---\ninput"
        assert result == expected

    @pytest.mark.asyncio
    async def test_add_context_variables_missing_vars(self):
        """Test AddContextVariables with missing context variables."""
        processor = AddContextVariables(vars=["key1", "key2"])

        # Context only has key1, key2 is missing
        context = type("MockContext", (), {"key1": "value1"})()

        result = await processor.process("input", context)

        expected = "--- CONTEXT ---\nkey1: value1\nkey2: None\n---\ninput"
        assert result == expected

    @pytest.mark.asyncio
    async def test_add_context_variables_complex_values(self):
        """Test AddContextVariables with complex values."""
        processor = AddContextVariables(
            vars=["string", "number", "list", "dict", "boolean", "none"]
        )

        context = type(
            "MockContext",
            (),
            {
                "string": "test",
                "number": 42,
                "list": [1, 2, 3],
                "dict": {"nested": "value"},
                "boolean": True,
                "none": None,
            },
        )()

        result = await processor.process("input", context)

        expected = "--- CONTEXT ---\nstring: test\nnumber: 42\nlist: [1, 2, 3]\ndict: {'nested': 'value'}\nboolean: True\nnone: None\n---\ninput"
        assert result == expected


class TestStripMarkdownFences:
    """Test StripMarkdownFences processor."""

    @pytest.mark.asyncio
    async def test_strip_markdown_fences_basic(self):
        """Test basic StripMarkdownFences functionality."""
        processor = StripMarkdownFences(language="python")

        input_data = "```python\nprint('hello')\n```"
        result = await processor.process(input_data, None)

        assert result == "print('hello')"

    @pytest.mark.asyncio
    async def test_strip_markdown_fences_with_language(self):
        """Test StripMarkdownFences with language specification."""
        processor = StripMarkdownFences(language="json")

        input_data = '```json\n{"key": "value"}\n```'
        result = await processor.process(input_data, None)

        assert result == '{"key": "value"}'

    @pytest.mark.asyncio
    async def test_strip_markdown_fences_no_fences(self):
        """Test StripMarkdownFences with text that has no fences."""
        processor = StripMarkdownFences(language="python")

        input_data = "This is just plain text"
        result = await processor.process(input_data, None)

        assert result == input_data

    @pytest.mark.asyncio
    async def test_strip_markdown_fences_partial_fences(self):
        """Test StripMarkdownFences with partial fences."""
        processor = StripMarkdownFences(language="python")

        input_data = "```python\nprint('hello')"
        result = await processor.process(input_data, None)

        assert result == input_data

    @pytest.mark.asyncio
    async def test_strip_markdown_fences_nested_fences(self):
        """Test StripMarkdownFences with nested fences."""
        processor = StripMarkdownFences(language="python")

        input_data = "```python\n```python\nprint('hello')\n```\n```"
        result = await processor.process(input_data, None)

        # Should match the first complete fence
        assert result == "```python\nprint('hello')"

    @pytest.mark.asyncio
    async def test_strip_markdown_fences_multiple_blocks(self):
        """Test StripMarkdownFences with multiple code blocks."""
        processor = StripMarkdownFences(language="python")

        input_data = "```python\nprint('first')\n```\n```python\nprint('second')\n```"
        result = await processor.process(input_data, None)

        # Should match the first block
        assert result == "print('first')"

    @pytest.mark.asyncio
    async def test_strip_markdown_fences_empty_block(self):
        """Test StripMarkdownFences with empty code block."""
        processor = StripMarkdownFences(language="python")

        input_data = "```python\n\n```"
        result = await processor.process(input_data, None)

        assert result == ""

    @pytest.mark.asyncio
    async def test_strip_markdown_fences_whitespace_handling(self):
        """Test StripMarkdownFences with various whitespace scenarios."""
        processor = StripMarkdownFences(language="python")

        input_data = "```python\n  print('hello')  \n```"
        result = await processor.process(input_data, None)

        assert result == "print('hello')"

    @pytest.mark.asyncio
    async def test_strip_markdown_fences_special_characters(self):
        """Test StripMarkdownFences with special characters."""
        processor = StripMarkdownFences(language="python")

        input_data = "```python\nprint('hello\nworld')\n```"
        result = await processor.process(input_data, None)

        assert result == "print('hello\nworld')"

    @pytest.mark.asyncio
    async def test_strip_markdown_fences_non_string_input(self):
        """Test StripMarkdownFences with non-string input."""
        processor = StripMarkdownFences(language="python")

        input_data = 42
        result = await processor.process(input_data, None)

        assert result == input_data


class TestEnforceJsonResponse:
    """Test EnforceJsonResponse processor."""

    @pytest.mark.asyncio
    async def test_enforce_json_response_dict_input(self):
        """Test EnforceJsonResponse with dict input."""
        processor = EnforceJsonResponse()

        input_data = {"key": "value", "number": 42}
        result = await processor.process(input_data, None)

        assert result == input_data

    @pytest.mark.asyncio
    async def test_enforce_json_response_list_input(self):
        """Test EnforceJsonResponse with list input."""
        processor = EnforceJsonResponse()

        input_list = [1, "test", {"key": "value"}, [1, 2, 3]]
        result = await processor.process(input_list, None)

        assert result == input_list

    @pytest.mark.asyncio
    async def test_enforce_json_response_string_json(self):
        """Test EnforceJsonResponse with JSON string."""
        processor = EnforceJsonResponse()

        input_data = '{"key": "value", "number": 42}'
        result = await processor.process(input_data, None)

        assert result == {"key": "value", "number": 42}

    @pytest.mark.asyncio
    async def test_enforce_json_response_string_list_json(self):
        """Test EnforceJsonResponse with JSON string containing list."""
        processor = EnforceJsonResponse()

        input_data = '[1, 2, 3, "test"]'
        result = await processor.process(input_data, None)

        assert result == [1, 2, 3, "test"]

    @pytest.mark.asyncio
    async def test_enforce_json_response_invalid_json(self):
        """Test EnforceJsonResponse with invalid JSON string."""
        processor = EnforceJsonResponse()

        input_data = '{"key": "value", "number": 42'  # Missing closing brace
        result = await processor.process(input_data, None)

        # Should return the original string when JSON parsing fails
        assert result == input_data

    @pytest.mark.asyncio
    async def test_enforce_json_response_plain_string(self):
        """Test EnforceJsonResponse with plain string (not JSON)."""
        processor = EnforceJsonResponse()

        input_data = "This is not JSON"
        result = await processor.process(input_data, None)

        # Should return the original string when it's not valid JSON
        assert result == input_data

    @pytest.mark.asyncio
    async def test_enforce_json_response_none_input(self):
        """Test EnforceJsonResponse with None input."""
        processor = EnforceJsonResponse()

        result = await processor.process(None, None)

        assert result is None

    @pytest.mark.asyncio
    async def test_enforce_json_response_empty_string(self):
        """Test EnforceJsonResponse with empty string."""
        processor = EnforceJsonResponse()

        result = await processor.process("", None)

        assert result == ""

    @pytest.mark.asyncio
    async def test_enforce_json_response_whitespace_string(self):
        """Test EnforceJsonResponse with whitespace-only string."""
        processor = EnforceJsonResponse()

        result = await processor.process("   ", None)

        assert result == "   "

    @pytest.mark.asyncio
    async def test_enforce_json_response_complex_json(self):
        """Test EnforceJsonResponse with complex JSON structure."""
        processor = EnforceJsonResponse()

        complex_json = """
        {
            "string": "test",
            "number": 42,
            "boolean": true,
            "null": null,
            "array": [1, 2, 3],
            "object": {
                "nested": "value",
                "deep": {
                    "key": "value"
                }
            }
        }
        """

        result = await processor.process(complex_json, None)

        assert isinstance(result, dict)
        assert result["string"] == "test"
        assert result["number"] == 42
        assert result["boolean"] is True
        assert result["null"] is None
        assert result["array"] == [1, 2, 3]
        assert result["object"]["nested"] == "value"
        assert result["object"]["deep"]["key"] == "value"

    @pytest.mark.asyncio
    async def test_enforce_json_response_numeric_input(self):
        """Test EnforceJsonResponse with numeric input."""
        processor = EnforceJsonResponse()

        # Test with integer
        result = await processor.process(42, None)
        assert result == 42

        # Test with float
        result = await processor.process(3.14, None)
        assert result == 3.14

    @pytest.mark.asyncio
    async def test_enforce_json_response_boolean_input(self):
        """Test EnforceJsonResponse with boolean input."""
        processor = EnforceJsonResponse()

        # Test with True
        result = await processor.process(True, None)
        assert result is True

        # Test with False
        result = await processor.process(False, None)
        assert result is False

    @pytest.mark.asyncio
    async def test_enforce_json_response_json_with_unicode(self):
        """Test EnforceJsonResponse with JSON containing unicode characters."""
        processor = EnforceJsonResponse()

        input_data = '{"message": "Hello, 世界!", "emoji": "🚀"}'
        result = await processor.process(input_data, None)

        assert result["message"] == "Hello, 世界!"
        assert result["emoji"] == "🚀"

    @pytest.mark.asyncio
    async def test_enforce_json_response_malformed_json_edge_cases(self):
        """Test EnforceJsonResponse with various malformed JSON edge cases."""
        processor = EnforceJsonResponse()

        # Test with trailing comma
        input_data = '{"key": "value",}'
        result = await processor.process(input_data, None)
        assert result == input_data

        # Test with missing quotes
        input_data = '{key: "value"}'
        result = await processor.process(input_data, None)
        assert result == input_data

        # Test with unclosed brackets
        input_data = '{"key": "value"'
        result = await processor.process(input_data, None)
        assert result == input_data


class TestProcessorIntegration:
    """Test integration between different processors."""

    @pytest.mark.asyncio
    async def test_processors_chain(self):
        """Test chaining multiple processors."""
        # Create processors
        add_vars = AddContextVariables(vars=["product"])
        strip_fences = StripMarkdownFences(language="text")
        enforce_json = EnforceJsonResponse()

        # Create context
        context = type("MockContext", (), {"product": "Widget"})()

        # Test input with markdown fences and JSON
        input_data = '```text\n{"key": "value"}\n```'

        # Process through chain
        result1 = await strip_fences.process(input_data, context)
        result2 = await enforce_json.process(result1, context)
        result3 = await add_vars.process(result2, context)

        assert result1 == '{"key": "value"}'
        assert result2 == {"key": "value"}
        expected = "--- CONTEXT ---\nproduct: Widget\n---\n{'key': 'value'}"
        assert result3 == expected

    @pytest.mark.asyncio
    async def test_processors_with_context(self):
        """Test processors with context handling."""
        add_vars = AddContextVariables(vars=["step"])
        context = type("MockContext", (), {"step": "test"})()

        # Test that context is properly updated
        result = await add_vars.process("input", context)

        expected = "--- CONTEXT ---\nstep: test\n---\ninput"
        assert result == expected

    @pytest.mark.asyncio
    async def test_processors_error_handling(self):
        """Test processors with error handling."""
        # Test AddContextVariables with None context
        add_vars = AddContextVariables(vars=["key"])
        result = await add_vars.process("input", None)
        assert result == "input"

        # Test StripMarkdownFences with None input
        strip_fences = StripMarkdownFences(language="python")
        result = await strip_fences.process(None, None)
        assert result is None

        # Test EnforceJsonResponse with invalid JSON
        enforce_json = EnforceJsonResponse()
        result = await enforce_json.process('{"invalid": json}', None)
        assert result == '{"invalid": json}'


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_add_context_variables_missing_context_attribute(self):
        """Test AddContextVariables with context missing requested variables."""
        processor = AddContextVariables(vars=["missing_key"])

        # Create context without the requested variable
        context = type("MockContext", (), {})()

        result = await processor.process("input", context)
        expected = "--- CONTEXT ---\nmissing_key: None\n---\ninput"
        assert result == expected

    @pytest.mark.asyncio
    async def test_strip_markdown_fences_edge_cases(self):
        """Test StripMarkdownFences with edge cases."""
        processor = StripMarkdownFences(language="python")

        # Test with only backticks
        result = await processor.process("```", None)
        assert result == "```"

        # Test with empty string
        result = await processor.process("", None)
        assert result == ""

        # Test with whitespace only
        result = await processor.process("   ", None)
        assert result == "   "

    @pytest.mark.asyncio
    async def test_enforce_json_response_edge_cases(self):
        """Test EnforceJsonResponse with edge cases."""
        processor = EnforceJsonResponse()

        # Test with empty JSON object
        result = await processor.process("{}", None)
        assert result == {}

        # Test with empty JSON array
        result = await processor.process("[]", None)
        assert result == []

        # Test with null JSON
        result = await processor.process("null", None)
        assert result is None  # Should return None since it's not a dict/list

        # Test with boolean JSON
        result = await processor.process("true", None)
        assert result is True  # Should return boolean since it's not a dict/list

        # Test with number JSON
        result = await processor.process("42", None)
        assert result == 42  # Should return number since it's not a dict/list
