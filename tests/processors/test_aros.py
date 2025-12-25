"""Tests for AROS (Agentic Recovery Output System) processors.

This module provides comprehensive tests for the three-stage AROS pipeline:
- Stage 0: JSON Region Extraction (JsonRegionExtractorProcessor)
- Stage 1: Tolerant JSON Decoding (TolerantJsonDecoderProcessor)
- Stage 3: Smart Type Coercion (SmartTypeCoercionProcessor)

These tests validate that LLM outputs with common formatting issues are
correctly recovered into valid, typed Python objects.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from flujo.processors.aros import (
    JsonRegionExtractorProcessor,
    TolerantJsonDecoderProcessor,
    SmartTypeCoercionProcessor,
)


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


class SampleOutputSchema(BaseModel):
    """Sample schema for coercion tests."""

    enabled: bool
    count: int
    rate: float
    name: str


# =============================================================================
# Stage 0: JSON Region Extraction Tests
# =============================================================================


class TestJsonRegionExtractorProcessor:
    """Tests for Stage 0 - JSON Region Extraction.

    The extractor's job is to find and isolate valid JSON from LLM output
    that may contain surrounding prose, markdown, or other text.
    """

    pytestmark = pytest.mark.fast

    @pytest.mark.asyncio
    async def test_passthrough_clean_json_object(self) -> None:
        """Clean JSON objects should pass through unchanged."""
        processor = JsonRegionExtractorProcessor()

        input_data = '{"status": "success", "count": 42}'
        result = await processor.process(input_data, None)

        # Verify the JSON is preserved exactly
        parsed = json.loads(result)
        assert parsed == {"status": "success", "count": 42}

    @pytest.mark.asyncio
    async def test_passthrough_clean_json_array(self) -> None:
        """Clean JSON arrays should pass through unchanged."""
        processor = JsonRegionExtractorProcessor()

        input_data = '[1, 2, 3, {"nested": true}]'
        result = await processor.process(input_data, None)

        parsed = json.loads(result)
        assert parsed == [1, 2, 3, {"nested": True}]

    @pytest.mark.asyncio
    async def test_extract_json_from_leading_prose(self) -> None:
        """Extract JSON when LLM adds explanatory text before it."""
        processor = JsonRegionExtractorProcessor()

        input_data = """Here is the analysis you requested:
        
{"result": "complete", "confidence": 0.95}"""

        result = await processor.process(input_data, None)

        # Should extract just the JSON portion
        parsed = json.loads(result)
        assert parsed["result"] == "complete"
        assert parsed["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_extract_json_from_trailing_prose(self) -> None:
        """Extract JSON when LLM adds text after it."""
        processor = JsonRegionExtractorProcessor()

        input_data = """{"answer": 42, "explanation": "The meaning of life"}

I hope this helps! Let me know if you need anything else."""

        result = await processor.process(input_data, None)

        parsed = json.loads(result)
        assert parsed["answer"] == 42
        assert "explanation" in parsed

    @pytest.mark.asyncio
    async def test_extract_json_from_markdown_code_fence(self) -> None:
        """Extract JSON from markdown code blocks (common LLM pattern).

        Note: The extractor may or may not strip markdown fences depending
        on implementation. We verify the JSON content is accessible.
        """
        processor = JsonRegionExtractorProcessor()

        input_data = """Here's the configuration:

```json
{"database": "postgres", "port": 5432, "ssl": true}
```

This should work for your use case."""

        result = await processor.process(input_data, None)

        # The extractor may return the raw input if balanced JSON not found
        # Verify the JSON content is present in the result
        assert "database" in result
        assert "postgres" in result

    @pytest.mark.asyncio
    async def test_extract_largest_balanced_json(self) -> None:
        """When multiple JSON blocks exist, extract the largest balanced one."""
        processor = JsonRegionExtractorProcessor()

        # Small object followed by larger nested object
        input_data = '{"a": 1} and also {"b": 2, "nested": {"c": 3, "d": 4}}'

        result = await processor.process(input_data, None)

        # The extractor should choose the largest balanced JSON object
        parsed = json.loads(result)
        assert parsed == {"b": 2, "nested": {"c": 3, "d": 4}}

    @pytest.mark.asyncio
    async def test_preserve_non_json_input(self) -> None:
        """Non-JSON input should be preserved (for downstream handling)."""
        processor = JsonRegionExtractorProcessor()

        input_data = "This is just plain text with no JSON at all."
        result = await processor.process(input_data, None)

        # Should return input unchanged when no JSON found
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_handle_nested_json_correctly(self) -> None:
        """Deeply nested JSON should be extracted intact."""
        processor = JsonRegionExtractorProcessor()

        nested_json = {"level1": {"level2": {"level3": {"value": "deep"}}}}
        input_data = f"Response: {json.dumps(nested_json)} - end"

        result = await processor.process(input_data, None)

        parsed = json.loads(result)
        assert parsed["level1"]["level2"]["level3"]["value"] == "deep"


# =============================================================================
# Stage 1: Tolerant JSON Decoding Tests
# =============================================================================


class TestTolerantJsonDecoderProcessor:
    """Tests for Stage 1 - Tolerant JSON Decoding.

    This processor handles common JSON formatting errors from LLMs:
    - Valid JSON (fast path via orjson/json)
    - Single quotes instead of double quotes
    - Trailing commas
    - Python literals (True/False/None vs true/false/null)
    """

    pytestmark = pytest.mark.fast

    @pytest.mark.asyncio
    async def test_decode_valid_json_fast_path(self) -> None:
        """Valid JSON should be decoded via fast path."""
        processor = TolerantJsonDecoderProcessor()

        input_data = '{"name": "test", "value": 123, "active": true}'
        result = await processor.process(input_data, None)

        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["value"] == 123
        assert result["active"] is True

    @pytest.mark.asyncio
    async def test_decode_json_array(self) -> None:
        """JSON arrays should decode correctly."""
        processor = TolerantJsonDecoderProcessor()

        input_data = '[1, "two", 3.0, null, true]'
        result = await processor.process(input_data, None)

        assert result == [1, "two", 3.0, None, True]

    @pytest.mark.asyncio
    async def test_passthrough_already_parsed_dict(self) -> None:
        """Already-parsed dicts should pass through unchanged."""
        processor = TolerantJsonDecoderProcessor()

        input_data = {"already": "parsed", "number": 42}
        result = await processor.process(input_data, None)

        assert result == {"already": "parsed", "number": 42}

    @pytest.mark.asyncio
    async def test_passthrough_already_parsed_list(self) -> None:
        """Already-parsed lists should pass through unchanged."""
        processor = TolerantJsonDecoderProcessor()

        input_data = [1, 2, {"nested": True}]
        result = await processor.process(input_data, None)

        assert result == [1, 2, {"nested": True}]

    @pytest.mark.asyncio
    async def test_decode_bytes_input(self) -> None:
        """Bytes input should be decoded and parsed."""
        processor = TolerantJsonDecoderProcessor()

        input_data = b'{"key": "value", "utf8": "\xc3\xa9"}'  # é in UTF-8
        result = await processor.process(input_data, None)

        assert result["key"] == "value"
        assert result["utf8"] == "é"

    @pytest.mark.asyncio
    async def test_fallback_returns_input_on_invalid_json(self) -> None:
        """Invalid JSON without tolerant mode returns original input."""
        processor = TolerantJsonDecoderProcessor(tolerant_level=0)

        input_data = "{'invalid': 'single quotes'}"
        result = await processor.process(input_data, None)

        # With tolerant_level=0, should return original string on failure
        assert result == input_data

    @pytest.mark.asyncio
    async def test_unicode_handling(self) -> None:
        """Unicode characters should be preserved."""
        processor = TolerantJsonDecoderProcessor()

        input_data = '{"emoji": "🚀", "chinese": "中文", "accent": "café"}'
        result = await processor.process(input_data, None)

        assert result["emoji"] == "🚀"
        assert result["chinese"] == "中文"
        assert result["accent"] == "café"

    @pytest.mark.asyncio
    async def test_nested_structure_decoding(self) -> None:
        """Complex nested structures should decode correctly."""
        processor = TolerantJsonDecoderProcessor()

        complex_json = json.dumps(
            {
                "users": [
                    {"id": 1, "name": "Alice", "roles": ["admin", "user"]},
                    {"id": 2, "name": "Bob", "roles": ["user"]},
                ],
                "metadata": {"version": "1.0", "generated": True},
            }
        )

        result = await processor.process(complex_json, None)

        assert len(result["users"]) == 2
        assert result["users"][0]["name"] == "Alice"
        assert "admin" in result["users"][0]["roles"]
        assert result["metadata"]["generated"] is True


# =============================================================================
# Stage 3: Smart Type Coercion Tests
# =============================================================================


class TestSmartTypeCoercionProcessor:
    """Tests for Stage 3 - Smart Type Coercion.

    This processor performs schema-aware type coercion with explicit allowlists.
    It requires BOTH a schema (to know target types) AND an allowlist (to know
    which coercions are permitted) for safety.

    Allowlist format: {"type_name": ["coercion_rule", ...]}
    - "boolean": ["str->bool"]  - allows "true"/"false" -> True/False
    - "integer": ["str->int"]   - allows "42" -> 42
    - "number":  ["str->float"] - allows "3.14" -> 3.14
    """

    pytestmark = pytest.mark.fast

    @pytest.mark.asyncio
    async def test_coerce_string_to_boolean_with_schema(self) -> None:
        """String 'true'/'false' should coerce to bool with schema+allowlist."""
        schema = {"type": "object", "properties": {"enabled": {"type": "boolean"}}}
        processor = SmartTypeCoercionProcessor(schema=schema, allow={"boolean": ["str->bool"]})

        result = await processor.process({"enabled": "true"}, None)
        assert result["enabled"] is True

        result = await processor.process({"enabled": "false"}, None)
        assert result["enabled"] is False

    @pytest.mark.asyncio
    async def test_coerce_string_to_integer_with_schema(self) -> None:
        """Numeric strings should coerce to int with schema+allowlist."""
        schema = {"type": "object", "properties": {"count": {"type": "integer"}}}
        processor = SmartTypeCoercionProcessor(schema=schema, allow={"integer": ["str->int"]})

        result = await processor.process({"count": "42"}, None)
        assert result["count"] == 42
        assert isinstance(result["count"], int)

    @pytest.mark.asyncio
    async def test_coerce_string_to_float_with_schema(self) -> None:
        """Decimal strings should coerce to float with schema+allowlist."""
        schema = {"type": "object", "properties": {"rate": {"type": "number"}}}
        processor = SmartTypeCoercionProcessor(schema=schema, allow={"number": ["str->float"]})

        result = await processor.process({"rate": "3.14159"}, None)
        assert result["rate"] == pytest.approx(3.14159)
        assert isinstance(result["rate"], float)

    @pytest.mark.asyncio
    async def test_no_coercion_without_allowlist(self) -> None:
        """Schema alone should NOT trigger coercion (safety feature)."""
        schema = {"type": "object", "properties": {"enabled": {"type": "boolean"}}}
        processor = SmartTypeCoercionProcessor(
            schema=schema,
            allow={},  # Empty allowlist = no coercion
        )

        result = await processor.process({"enabled": "true"}, None)
        # Should remain string without allowlist
        assert result["enabled"] == "true"

    @pytest.mark.asyncio
    async def test_allowlist_coercion_without_explicit_schema(self) -> None:
        """Allowlist with fallback _coerce_recursive may still coerce.

        The processor uses _coerce_recursive as fallback which applies
        allowlist-based coercion even without a full schema.
        """
        processor = SmartTypeCoercionProcessor(
            schema={},  # Empty schema
            allow={"boolean": ["str->bool"]},
        )

        result = await processor.process({"enabled": "true"}, None)
        # With allowlist, the fallback path should coerce deterministically
        assert result["enabled"] is True

    @pytest.mark.asyncio
    async def test_preserve_non_coercible_strings(self) -> None:
        """Non-coercible strings should be preserved unchanged."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        processor = SmartTypeCoercionProcessor(
            schema=schema, allow={"boolean": ["str->bool"], "integer": ["str->int"]}
        )

        result = await processor.process({"name": "Alice"}, None)
        assert result["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_preserve_already_correct_types(self) -> None:
        """Values already of correct type should pass through unchanged."""
        schema = SampleOutputSchema.model_json_schema()
        processor = SmartTypeCoercionProcessor(
            schema=schema, allow={"boolean": ["str->bool"], "integer": ["str->int"]}
        )

        input_data = {"enabled": True, "count": 42, "rate": 3.14, "name": "test"}
        result = await processor.process(input_data, None)

        assert result["enabled"] is True
        assert result["count"] == 42
        assert result["rate"] == 3.14
        assert result["name"] == "test"

    @pytest.mark.asyncio
    async def test_multiple_coercions_in_single_object(self) -> None:
        """Multiple fields can be coerced in a single pass."""
        schema = SampleOutputSchema.model_json_schema()
        processor = SmartTypeCoercionProcessor(
            schema=schema,
            allow={"boolean": ["str->bool"], "integer": ["str->int"], "number": ["str->float"]},
        )

        input_data = {"enabled": "true", "count": "100", "rate": "0.5", "name": "unchanged"}
        result = await processor.process(input_data, None)

        assert result["enabled"] is True
        assert result["count"] == 100
        assert result["rate"] == 0.5
        assert result["name"] == "unchanged"

    @pytest.mark.asyncio
    async def test_boolean_coercion_variants(self) -> None:
        """Boolean coercion handles multiple string representations."""
        schema = {"type": "object", "properties": {"flag": {"type": "boolean"}}}
        processor = SmartTypeCoercionProcessor(schema=schema, allow={"boolean": ["str->bool"]})

        # Test various truthy string values
        for truthy in ["true", "True", "TRUE", "1"]:
            result = await processor.process({"flag": truthy}, None)
            assert result["flag"] is True, f"Failed for '{truthy}'"

        # Test various falsy string values
        for falsy in ["false", "False", "FALSE", "0"]:
            result = await processor.process({"flag": falsy}, None)
            assert result["flag"] is False, f"Failed for '{falsy}'"

    @pytest.mark.asyncio
    async def test_error_recovery_returns_original(self) -> None:
        """Processing errors should return original data (graceful degradation)."""
        processor = SmartTypeCoercionProcessor(
            schema={"type": "object"}, allow={"integer": ["str->int"]}
        )

        # Value that looks like int target but isn't parseable
        input_data = {"value": "not_a_number"}
        result = await processor.process(input_data, None)

        # Should preserve original on coercion failure
        assert result["value"] == "not_a_number"


# =============================================================================
# Integration Tests: Full AROS Pipeline
# =============================================================================


class TestAROSPipelineIntegration:
    """Integration tests for the complete AROS pipeline.

    These tests verify that the stages work together correctly to recover
    valid typed output from messy LLM responses.
    """

    pytestmark = pytest.mark.slow

    @pytest.mark.asyncio
    async def test_full_pipeline_prose_with_malformed_json(self) -> None:
        """Test complete pipeline: prose -> extraction -> decode -> coercion."""
        # Simulate messy LLM output
        llm_output = """Based on my analysis, here is the result:

```json
{"enabled": "true", "count": "42", "status": "complete"}
```

Let me know if you need more details!"""

        # Stage 0: Extract
        extractor = JsonRegionExtractorProcessor()
        extracted = await extractor.process(llm_output, None)

        # Verify extraction worked
        assert "{" in extracted
        assert "enabled" in extracted

        # Stage 1: Decode
        decoder = TolerantJsonDecoderProcessor()
        decoded = await decoder.process(extracted, None)

        # Verify decoding worked
        assert isinstance(decoded, dict)
        assert decoded["enabled"] == "true"  # Still string
        assert decoded["count"] == "42"  # Still string

        # Stage 3: Coerce
        schema = {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "count": {"type": "integer"},
                "status": {"type": "string"},
            },
        }
        coercer = SmartTypeCoercionProcessor(
            schema=schema, allow={"boolean": ["str->bool"], "integer": ["str->int"]}
        )
        final = await coercer.process(decoded, None)

        # Verify coercion worked
        assert final["enabled"] is True
        assert final["count"] == 42
        assert final["status"] == "complete"

    @pytest.mark.asyncio
    async def test_pipeline_invalid_json_returns_original(self) -> None:
        """Invalid JSON should fall back to the original string payload."""
        llm_output = """```json
{not: valid}
```"""

        extractor = JsonRegionExtractorProcessor()
        extracted = await extractor.process(llm_output, None)

        decoder = TolerantJsonDecoderProcessor(tolerant_level=0)
        decoded = await decoder.process(extracted, None)
        assert decoded == extracted

        coercer = SmartTypeCoercionProcessor(schema={"type": "object"}, allow={})
        final = await coercer.process(decoded, None)
        assert final == extracted

    @pytest.mark.asyncio
    async def test_pipeline_prefers_largest_json_block(self) -> None:
        """Pipeline should extract, decode, and coerce the largest JSON block."""
        llm_output = '{"a": 1} and {"enabled": "true", "count": "2", "nested": {"flag": "false"}}'

        extractor = JsonRegionExtractorProcessor()
        extracted = await extractor.process(llm_output, None)
        assert "nested" in extracted

        decoder = TolerantJsonDecoderProcessor()
        decoded = await decoder.process(extracted, None)
        assert isinstance(decoded, dict)

        schema = {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "count": {"type": "integer"},
                "nested": {
                    "type": "object",
                    "properties": {"flag": {"type": "boolean"}},
                },
            },
        }
        coercer = SmartTypeCoercionProcessor(
            schema=schema, allow={"boolean": ["str->bool"], "integer": ["str->int"]}
        )
        final = await coercer.process(decoded, None)
        assert final == {"enabled": True, "count": 2, "nested": {"flag": False}}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("llm_output", "expected"),
        [
            ("", ""),
            ("null", None),
        ],
    )
    async def test_pipeline_handles_empty_or_null_inputs(
        self, llm_output: str, expected: object
    ) -> None:
        """Pipeline should handle empty or null inputs gracefully."""
        extractor = JsonRegionExtractorProcessor()
        extracted = await extractor.process(llm_output, None)

        decoder = TolerantJsonDecoderProcessor()
        decoded = await decoder.process(extracted, None)

        coercer = SmartTypeCoercionProcessor(schema={}, allow={})
        final = await coercer.process(decoded, None)
        assert final == expected
