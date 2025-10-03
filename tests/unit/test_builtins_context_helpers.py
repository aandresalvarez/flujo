"""Tests for built-in context manipulation helpers (Task 2.3).

This tests the context_set, context_merge, and context_get built-in skills.
"""

import pytest
from flujo.domain.models import PipelineContext
from flujo.builtins import context_set, context_merge, context_get


@pytest.mark.asyncio
async def test_context_set_simple_path():
    """context_set should set scratchpad.field."""
    context = PipelineContext()

    result = await context_set(path="scratchpad.counter", value=42, context=context)

    assert result["path"] == "scratchpad.counter"
    assert result["value"] == 42
    assert result["success"] is True
    assert context.scratchpad["counter"] == 42


@pytest.mark.asyncio
async def test_context_set_nested_path():
    """context_set should set scratchpad.a.b.c."""
    context = PipelineContext()
    # First create the nested structure
    context.scratchpad["a"] = {"b": {}}

    result = await context_set(path="scratchpad.a.b.c", value="nested_value", context=context)

    assert result["success"] is True
    assert context.scratchpad["a"]["b"]["c"] == "nested_value"


@pytest.mark.asyncio
async def test_context_set_without_context():
    """context_set should handle missing context gracefully."""
    result = await context_set(path="scratchpad.field", value="test", context=None)

    assert result["path"] == "scratchpad.field"
    assert result["value"] == "test"
    assert result["success"] is False  # No context to update


@pytest.mark.asyncio
async def test_context_merge_dict():
    """context_merge should merge dictionary at path."""
    context = PipelineContext()
    context.scratchpad["settings"] = {"theme": "light", "lang": "en"}

    result = await context_merge(
        path="scratchpad.settings",
        value={"theme": "dark", "notifications": True},
        context=context,
    )

    assert result["success"] is True
    assert "theme" in result["merged_keys"]
    assert "notifications" in result["merged_keys"]
    assert context.scratchpad["settings"]["theme"] == "dark"
    assert context.scratchpad["settings"]["notifications"] is True
    assert context.scratchpad["settings"]["lang"] == "en"  # Preserved


@pytest.mark.asyncio
async def test_context_merge_creates_path_if_missing():
    """context_merge should create path if it doesn't exist."""
    context = PipelineContext()

    result = await context_merge(
        path="scratchpad.new_settings",
        value={"key1": "value1", "key2": "value2"},
        context=context,
    )

    # Should create the path and merge
    assert result["success"] is True
    assert context.scratchpad["new_settings"]["key1"] == "value1"


@pytest.mark.asyncio
async def test_context_get_with_default():
    """context_get should return value or default."""
    context = PipelineContext()
    context.scratchpad["counter"] = 10

    # Get existing value
    result = await context_get(path="scratchpad.counter", default=0, context=context)
    assert result == 10

    # Get non-existent value (should return default)
    result = await context_get(path="scratchpad.missing", default=99, context=context)
    assert result == 99


@pytest.mark.asyncio
async def test_context_get_nested():
    """context_get should retrieve nested values."""
    context = PipelineContext()
    context.scratchpad["user"] = {"name": "Alice", "settings": {"theme": "dark"}}

    result = await context_get(path="scratchpad.user.name", context=context)
    assert result == "Alice"

    result = await context_get(path="scratchpad.user.settings.theme", context=context)
    assert result == "dark"


@pytest.mark.asyncio
async def test_context_get_without_context():
    """context_get should return default when context is None."""
    result = await context_get(path="scratchpad.field", default="fallback", context=None)
    assert result == "fallback"


@pytest.mark.asyncio
async def test_context_helpers_type_safety():
    """Context helpers should work with various types."""
    context = PipelineContext()

    # Set different types
    await context_set(path="scratchpad.string_val", value="text", context=context)
    await context_set(path="scratchpad.int_val", value=123, context=context)
    await context_set(path="scratchpad.list_val", value=[1, 2, 3], context=context)
    await context_set(path="scratchpad.dict_val", value={"key": "value"}, context=context)

    # Retrieve and verify types
    assert isinstance(await context_get(path="scratchpad.string_val", context=context), str)
    assert isinstance(await context_get(path="scratchpad.int_val", context=context), int)
    assert isinstance(await context_get(path="scratchpad.list_val", context=context), list)
    assert isinstance(await context_get(path="scratchpad.dict_val", context=context), dict)


@pytest.mark.asyncio
async def test_context_set_updates_existing():
    """context_set should update existing values."""
    context = PipelineContext()
    context.scratchpad["counter"] = 0

    await context_set(path="scratchpad.counter", value=1, context=context)
    assert context.scratchpad["counter"] == 1

    await context_set(path="scratchpad.counter", value=2, context=context)
    assert context.scratchpad["counter"] == 2


@pytest.mark.asyncio
async def test_context_merge_empty_dict():
    """context_merge should handle empty dict gracefully."""
    context = PipelineContext()
    context.scratchpad["settings"] = {"existing": "value"}

    result = await context_merge(path="scratchpad.settings", value={}, context=context)

    assert result["success"] is False  # No keys merged
    assert result["merged_keys"] == []
    assert context.scratchpad["settings"]["existing"] == "value"  # Preserved
