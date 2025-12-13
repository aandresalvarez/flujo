"""Tests for built-in context manipulation helpers (Task 2.3).

This tests the context_set, context_merge, and context_get built-in skills.
"""

import pytest
from pydantic import Field
from flujo.domain.models import PipelineContext
from flujo.builtins import context_set, context_merge, context_get


class _Ctx(PipelineContext):
    counter: int = 0
    a: dict[str, object] = Field(default_factory=dict)
    settings: dict[str, object] = Field(default_factory=dict)
    new_settings: dict[str, object] = Field(default_factory=dict)
    user: dict[str, object] = Field(default_factory=dict)
    string_val: str | None = None
    int_val: int | None = None
    list_val: list[int] = Field(default_factory=list)
    dict_val: dict[str, object] = Field(default_factory=dict)


@pytest.mark.asyncio
async def test_context_set_simple_path():
    """context_set should set a typed context field."""
    context = _Ctx()

    result = await context_set(path="counter", value=42, context=context)

    assert result["path"] == "counter"
    assert result["value"] == 42
    assert result["success"] is True
    assert context.counter == 42


@pytest.mark.asyncio
async def test_context_set_nested_path():
    """context_set should set a nested dict path."""
    context = _Ctx()
    # First create the nested structure
    context.a = {"b": {}}

    result = await context_set(path="a.b.c", value="nested_value", context=context)

    assert result["success"] is True
    assert context.a["b"]["c"] == "nested_value"


@pytest.mark.asyncio
async def test_context_set_without_context():
    """context_set should handle missing context gracefully."""
    result = await context_set(path="field", value="test", context=None)

    assert result["path"] == "field"
    assert result["value"] == "test"
    assert result["success"] is False  # No context to update


@pytest.mark.asyncio
async def test_context_merge_dict():
    """context_merge should merge dictionary at path."""
    context = _Ctx()
    context.settings = {"theme": "light", "lang": "en"}

    result = await context_merge(
        path="settings",
        value={"theme": "dark", "notifications": True},
        context=context,
    )

    assert result["success"] is True
    assert "theme" in result["merged_keys"]
    assert "notifications" in result["merged_keys"]
    assert context.settings["theme"] == "dark"
    assert context.settings["notifications"] is True
    assert context.settings["lang"] == "en"  # Preserved


@pytest.mark.asyncio
async def test_context_merge_creates_path_if_missing():
    """context_merge should create path if it doesn't exist."""
    context = _Ctx()

    result = await context_merge(
        path="new_settings",
        value={"key1": "value1", "key2": "value2"},
        context=context,
    )

    # Should create the path and merge
    assert result["success"] is True
    assert context.new_settings["key1"] == "value1"


@pytest.mark.asyncio
async def test_context_get_with_default():
    """context_get should return value or default."""
    context = _Ctx()
    context.counter = 10

    # Get existing value
    result = await context_get(path="counter", default=0, context=context)
    assert result == 10

    # Get non-existent value (should return default)
    result = await context_get(path="missing", default=99, context=context)
    assert result == 99


@pytest.mark.asyncio
async def test_context_get_nested():
    """context_get should retrieve nested values."""
    context = _Ctx()
    context.user = {"name": "Alice", "settings": {"theme": "dark"}}

    result = await context_get(path="user.name", context=context)
    assert result == "Alice"

    result = await context_get(path="user.settings.theme", context=context)
    assert result == "dark"


@pytest.mark.asyncio
async def test_context_get_without_context():
    """context_get should return default when context is None."""
    result = await context_get(path="field", default="fallback", context=None)
    assert result == "fallback"


@pytest.mark.asyncio
async def test_context_helpers_type_safety():
    """Context helpers should work with various types."""
    context = _Ctx()

    # Set different types
    await context_set(path="string_val", value="text", context=context)
    await context_set(path="int_val", value=123, context=context)
    await context_set(path="list_val", value=[1, 2, 3], context=context)
    await context_set(path="dict_val", value={"key": "value"}, context=context)

    # Retrieve and verify types
    assert isinstance(await context_get(path="string_val", context=context), str)
    assert isinstance(await context_get(path="int_val", context=context), int)
    assert isinstance(await context_get(path="list_val", context=context), list)
    assert isinstance(await context_get(path="dict_val", context=context), dict)


@pytest.mark.asyncio
async def test_context_set_updates_existing():
    """context_set should update existing values."""
    context = _Ctx()
    context.counter = 0

    await context_set(path="counter", value=1, context=context)
    assert context.counter == 1

    await context_set(path="counter", value=2, context=context)
    assert context.counter == 2


@pytest.mark.asyncio
async def test_context_merge_empty_dict():
    """context_merge should handle empty dict gracefully."""
    context = _Ctx()
    context.settings = {"existing": "value"}

    result = await context_merge(path="settings", value={}, context=context)

    assert result["success"] is False  # No keys merged
    assert result["merged_keys"] == []
    assert context.settings["existing"] == "value"  # Preserved
