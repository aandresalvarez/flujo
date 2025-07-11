#!/usr/bin/env python3
"""Test script to verify the cache serialization fix for nested BaseModel instances."""

from pydantic import BaseModel
from typing import Any, Dict, List
import json

from flujo.steps.cache_step import _serialize_for_key, _generate_cache_key
from flujo.domain.models import PipelineContext
from flujo.domain.dsl import Step


class NestedModel(BaseModel):
    value: int
    description: str


class ComplexModel(BaseModel):
    name: str
    nested: NestedModel
    items: List[str]
    metadata: Dict[str, Any]


def test_nested_base_model_serialization():
    """Test that nested BaseModel instances are properly serialized recursively."""

    # Create a complex nested model
    nested = NestedModel(value=42, description="test")
    complex_obj = ComplexModel(
        name="test",
        nested=nested,
        items=["a", "b", "c"],
        metadata={"key": "value", "number": 123}
    )

    # Test the serialization
    serialized = _serialize_for_key(complex_obj)

    print("Original object:", complex_obj)
    print("Serialized result:", serialized)

    # Verify it's JSON serializable
    try:
        json_str = json.dumps(serialized, sort_keys=True)
        print("✅ Successfully serialized to JSON")
        print("JSON length:", len(json_str))
    except Exception as e:
        print(f"❌ Failed to serialize to JSON: {e}")
        return False

    # Test with PipelineContext (which has special handling)
    context = PipelineContext(
        initial_prompt="test prompt",
        run_id="test-run-123"
    )

    context_serialized = _serialize_for_key(context)
    print("\nPipelineContext serialized:", context_serialized)

    # Verify run_id is excluded
    if "run_id" not in context_serialized:
        print("✅ run_id correctly excluded from PipelineContext")
    else:
        print("❌ run_id not excluded from PipelineContext")
        return False

    return True


def test_cache_key_generation():
    """Test that cache key generation works with complex nested models."""

    # Create a mock step
    class MockStep(Step[str, str]):
        def run(self, data: str, context: Any = None, resources: Any = None) -> str:
            return f"processed: {data}"

    step = MockStep(name="test_step")

    # Create complex data with nested models
    nested = NestedModel(value=100, description="cache test")
    complex_data = ComplexModel(
        name="cache_test",
        nested=nested,
        items=["x", "y", "z"],
        metadata={"cache": True, "version": 1}
    )

    # Generate cache key
    cache_key = _generate_cache_key(step, complex_data)

    print(f"\nGenerated cache key: {cache_key}")

    if cache_key and cache_key.startswith("test_step:"):
        print("✅ Cache key generated successfully")
        return True
    else:
        print("❌ Cache key generation failed")
        return False


if __name__ == "__main__":
    print("Testing cache serialization fix...")

    success1 = test_nested_base_model_serialization()
    success2 = test_cache_key_generation()

    if success1 and success2:
        print("\n🎉 All tests passed! The cache serialization fix is working correctly.")
    else:
        print("\n❌ Some tests failed. The fix may not be complete.")
