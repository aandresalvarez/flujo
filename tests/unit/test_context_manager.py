"""Unit tests for ContextManager functionality."""

from pydantic import BaseModel
from flujo.application.core.context_manager import ContextManager


class MockContext(BaseModel):
    """Mock context class for testing."""

    value: int = 0
    nested: dict = {}
    data_store: dict = {}


class MockContextManager:
    """Test suite for ContextManager."""

    def test_isolate_none_context(self):
        """Test that isolating None returns None."""
        result = ContextManager.isolate(None)
        assert result is None

    def test_isolate_basic_context(self):
        """Test basic context isolation."""
        original = MockContext(value=42, nested={"key": "value"})
        isolated = ContextManager.isolate(original)

        assert isolated is not original
        assert isolated.value == 42
        assert isolated.nested == {"key": "value"}

        # Modifying isolated should not affect original
        isolated.value = 100
        isolated.nested["key"] = "modified"
        assert original.value == 42
        assert original.nested["key"] == "value"

    def test_isolate_with_include_keys(self):
        """Test context isolation with specific keys."""
        original = MockContext(value=42, nested={"key": "value"}, data_store={"data": "test"})
        isolated = ContextManager.isolate(original, include_keys=["value", "data_store"])

        assert isolated.value == 42
        assert isolated.data_store == {"data": "test"}
        # nested should be empty/default since not included
        assert isolated.nested == {}

    def test_merge_none_contexts(self):
        """Test merging when both contexts are None."""
        result = ContextManager.merge(None, None)
        assert result is None

    def test_merge_with_none_source(self):
        """Test merging when source context is None."""
        target = MockContext(value=10, nested={"existing": "data"})
        result = ContextManager.merge(target, None)
        assert result is target

    def test_merge_with_none_target(self):
        """Test merging when target context is None."""
        source = MockContext(value=20, nested={"new": "data"})
        result = ContextManager.merge(None, source)
        assert result is source

    def test_merge_basic_contexts(self):
        """Test basic context merging."""
        target = MockContext(value=10, nested={"existing": "data"}, data_store={"old": "value"})
        source = MockContext(value=20, nested={"new": "data"}, data_store={"new": "value"})

        result = ContextManager.merge(target, source)

        # Should return the target context (modified in-place)
        assert result is target

        # Should merge values
        assert result.value == 20  # source overwrites target
        assert result.nested == {"existing": "data", "new": "data"}  # dicts merged
        assert result.data_store == {"old": "value", "new": "value"}  # dicts merged

    def test_merge_preserves_target_structure(self):
        """Test that merging preserves the target context structure."""
        target = MockContext(value=10)
        source = MockContext(value=20, nested={"key": "value"}, data_store={"data": "test"})

        result = ContextManager.merge(target, source)

        # Should have all fields from target structure
        assert hasattr(result, "value")
        assert hasattr(result, "nested")
        assert hasattr(result, "data_store")

        # Should merge values correctly
        assert result.value == 20
        assert result.nested == {"key": "value"}
        assert result.data_store == {"data": "test"}

    def test_merge_deep_nested_structures(self):
        """Test merging with deeply nested structures."""
        target = MockContext(
            value=1,
            nested={"level1": {"level2": {"target": "data"}}},
            data_store={"deep": {"nested": {"target": "value"}}},
        )
        source = MockContext(
            value=2,
            nested={"level1": {"level2": {"source": "data"}}},
            data_store={"deep": {"nested": {"source": "value"}}},
        )

        result = ContextManager.merge(target, source)

        # Should merge nested structures
        assert result.nested["level1"]["level2"]["target"] == "data"
        assert result.nested["level1"]["level2"]["source"] == "data"
        assert result.data_store["deep"]["nested"]["target"] == "value"
        assert result.data_store["deep"]["nested"]["source"] == "value"

    def test_merge_handles_missing_attributes(self):
        """Test that merging handles missing attributes gracefully."""
        target = MockContext(value=10)
        source = MockContext(value=20, nested={"key": "value"})

        result = ContextManager.merge(target, source)

        # Should merge values correctly
        assert result.value == 20
        assert result.nested == {"key": "value"}

    def test_isolate_preserves_metadata(self):
        """Test that isolation preserves context metadata."""
        original = MockContext(value=42)
        original.__dict__["_metadata"] = {"test": "data"}

        isolated = ContextManager.isolate(original)

        # Should preserve metadata
        assert hasattr(isolated, "_metadata")
        assert isolated._metadata == {"test": "data"}

    def test_merge_with_complex_types(self):
        """Test merging with complex data types."""
        target = MockContext(
            value=1,
            nested={"list": [1, 2, 3], "dict": {"a": 1, "b": 2}},
            data_store={"set": {1, 2, 3}},
        )
        source = MockContext(
            value=2, nested={"list": [4, 5], "dict": {"c": 3}}, data_store={"set": {4, 5}}
        )

        result = ContextManager.merge(target, source)

        # Should merge complex types appropriately
        assert result.nested["list"] == [1, 2, 3, 4, 5]  # lists concatenated
        assert result.nested["dict"] == {"a": 1, "b": 2, "c": 3}  # dicts merged
        # Note: sets are not handled specially by the merge function, so they get overwritten
        assert result.data_store["set"] == {4, 5}  # sets overwritten

    def test_context_isolation_integration(self):
        """Integration test for context isolation workflow."""
        # Create initial context
        original = MockContext(value=0, data_store={"counter": 0})

        # Isolate for parallel execution
        branch1_context = ContextManager.isolate(original)
        branch2_context = ContextManager.isolate(original)

        # Modify in parallel branches
        branch1_context.value = 10
        branch1_context.data_store["counter"] = 1
        branch2_context.value = 20
        branch2_context.data_store["counter"] = 2

        # Original should remain unchanged
        assert original.value == 0
        assert original.data_store["counter"] == 0

        # Merge results back
        merged = ContextManager.merge(original, branch1_context)
        final = ContextManager.merge(merged, branch2_context)

        # Should have combined results
        assert final.value == 20  # last write wins
        assert final.data_store["counter"] == 2  # last write wins
