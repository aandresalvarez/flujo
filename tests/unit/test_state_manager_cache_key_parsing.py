"""Unit tests for StateManager cache key parsing with run_ids containing underscores."""

import pytest

from flujo.application.core.state_manager import StateManager


class TestStateManagerCacheKeyParsing:
    """Test cache key parsing with run_ids containing underscores."""

    @pytest.fixture
    def state_manager(self):
        """Create a StateManager instance for testing."""
        return StateManager()

    def test_create_cache_key_with_underscores_in_run_id(self, state_manager):
        """Test that cache keys are created correctly with run_ids containing underscores."""
        run_id = "user_123_pipeline_456"
        context_hash = "abc123def456"

        cache_key = state_manager._create_cache_key(run_id, context_hash)

        # Should use pipe separator instead of underscore
        expected_key = f"{run_id}|{context_hash}"
        assert cache_key == expected_key

    def test_parse_cache_key_with_underscores_in_run_id(self, state_manager):
        """Test that cache keys can be parsed correctly with run_ids containing underscores."""
        run_id = "production_workflow_2024_01_15"
        context_hash = "def789ghi012"

        cache_key = state_manager._create_cache_key(run_id, context_hash)
        parsed_run_id, parsed_context_hash = state_manager._parse_cache_key(cache_key)

        assert parsed_run_id == run_id
        assert parsed_context_hash == context_hash

    def test_cache_key_roundtrip_with_complex_run_id(self, state_manager):
        """Test cache key creation and parsing with complex run_ids."""
        test_cases = [
            "user_123_pipeline_456",
            "test_run_with_underscores",
            "production_workflow_2024_01_15",
            "simple_run",  # No underscores
            "run_with_many_underscores_in_the_middle",
            "prefix_123_suffix_456",
            "a_b_c_d_e_f_g",
            "run_id_with_123_numbers_456",
        ]

        for run_id in test_cases:
            context_hash = "test_hash_123"
            cache_key = state_manager._create_cache_key(run_id, context_hash)
            parsed_run_id, parsed_context_hash = state_manager._parse_cache_key(cache_key)

            assert parsed_run_id == run_id, f"Failed for run_id: {run_id}"
            assert parsed_context_hash == context_hash, f"Failed for run_id: {run_id}"

    def test_parse_cache_key_invalid_format(self, state_manager):
        """Test that invalid cache key formats raise ValueError."""
        invalid_keys = [
            "no_separator",
            "",  # Empty string
        ]

        for invalid_key in invalid_keys:
            print(f"Testing invalid key: '{invalid_key}'")
            try:
                result = state_manager._parse_cache_key(invalid_key)
                print(f"  Unexpected success: {result}")
            except ValueError as e:
                print(f"  Expected error: {e}")
                continue
            pytest.fail(f"Expected ValueError for '{invalid_key}' but got {result}")

    def test_parse_cache_key_edge_cases(self, state_manager):
        """Test edge cases that should work correctly."""
        edge_cases = [
            ("only|", ("only", "")),  # Empty context_hash is valid
            ("|only_hash", ("", "only_hash")),  # Empty run_id is valid
            ("run_id|context_hash", ("run_id", "context_hash")),
        ]

        for cache_key, expected in edge_cases:
            result = state_manager._parse_cache_key(cache_key)
            assert result == expected, (
                f"Failed for '{cache_key}': expected {expected}, got {result}"
            )

    def test_legacy_cache_key_handling(self, state_manager):
        """Test that legacy cache keys (old format) are handled gracefully."""
        # Legacy format used underscore separator
        legacy_key = "user_123_pipeline_456_abc123def456"

        with pytest.raises(ValueError, match="Invalid cache key format"):
            state_manager._parse_cache_key(legacy_key)

    def test_cache_key_with_pipe_in_run_id(self, state_manager):
        """Test that cache keys work correctly even if run_id contains pipe characters."""
        # This is an edge case - run_id with pipe character
        run_id = "user|123|pipeline|456"
        context_hash = "abc123def456"

        cache_key = state_manager._create_cache_key(run_id, context_hash)
        parsed_run_id, parsed_context_hash = state_manager._parse_cache_key(cache_key)

        assert parsed_run_id == run_id
        assert parsed_context_hash == context_hash

    def test_cache_key_with_multiple_pipes_in_run_id(self, state_manager):
        """Test that cache keys work correctly with multiple pipes in run_id."""
        # Test with multiple pipes in run_id
        run_id = "user|123|pipeline|456|extra"
        context_hash = "abc123def456"

        cache_key = state_manager._create_cache_key(run_id, context_hash)
        parsed_run_id, parsed_context_hash = state_manager._parse_cache_key(cache_key)

        assert parsed_run_id == run_id
        assert parsed_context_hash == context_hash

    def test_cache_key_with_special_characters(self, state_manager):
        """Test cache keys with various special characters in run_id."""
        test_cases = [
            "run-id-with-dashes",
            "run.id.with.dots",
            "run:id:with:colons",
            "run id with spaces",
            "run\tid\twith\ttabs",
            "run\nid\nwith\nnewlines",
            "run_id_with_unicode_ñáéíóú",
            "run_id_with_emoji_🚀🎉",
        ]

        for run_id in test_cases:
            context_hash = "test_hash_123"
            cache_key = state_manager._create_cache_key(run_id, context_hash)
            parsed_run_id, parsed_context_hash = state_manager._parse_cache_key(cache_key)

            assert parsed_run_id == run_id, f"Failed for run_id: {run_id}"
            assert parsed_context_hash == context_hash, f"Failed for run_id: {run_id}"

    @pytest.mark.slow  # Mark as slow due to performance measurement
    def test_cache_key_performance(self, state_manager):
        """Test that cache key operations are performant."""
        import time

        run_id = "performance_test_run_id_with_many_underscores_in_the_middle"
        context_hash = "performance_test_hash_123"

        # Test creation performance
        start_time = time.perf_counter()
        for _ in range(1000):
            cache_key = state_manager._create_cache_key(run_id, context_hash)
        creation_time = time.perf_counter() - start_time

        # Test parsing performance
        cache_key = state_manager._create_cache_key(run_id, context_hash)
        start_time = time.perf_counter()
        for _ in range(1000):
            parsed_run_id, parsed_context_hash = state_manager._parse_cache_key(cache_key)
        parsing_time = time.perf_counter() - start_time

        # Should be very fast (less than 1ms for 1000 operations)
        assert creation_time < 0.005, f"Cache key creation too slow: {creation_time:.6f}s"
        assert parsing_time < 0.005, f"Cache key parsing too slow: {parsing_time:.6f}s"
