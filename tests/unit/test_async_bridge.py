"""Unit tests for async bridge utilities."""

from __future__ import annotations

import asyncio
import pytest

from flujo.utils.async_bridge import run_sync

pytestmark = pytest.mark.fast


class TestRunSync:
    """Tests for the run_sync utility."""

    def test_run_sync_basic(self) -> None:
        """Test basic async to sync conversion."""

        async def async_func() -> str:
            return "hello"

        result = run_sync(async_func())
        assert result == "hello"

    def test_run_sync_with_await(self) -> None:
        """Test async function that uses await internally."""

        async def async_func() -> int:
            await asyncio.sleep(0.01)
            return 42

        result = run_sync(async_func())
        assert result == 42

    def test_run_sync_preserves_exception(self) -> None:
        """Test that exceptions are properly propagated."""

        async def async_func() -> None:
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            run_sync(async_func())

    def test_run_sync_with_complex_return(self) -> None:
        """Test returning complex objects."""

        async def async_func() -> dict[str, list[int]]:
            return {"numbers": [1, 2, 3]}

        result = run_sync(async_func())
        assert result == {"numbers": [1, 2, 3]}

    @pytest.mark.asyncio
    async def test_run_sync_from_async_context(self) -> None:
        """Test calling run_sync when already inside an event loop.

        This is the critical case - we need to handle nested loops safely.
        """

        async def inner_async() -> str:
            await asyncio.sleep(0.01)
            return "from nested"

        with pytest.raises(TypeError, match=r"cannot be called from a running event loop thread"):
            run_sync(inner_async())

    @pytest.mark.asyncio
    async def test_run_sync_custom_running_loop_error(self) -> None:
        """Custom error messages should be preserved for callers with better guidance."""

        async def inner_async() -> str:
            return "never returned"

        with pytest.raises(TypeError, match="use async APIs here"):
            run_sync(inner_async(), running_loop_error="use async APIs here")

    @pytest.mark.asyncio
    async def test_run_sync_double_invoke(self) -> None:
        """Test multiple consecutive calls from async context."""

        async def get_value(n: int) -> int:
            await asyncio.sleep(0.001)
            return n * 2

        with pytest.raises(TypeError, match=r"cannot be called from a running event loop thread"):
            run_sync(get_value(1))
        with pytest.raises(TypeError, match=r"cannot be called from a running event loop thread"):
            run_sync(get_value(2))
        with pytest.raises(TypeError, match=r"cannot be called from a running event loop thread"):
            run_sync(get_value(3))

    def test_run_sync_no_running_loop(self) -> None:
        """Test run_sync when there's no running loop (uses asyncio.run directly)."""

        async def simple() -> str:
            return "direct"

        # When no loop is running, should use asyncio.run directly
        result = run_sync(simple())
        assert result == "direct"

    @pytest.mark.asyncio
    async def test_run_sync_exception_in_nested_context(self) -> None:
        """Test exception handling when called from async context."""

        async def failing_func() -> None:
            await asyncio.sleep(0.001)
            raise RuntimeError("nested failure")

        with pytest.raises(TypeError, match=r"cannot be called from a running event loop thread"):
            run_sync(failing_func())

    def test_run_sync_with_coroutine_return_none(self) -> None:
        """Test coroutine that returns None explicitly."""

        async def void_func() -> None:
            await asyncio.sleep(0.001)
            return None

        result = run_sync(void_func())
        assert result is None
