"""Unit tests for async bridge utilities."""

from __future__ import annotations

import asyncio
import pytest

from flujo.utils.async_bridge import run_sync


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

        # Call run_sync from inside an async function (existing loop scenario)
        result = run_sync(inner_async())
        assert result == "from nested"

    @pytest.mark.asyncio
    async def test_run_sync_double_invoke(self) -> None:
        """Test multiple consecutive calls from async context."""

        async def get_value(n: int) -> int:
            await asyncio.sleep(0.001)
            return n * 2

        # Multiple invocations should all work correctly
        r1 = run_sync(get_value(1))
        r2 = run_sync(get_value(2))
        r3 = run_sync(get_value(3))

        assert r1 == 2
        assert r2 == 4
        assert r3 == 6

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

        with pytest.raises(RuntimeError, match="nested failure"):
            run_sync(failing_func())

    def test_run_sync_with_coroutine_return_none(self) -> None:
        """Test coroutine that returns None explicitly."""

        async def void_func() -> None:
            await asyncio.sleep(0.001)
            return None

        result = run_sync(void_func())
        assert result is None
