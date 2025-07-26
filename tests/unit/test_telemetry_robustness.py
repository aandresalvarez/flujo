"""Tests for robust telemetry handling during cleanup."""

import asyncio
import logging
import pytest
from unittest.mock import Mock, patch

from flujo.infra.telemetry import _safe_log, _MockLogfire, _SafeLogfireWrapper


class TestTelemetryRobustness:
    """Test that telemetry handles I/O errors gracefully during cleanup."""

    def test_safe_log_handles_closed_file_error(self):
        """Test that _safe_log handles I/O errors gracefully."""
        # Create a mock logger that raises I/O errors
        mock_logger = Mock()
        mock_logger.log.side_effect = ValueError("I/O operation on closed file")

        # This should not raise an exception
        _safe_log(mock_logger, logging.INFO, "test message")

        # Verify the logger was called
        mock_logger.log.assert_called_once()

    def test_safe_log_handles_other_closed_errors(self):
        """Test that _safe_log handles various closed file error messages."""
        mock_logger = Mock()
        mock_logger.log.side_effect = ValueError("closed file")

        # This should not raise an exception
        _safe_log(mock_logger, logging.INFO, "test message")

        # Verify the logger was called
        mock_logger.log.assert_called_once()

    def test_safe_log_raises_unexpected_errors(self):
        """Test that _safe_log re-raises unexpected errors."""
        mock_logger = Mock()
        mock_logger.log.side_effect = ValueError("unexpected error")

        # This should raise the exception
        with pytest.raises(ValueError, match="unexpected error"):
            _safe_log(mock_logger, logging.INFO, "test message")

    def test_mock_logfire_handles_closed_file_error(self):
        """Test that _MockLogfire handles I/O errors gracefully."""
        # Create a mock logger that raises I/O errors
        mock_logger = Mock()
        mock_logfire = _MockLogfire(mock_logger)

        # Set up the logger to raise I/O errors
        mock_logger.log.side_effect = ValueError("I/O operation on closed file")

        # These calls should not raise exceptions
        mock_logfire.info("test info")
        mock_logfire.warn("test warn")
        mock_logfire.warning("test warning")
        mock_logfire.error("test error")
        mock_logfire.debug("test debug")

        # Verify all methods were called
        assert mock_logger.log.call_count == 5

    def test_safe_logfire_wrapper_handles_closed_file_error(self):
        """Test that _SafeLogfireWrapper handles I/O errors gracefully."""
        # Create a mock real logfire that raises I/O errors
        mock_real_logfire = Mock()
        mock_real_logfire.info.side_effect = ValueError("I/O operation on closed file")
        mock_real_logfire.warn.side_effect = ValueError("I/O operation on closed file")
        mock_real_logfire.warning.side_effect = ValueError("I/O operation on closed file")
        mock_real_logfire.error.side_effect = ValueError("I/O operation on closed file")
        mock_real_logfire.debug.side_effect = ValueError("I/O operation on closed file")

        wrapper = _SafeLogfireWrapper(mock_real_logfire)

        # These calls should not raise exceptions
        wrapper.info("test info")
        wrapper.warn("test warn")
        wrapper.warning("test warning")
        wrapper.error("test error")
        wrapper.debug("test debug")

        # Verify all methods were called
        assert mock_real_logfire.info.call_count == 1
        assert mock_real_logfire.warn.call_count == 1
        assert mock_real_logfire.warning.call_count == 1
        assert mock_real_logfire.error.call_count == 1
        assert mock_real_logfire.debug.call_count == 1

    def test_safe_logfire_wrapper_raises_unexpected_errors(self):
        """Test that _SafeLogfireWrapper re-raises unexpected errors."""
        mock_real_logfire = Mock()
        mock_real_logfire.info.side_effect = ValueError("unexpected error")

        wrapper = _SafeLogfireWrapper(mock_real_logfire)

        # This should raise the exception
        with pytest.raises(ValueError, match="unexpected error"):
            wrapper.info("test message")

    @pytest.mark.asyncio
    async def test_telemetry_during_cancellation(self):
        """Test that telemetry works correctly during asyncio cancellation."""
        from flujo.infra.telemetry import logfire

        # Create a task that will be cancelled
        async def long_running_task():
            try:
                await asyncio.sleep(10)  # This will be cancelled
            except asyncio.CancelledError:
                # Log during cancellation
                logfire.info("Task cancelled")
                raise

        # Start the task
        task = asyncio.create_task(long_running_task())

        # Let it run briefly
        await asyncio.sleep(0.01)

        # Cancel the task
        task.cancel()

        # Wait for it to complete
        try:
            await task
        except asyncio.CancelledError:
            pass

        # The test should complete without I/O errors

    def test_telemetry_initialization_robustness(self):
        """Test that telemetry initialization is robust."""
        from flujo.infra.telemetry import init_telemetry

        # Mock settings to disable telemetry
        with patch("flujo.infra.settings.settings") as mock_settings:
            mock_settings.telemetry_export_enabled = False

            # This should not raise any exceptions
            init_telemetry()

            # Verify it was called
            assert mock_settings.telemetry_export_enabled is False

    def test_logging_handler_cleanup(self):
        """Test that logging handlers are cleaned up properly."""
        # Create a logger with handlers
        logger = logging.getLogger("test_cleanup")
        handler = logging.StreamHandler()
        logger.addHandler(handler)

        # Close the handler
        handler.close()

        # Try to log - this should not cause I/O errors with our safe logging
        _safe_log(logger, logging.INFO, "test message")

        # Clean up
        logger.removeHandler(handler)
