from flujo.type_definitions.common import JSONObject

"""Error handling utilities for robustness tests.

This module provides standardized error handling patterns and recovery
mechanisms for robustness testing scenarios.
"""
# ruff: noqa

import asyncio
import time
import traceback
from contextlib import contextmanager
from typing import Any, Callable, Optional, Dict, List, Type, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ErrorResult:
    """Result of an error handling test."""

    success: bool
    error_type: Optional[Type[Exception]] = None
    error_message: Optional[str] = None
    recovery_time: Optional[float] = None
    cleanup_successful: bool = False
    metadata: Optional[JSONObject] = None


class RobustnessErrorHandler:
    """Handles errors and recovery scenarios in robustness tests."""

    def __init__(self):
        self.error_history: List[ErrorResult] = []
        self.recovery_attempts = 0

    @contextmanager
    def expect_error(
        self,
        expected_errors: Union[Type[Exception], List[Type[Exception]]],
        recovery_action: Optional[Callable] = None,
        timeout: float = 30.0,
    ):
        """Context manager that expects and handles specific errors.

        Args:
            expected_errors: Exception type(s) that are expected
            recovery_action: Optional function to run for recovery
            timeout: Maximum time to wait for operation

        Yields:
            ErrorResult: Information about the error and recovery
        """
        if isinstance(expected_errors, type):
            expected_errors = [expected_errors]

        start_time = time.time()
        error_result = ErrorResult(success=False)

        try:
            yield error_result
            # If we get here without exception, operation succeeded
            error_result.success = True
            error_result.recovery_time = time.time() - start_time

        except Exception as e:
            error_time = time.time()
            error_result.error_type = type(e)
            error_result.error_message = str(e)
            error_result.recovery_time = error_time - start_time

            # Check if this is an expected error
            if any(isinstance(e, expected_type) for expected_type in expected_errors):
                logger.info(f"Expected error caught: {type(e).__name__}: {e}")

                # Try recovery action if provided
                if recovery_action:
                    try:
                        recovery_action()
                        error_result.cleanup_successful = True
                        logger.info("Recovery action completed successfully")
                    except Exception as recovery_error:
                        logger.warning(f"Recovery action failed: {recovery_error}")
                        error_result.metadata = {"recovery_error": str(recovery_error)}

                error_result.success = True  # Expected error was handled
            else:
                logger.error(f"Unexpected error: {type(e).__name__}: {e}")
                error_result.success = False

        finally:
            self.error_history.append(error_result)

    async def async_expect_error(
        self,
        expected_errors: Union[Type[Exception], List[Type[Exception]]],
        operation: Callable,
        recovery_action: Optional[Callable] = None,
        timeout: float = 30.0,
    ) -> ErrorResult:
        """Async version of expect_error for async operations.

        Args:
            expected_errors: Exception type(s) that are expected
            operation: Async callable to execute
            recovery_action: Optional function to run for recovery
            timeout: Maximum time to wait for operation

        Returns:
            ErrorResult: Information about the error and recovery
        """
        if isinstance(expected_errors, type):
            expected_errors = [expected_errors]

        start_time = time.time()
        error_result = ErrorResult(success=False)

        try:
            # Execute the async operation with timeout
            await asyncio.wait_for(operation(), timeout=timeout)
            error_result.success = True
            error_result.recovery_time = time.time() - start_time

        except asyncio.TimeoutError:
            error_result.error_type = asyncio.TimeoutError
            error_result.error_message = f"Operation timed out after {timeout}s"
            error_result.recovery_time = time.time() - start_time

            if asyncio.TimeoutError in expected_errors:
                error_result.success = True
                logger.info("Expected timeout occurred")
            else:
                logger.error("Unexpected timeout")

        except Exception as e:
            error_time = time.time()
            error_result.error_type = type(e)
            error_result.error_message = str(e)
            error_result.recovery_time = error_time - start_time

            # Check if this is an expected error
            if any(isinstance(e, expected_type) for expected_type in expected_errors):
                logger.info(f"Expected error caught: {type(e).__name__}: {e}")

                # Try recovery action if provided
                if recovery_action:
                    try:
                        if asyncio.iscoroutinefunction(recovery_action):
                            await recovery_action()
                        else:
                            recovery_action()
                        error_result.cleanup_successful = True
                        logger.info("Recovery action completed successfully")
                    except Exception as recovery_error:
                        logger.warning(f"Recovery action failed: {recovery_error}")
                        error_result.metadata = {"recovery_error": str(recovery_error)}

                error_result.success = True  # Expected error was handled
            else:
                logger.error(f"Unexpected error: {type(e).__name__}: {e}")
                error_result.success = False

        self.error_history.append(error_result)
        return error_result

    def retry_operation(
        self,
        operation: Callable,
        max_attempts: int = 3,
        backoff_factor: float = 1.5,
        exceptions_to_retry: Optional[List[Type[Exception]]] = None,
    ) -> Any:
        """Retry an operation with exponential backoff.

        Args:
            operation: Function to retry
            max_attempts: Maximum number of attempts
            backoff_factor: Exponential backoff multiplier
            exceptions_to_retry: Exception types to retry on

        Returns:
            Result of the operation

        Raises:
            Last exception if all retries fail
        """
        if exceptions_to_retry is None:
            exceptions_to_retry = [Exception]

        last_exception = None
        wait_time = 0.1  # Start with 100ms

        for attempt in range(max_attempts):
            try:
                self.recovery_attempts += 1
                return operation()
            except Exception as e:
                last_exception = e
                if not any(isinstance(e, exc_type) for exc_type in exceptions_to_retry):
                    raise  # Don't retry for unexpected exceptions

                if attempt < max_attempts - 1:
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.2f}s..."
                    )
                    time.sleep(wait_time)
                    wait_time *= backoff_factor
                else:
                    logger.error(f"All {max_attempts} attempts failed. Last error: {e}")

        raise last_exception

    async def async_retry_operation(
        self,
        operation: Callable,
        max_attempts: int = 3,
        backoff_factor: float = 1.5,
        exceptions_to_retry: Optional[List[Type[Exception]]] = None,
    ) -> Any:
        """Async version of retry_operation."""
        if exceptions_to_retry is None:
            exceptions_to_retry = [Exception]

        last_exception = None
        wait_time = 0.1  # Start with 100ms

        for attempt in range(max_attempts):
            try:
                self.recovery_attempts += 1
                return await operation()
            except Exception as e:
                last_exception = e
                if not any(isinstance(e, exc_type) for exc_type in exceptions_to_retry):
                    raise  # Don't retry for unexpected exceptions

                if attempt < max_attempts - 1:
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.2f}s..."
                    )
                    await asyncio.sleep(wait_time)
                    wait_time *= backoff_factor
                else:
                    logger.error(f"All {max_attempts} attempts failed. Last error: {e}")

        raise last_exception

    def get_error_summary(self) -> JSONObject:
        """Get summary of error handling results."""
        total_errors = len(self.error_history)
        successful_handling = sum(1 for r in self.error_history if r.success)
        cleanup_success = sum(1 for r in self.error_history if r.cleanup_successful)

        error_types = {}
        for result in self.error_history:
            if result.error_type:
                type_name = result.error_type.__name__
                error_types[type_name] = error_types.get(type_name, 0) + 1

        avg_recovery_time = None
        if self.error_history:
            recovery_times = [r.recovery_time for r in self.error_history if r.recovery_time]
            if recovery_times:
                avg_recovery_time = sum(recovery_times) / len(recovery_times)

        return {
            "total_errors_handled": total_errors,
            "successful_handling_rate": successful_handling / total_errors
            if total_errors > 0
            else 1.0,
            "cleanup_success_rate": cleanup_success / total_errors if total_errors > 0 else 1.0,
            "error_types": error_types,
            "average_recovery_time": avg_recovery_time,
            "total_recovery_attempts": self.recovery_attempts,
        }


# Global error handler instance
_error_handler = None


def get_error_handler() -> RobustnessErrorHandler:
    """Get the global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = RobustnessErrorHandler()
    return _error_handler


def with_error_recovery(
    expected_errors: Union[Type[Exception], List[Type[Exception]]],
    recovery_action: Optional[Callable] = None,
):
    """Decorator for functions that should handle specific errors gracefully."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            handler = get_error_handler()
            with handler.expect_error(expected_errors, recovery_action):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def async_with_error_recovery(
    expected_errors: Union[Type[Exception], List[Type[Exception]]],
    recovery_action: Optional[Callable] = None,
):
    """Decorator for async functions that should handle specific errors gracefully."""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            handler = get_error_handler()
            error_result = await handler.async_expect_error(expected_errors, func, recovery_action)
            return error_result

        return wrapper

    return decorator
