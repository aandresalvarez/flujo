"""
Circuit breaker implementation for preventing cascade failures.

This module provides a circuit breaker pattern implementation with configurable
failure thresholds, timeouts, state management, and automatic recovery to prevent
cascade failures and improve system resilience.
"""

import asyncio
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from threading import RLock
from enum import Enum
import statistics

from .optimized_telemetry import get_global_telemetry, MetricType


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, calls are rejected
    HALF_OPEN = "half_open"  # Testing if service has recovered


class FailureType(Enum):
    """Types of failures tracked by circuit breaker."""
    EXCEPTION = "exception"
    TIMEOUT = "timeout"
    SLOW_RESPONSE = "slow_response"
    CUSTOM = "custom"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    
    # Failure thresholds
    failure_threshold: int = 5  # Number of failures to open circuit
    failure_rate_threshold: float = 0.5  # Failure rate (0.0-1.0) to open circuit
    slow_call_threshold_ms: float = 5000.0  # Threshold for slow calls
    
    # Time windows
    failure_window_seconds: int = 60  # Time window for failure counting
    recovery_timeout_seconds: int = 30  # Time to wait before trying half-open
    half_open_max_calls: int = 3  # Max calls allowed in half-open state
    
    # Minimum calls
    minimum_calls: int = 10  # Minimum calls before evaluating failure rate
    
    # Custom failure detection
    custom_failure_predicate: Optional[Callable[[Any], bool]] = None
    
    # Monitoring
    enable_metrics: bool = True
    enable_detailed_logging: bool = False


@dataclass
class CallResult:
    """Result of a circuit breaker protected call."""
    
    success: bool
    duration_ms: float
    failure_type: Optional[FailureType] = None
    error: Optional[Exception] = None
    timestamp: float = field(default_factory=time.time)
    
    @property
    def is_slow(self) -> bool:
        """Check if call was slow."""
        return self.failure_type == FailureType.SLOW_RESPONSE


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker."""
    
    # State tracking
    current_state: CircuitState = CircuitState.CLOSED
    state_transitions: int = 0
    last_state_change: float = field(default_factory=time.time)
    
    # Call statistics
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    
    # Failure breakdown
    exception_failures: int = 0
    timeout_failures: int = 0
    slow_call_failures: int = 0
    custom_failures: int = 0
    
    # Timing statistics
    total_duration_ms: float = 0.0
    min_duration_ms: float = float('inf')
    max_duration_ms: float = 0.0
    
    # Recent performance
    recent_calls: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_calls == 0:
            return 1.0
        return self.successful_calls / self.total_calls
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls
    
    @property
    def rejection_rate(self) -> float:
        """Calculate rejection rate."""
        total_attempts = self.total_calls + self.rejected_calls
        if total_attempts == 0:
            return 0.0
        return self.rejected_calls / total_attempts
    
    @property
    def average_duration_ms(self) -> float:
        """Calculate average call duration."""
        if self.successful_calls == 0:
            return 0.0
        return self.total_duration_ms / self.successful_calls
    
    def update_call(self, result: CallResult) -> None:
        """Update statistics with call result."""
        self.total_calls += 1
        self.recent_calls.append(result)
        
        if result.success:
            self.successful_calls += 1
            self.total_duration_ms += result.duration_ms
            self.min_duration_ms = min(self.min_duration_ms, result.duration_ms)
            self.max_duration_ms = max(self.max_duration_ms, result.duration_ms)
        else:
            self.failed_calls += 1
            
            # Update failure type counters
            if result.failure_type == FailureType.EXCEPTION:
                self.exception_failures += 1
            elif result.failure_type == FailureType.TIMEOUT:
                self.timeout_failures += 1
            elif result.failure_type == FailureType.SLOW_RESPONSE:
                self.slow_call_failures += 1
            elif result.failure_type == FailureType.CUSTOM:
                self.custom_failures += 1
    
    def update_rejection(self) -> None:
        """Update rejection statistics."""
        self.rejected_calls += 1
    
    def update_state_change(self, new_state: CircuitState) -> None:
        """Update state change statistics."""
        if new_state != self.current_state:
            self.current_state = new_state
            self.state_transitions += 1
            self.last_state_change = time.time()
    
    def get_recent_failure_rate(self, window_size: int = 50) -> float:
        """Get failure rate for recent calls."""
        if not self.recent_calls:
            return 0.0
        
        recent = list(self.recent_calls)[-window_size:]
        if not recent:
            return 0.0
        
        failures = sum(1 for call in recent if not call.success)
        return failures / len(recent)
    
    def reset(self) -> None:
        """Reset all statistics."""
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.rejected_calls = 0
        
        self.exception_failures = 0
        self.timeout_failures = 0
        self.slow_call_failures = 0
        self.custom_failures = 0
        
        self.total_duration_ms = 0.0
        self.min_duration_ms = float('inf')
        self.max_duration_ms = 0.0
        
        self.recent_calls.clear()


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""
    
    def __init__(self, circuit_name: str, retry_after_seconds: float):
        self.circuit_name = circuit_name
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            f"Circuit breaker '{circuit_name}' is open. "
            f"Retry after {retry_after_seconds:.1f} seconds."
        )


class CircuitBreaker:
    """
    Circuit breaker implementation for preventing cascade failures.
    
    Features:
    - Configurable failure thresholds and timeouts
    - State management (CLOSED, OPEN, HALF_OPEN)
    - Automatic recovery testing
    - Failure counting and rate calculation
    - Slow call detection
    - Custom failure predicates
    - Comprehensive statistics and monitoring
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # State management
        self._state = CircuitState.CLOSED
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        
        # Failure tracking
        self._failure_window: deque = deque()
        self._call_history: deque = deque(maxlen=1000)
        
        # Statistics
        self._stats = CircuitBreakerStats()
        
        # Thread safety
        self._lock = RLock()
        
        # Telemetry
        self._telemetry = get_global_telemetry() if self.config.enable_metrics else None
        
        # State change callbacks
        self._state_change_callbacks: List[Callable[[CircuitState, CircuitState], None]] = []
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit breaker state."""
        return self._state
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting calls)."""
        return self._state == CircuitState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self._state == CircuitState.HALF_OPEN
    
    def add_state_change_callback(
        self, 
        callback: Callable[[CircuitState, CircuitState], None]
    ) -> None:
        """Add callback for state changes."""
        self._state_change_callbacks.append(callback)
    
    def remove_state_change_callback(
        self, 
        callback: Callable[[CircuitState, CircuitState], None]
    ) -> None:
        """Remove state change callback."""
        if callback in self._state_change_callbacks:
            self._state_change_callbacks.remove(callback)
    
    async def call(
        self, 
        func: Callable[..., Any], 
        *args, 
        timeout_seconds: Optional[float] = None,
        **kwargs
    ) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            timeout_seconds: Optional timeout for the call
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenException: If circuit is open
            Exception: Original exception from function
        """
        # Check if call should be allowed
        if not self._should_allow_call():
            with self._lock:
                self._stats.update_rejection()
            
            if self._telemetry:
                self._telemetry.increment_counter(
                    "circuit_breaker.calls_rejected",
                    tags={"circuit_name": self.name}
                )
            
            retry_after = self._calculate_retry_after()
            raise CircuitBreakerOpenException(self.name, retry_after)
        
        # Execute the call
        start_time = time.perf_counter()
        call_result = None
        
        try:
            # Execute with optional timeout
            if timeout_seconds and asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs), 
                    timeout=timeout_seconds
                )
            elif asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Check for slow calls
            is_slow = duration_ms > self.config.slow_call_threshold_ms
            
            # Check custom failure predicate
            is_custom_failure = (
                self.config.custom_failure_predicate and
                self.config.custom_failure_predicate(result)
            )
            
            if is_slow:
                call_result = CallResult(
                    success=False,
                    duration_ms=duration_ms,
                    failure_type=FailureType.SLOW_RESPONSE
                )
            elif is_custom_failure:
                call_result = CallResult(
                    success=False,
                    duration_ms=duration_ms,
                    failure_type=FailureType.CUSTOM
                )
            else:
                call_result = CallResult(
                    success=True,
                    duration_ms=duration_ms
                )
            
            # Record successful call
            self._record_call_result(call_result)
            
            # Handle half-open state success
            if self._state == CircuitState.HALF_OPEN and call_result.success:
                self._handle_half_open_success()
            
            return result
            
        except asyncio.TimeoutError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            call_result = CallResult(
                success=False,
                duration_ms=duration_ms,
                failure_type=FailureType.TIMEOUT,
                error=e
            )
            
            self._record_call_result(call_result)
            self._handle_failure()
            raise
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            call_result = CallResult(
                success=False,
                duration_ms=duration_ms,
                failure_type=FailureType.EXCEPTION,
                error=e
            )
            
            self._record_call_result(call_result)
            self._handle_failure()
            raise
    
    def _should_allow_call(self) -> bool:
        """Check if call should be allowed based on current state."""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            elif self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if time.time() - self._last_failure_time >= self.config.recovery_timeout_seconds:
                    self._transition_to_half_open()
                    return True
                return False
            elif self._state == CircuitState.HALF_OPEN:
                # Allow limited calls in half-open state
                return self._half_open_calls < self.config.half_open_max_calls
            
            return False
    
    def _record_call_result(self, result: CallResult) -> None:
        """Record call result and update statistics."""
        with self._lock:
            # Update statistics
            self._stats.update_call(result)
            
            # Add to call history
            self._call_history.append(result)
            
            # Update failure window for failure rate calculation
            current_time = time.time()
            
            # Remove old failures outside the window
            while (self._failure_window and 
                   current_time - self._failure_window[0] > self.config.failure_window_seconds):
                self._failure_window.popleft()
            
            # Add current failure if applicable
            if not result.success:
                self._failure_window.append(current_time)
        
        # Record telemetry
        if self._telemetry:
            self._telemetry.increment_counter(
                "circuit_breaker.calls_total",
                tags={
                    "circuit_name": self.name,
                    "success": str(result.success).lower(),
                    "failure_type": result.failure_type.value if result.failure_type else "none"
                }
            )
            
            self._telemetry.record_histogram(
                "circuit_breaker.call_duration_ms",
                result.duration_ms,
                tags={
                    "circuit_name": self.name,
                    "success": str(result.success).lower()
                }
            )
    
    def _handle_failure(self) -> None:
        """Handle call failure and potentially open circuit."""
        with self._lock:
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                # Failure in half-open state, go back to open
                self._transition_to_open()
            elif self._state == CircuitState.CLOSED:
                # Check if we should open the circuit
                if self._should_open_circuit():
                    self._transition_to_open()
    
    def _handle_half_open_success(self) -> None:
        """Handle successful call in half-open state."""
        with self._lock:
            self._half_open_calls += 1
            
            # If we've had enough successful calls, close the circuit
            if self._half_open_calls >= self.config.half_open_max_calls:
                self._transition_to_closed()
    
    def _should_open_circuit(self) -> bool:
        """Check if circuit should be opened based on failure criteria."""
        # Check minimum calls requirement
        if self._stats.total_calls < self.config.minimum_calls:
            return False
        
        # Check failure count threshold
        if len(self._failure_window) >= self.config.failure_threshold:
            return True
        
        # Check failure rate threshold
        recent_calls = min(self.config.minimum_calls, len(self._call_history))
        if recent_calls > 0:
            recent_failures = sum(
                1 for call in list(self._call_history)[-recent_calls:] 
                if not call.success
            )
            failure_rate = recent_failures / recent_calls
            
            if failure_rate >= self.config.failure_rate_threshold:
                return True
        
        return False
    
    def _transition_to_open(self) -> None:
        """Transition circuit to open state."""
        old_state = self._state
        self._state = CircuitState.OPEN
        self._half_open_calls = 0
        
        self._stats.update_state_change(self._state)
        self._notify_state_change(old_state, self._state)
        
        if self._telemetry:
            self._telemetry.increment_counter(
                "circuit_breaker.state_transitions",
                tags={
                    "circuit_name": self.name,
                    "from_state": old_state.value,
                    "to_state": self._state.value
                }
            )
    
    def _transition_to_half_open(self) -> None:
        """Transition circuit to half-open state."""
        old_state = self._state
        self._state = CircuitState.HALF_OPEN
        self._half_open_calls = 0
        
        self._stats.update_state_change(self._state)
        self._notify_state_change(old_state, self._state)
        
        if self._telemetry:
            self._telemetry.increment_counter(
                "circuit_breaker.state_transitions",
                tags={
                    "circuit_name": self.name,
                    "from_state": old_state.value,
                    "to_state": self._state.value
                }
            )
    
    def _transition_to_closed(self) -> None:
        """Transition circuit to closed state."""
        old_state = self._state
        self._state = CircuitState.CLOSED
        self._half_open_calls = 0
        
        # Clear failure window on successful recovery
        self._failure_window.clear()
        
        self._stats.update_state_change(self._state)
        self._notify_state_change(old_state, self._state)
        
        if self._telemetry:
            self._telemetry.increment_counter(
                "circuit_breaker.state_transitions",
                tags={
                    "circuit_name": self.name,
                    "from_state": old_state.value,
                    "to_state": self._state.value
                }
            )
    
    def _notify_state_change(self, old_state: CircuitState, new_state: CircuitState) -> None:
        """Notify registered callbacks of state change."""
        for callback in self._state_change_callbacks:
            try:
                callback(old_state, new_state)
            except Exception:
                # Don't let callback errors affect circuit breaker operation
                continue
    
    def _calculate_retry_after(self) -> float:
        """Calculate seconds until retry is allowed."""
        if self._state == CircuitState.OPEN:
            elapsed = time.time() - self._last_failure_time
            return max(0, self.config.recovery_timeout_seconds - elapsed)
        return 0.0
    
    def force_open(self) -> None:
        """Force circuit breaker to open state."""
        with self._lock:
            if self._state != CircuitState.OPEN:
                self._transition_to_open()
    
    def force_closed(self) -> None:
        """Force circuit breaker to closed state."""
        with self._lock:
            if self._state != CircuitState.CLOSED:
                self._transition_to_closed()
    
    def force_half_open(self) -> None:
        """Force circuit breaker to half-open state."""
        with self._lock:
            if self._state != CircuitState.HALF_OPEN:
                self._transition_to_half_open()
    
    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        with self._lock:
            old_state = self._state
            self._state = CircuitState.CLOSED
            self._last_failure_time = 0.0
            self._half_open_calls = 0
            self._failure_window.clear()
            self._call_history.clear()
            self._stats.reset()
            
            if old_state != self._state:
                self._notify_state_change(old_state, self._state)
    
    def get_stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics."""
        with self._lock:
            return self._stats
    
    def get_health_info(self) -> Dict[str, Any]:
        """Get comprehensive health information."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "is_healthy": self._state == CircuitState.CLOSED,
                "failure_count": len(self._failure_window),
                "failure_rate": self._stats.failure_rate,
                "success_rate": self._stats.success_rate,
                "rejection_rate": self._stats.rejection_rate,
                "average_duration_ms": self._stats.average_duration_ms,
                "total_calls": self._stats.total_calls,
                "rejected_calls": self._stats.rejected_calls,
                "last_failure_time": self._last_failure_time,
                "retry_after_seconds": self._calculate_retry_after(),
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "failure_rate_threshold": self.config.failure_rate_threshold,
                    "recovery_timeout_seconds": self.config.recovery_timeout_seconds,
                    "minimum_calls": self.config.minimum_calls
                }
            }


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = RLock()
    
    def get_or_create(
        self, 
        name: str, 
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get existing circuit breaker or create new one."""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        with self._lock:
            return self._breakers.get(name)
    
    def remove(self, name: str) -> bool:
        """Remove circuit breaker from registry."""
        with self._lock:
            if name in self._breakers:
                del self._breakers[name]
                return True
            return False
    
    def list_breakers(self) -> List[str]:
        """List all registered circuit breaker names."""
        with self._lock:
            return list(self._breakers.keys())
    
    def get_all_health_info(self) -> Dict[str, Dict[str, Any]]:
        """Get health information for all circuit breakers."""
        with self._lock:
            return {
                name: breaker.get_health_info()
                for name, breaker in self._breakers.items()
            }
    
    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
    
    def force_open_all(self) -> None:
        """Force all circuit breakers to open state."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.force_open()
    
    def force_closed_all(self) -> None:
        """Force all circuit breakers to closed state."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.force_closed()


# Global circuit breaker registry
_global_registry: Optional[CircuitBreakerRegistry] = None


def get_global_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = CircuitBreakerRegistry()
    return _global_registry


# Convenience functions
def get_circuit_breaker(
    name: str, 
    config: Optional[CircuitBreakerConfig] = None
) -> CircuitBreaker:
    """Get or create circuit breaker."""
    registry = get_global_registry()
    return registry.get_or_create(name, config)


async def with_circuit_breaker(
    name: str,
    func: Callable[..., Any],
    *args,
    config: Optional[CircuitBreakerConfig] = None,
    timeout_seconds: Optional[float] = None,
    **kwargs
) -> Any:
    """Execute function with circuit breaker protection."""
    breaker = get_circuit_breaker(name, config)
    return await breaker.call(func, *args, timeout_seconds=timeout_seconds, **kwargs)


def circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
    timeout_seconds: Optional[float] = None
):
    """Decorator for circuit breaker protection."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            return await with_circuit_breaker(
                name, func, *args, 
                config=config, 
                timeout_seconds=timeout_seconds, 
                **kwargs
            )
        
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we need to handle differently
            breaker = get_circuit_breaker(name, config)
            
            # Check if call should be allowed
            if not breaker._should_allow_call():
                retry_after = breaker._calculate_retry_after()
                raise CircuitBreakerOpenException(name, retry_after)
            
            # Execute the call
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                # Check for slow calls
                is_slow = duration_ms > breaker.config.slow_call_threshold_ms
                is_custom_failure = (
                    breaker.config.custom_failure_predicate and
                    breaker.config.custom_failure_predicate(result)
                )
                
                if is_slow or is_custom_failure:
                    call_result = CallResult(
                        success=False,
                        duration_ms=duration_ms,
                        failure_type=FailureType.SLOW_RESPONSE if is_slow else FailureType.CUSTOM
                    )
                    breaker._record_call_result(call_result)
                    breaker._handle_failure()
                else:
                    call_result = CallResult(success=True, duration_ms=duration_ms)
                    breaker._record_call_result(call_result)
                    
                    if breaker._state == CircuitState.HALF_OPEN:
                        breaker._handle_half_open_success()
                
                return result
                
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                call_result = CallResult(
                    success=False,
                    duration_ms=duration_ms,
                    failure_type=FailureType.EXCEPTION,
                    error=e
                )
                
                breaker._record_call_result(call_result)
                breaker._handle_failure()
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator