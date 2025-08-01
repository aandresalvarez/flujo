"""
Error recovery strategies for common failure patterns.

This module provides common recovery patterns (retry, fallback, circuit breaking),
error classification, automatic strategy selection, recovery success tracking,
and strategy optimization for robust error handling.
"""

import asyncio
import time
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union, Type, Tuple
from threading import RLock
from enum import Enum
import logging

from .optimized_error_handler import (
    ErrorContext, RecoveryStrategy, RecoveryAction, ErrorCategory, 
    ErrorSeverity, ErrorRecoveryResult
)
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, get_circuit_breaker
from .optimized_telemetry import get_global_telemetry, MetricType


logger = logging.getLogger(__name__)


class RetryPolicy(Enum):
    """Retry policy types."""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    JITTERED_BACKOFF = "jittered_backoff"
    FIBONACCI_BACKOFF = "fibonacci_backoff"


@dataclass
class RetryConfig:
    """Configuration for retry strategy."""
    
    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    policy: RetryPolicy = RetryPolicy.EXPONENTIAL_BACKOFF
    backoff_multiplier: float = 2.0
    jitter_factor: float = 0.1  # For jittered backoff
    
    # Retry conditions
    retry_on_exceptions: Optional[List[Type[Exception]]] = None
    retry_on_status_codes: Optional[List[int]] = None
    retry_predicate: Optional[Callable[[Exception], bool]] = None
    
    # Circuit breaker integration
    enable_circuit_breaker: bool = True
    circuit_breaker_name: Optional[str] = None


@dataclass
class FallbackConfig:
    """Configuration for fallback strategy."""
    
    fallback_value: Any = None
    fallback_function: Optional[Callable[..., Any]] = None
    fallback_exception_handler: Optional[Callable[[Exception], Any]] = None
    
    # Fallback conditions
    fallback_on_exceptions: Optional[List[Type[Exception]]] = None
    fallback_predicate: Optional[Callable[[Exception], bool]] = None
    
    # Fallback behavior
    log_fallback_usage: bool = True
    track_fallback_metrics: bool = True


@dataclass
class StrategyStats:
    """Statistics for recovery strategy usage."""
    
    strategy_name: str
    total_attempts: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    
    # Timing statistics
    total_recovery_time_ms: float = 0.0
    min_recovery_time_ms: float = float('inf')
    max_recovery_time_ms: float = 0.0
    
    # Usage patterns
    error_type_usage: Dict[str, int] = field(default_factory=dict)
    category_usage: Dict[str, int] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate strategy success rate."""
        if self.total_attempts == 0:
            return 0.0
        return self.successful_recoveries / self.total_attempts
    
    @property
    def average_recovery_time_ms(self) -> float:
        """Calculate average recovery time."""
        if self.successful_recoveries == 0:
            return 0.0
        return self.total_recovery_time_ms / self.successful_recoveries
    
    def update_attempt(
        self, 
        success: bool, 
        recovery_time_ms: float,
        error_type: str,
        category: str
    ) -> None:
        """Update statistics with recovery attempt."""
        self.total_attempts += 1
        
        if success:
            self.successful_recoveries += 1
            self.total_recovery_time_ms += recovery_time_ms
            self.min_recovery_time_ms = min(self.min_recovery_time_ms, recovery_time_ms)
            self.max_recovery_time_ms = max(self.max_recovery_time_ms, recovery_time_ms)
        else:
            self.failed_recoveries += 1
        
        # Update usage patterns
        self.error_type_usage[error_type] = self.error_type_usage.get(error_type, 0) + 1
        self.category_usage[category] = self.category_usage.get(category, 0) + 1


class BaseRecoveryStrategy(ABC):
    """Base class for recovery strategies."""
    
    def __init__(self, name: str, enable_telemetry: bool = True):
        self.name = name
        self.enable_telemetry = enable_telemetry
        self._stats = StrategyStats(name)
        self._telemetry = get_global_telemetry() if enable_telemetry else None
        self._lock = RLock()
    
    @abstractmethod
    async def execute_recovery(
        self, 
        error_context: ErrorContext,
        original_func: Callable[..., Any],
        *args,
        **kwargs
    ) -> ErrorRecoveryResult:
        """Execute recovery strategy."""
        pass
    
    @abstractmethod
    def should_apply(self, error_context: ErrorContext) -> bool:
        """Check if strategy should be applied to error."""
        pass
    
    def get_stats(self) -> StrategyStats:
        """Get strategy statistics."""
        with self._lock:
            return self._stats
    
    def reset_stats(self) -> None:
        """Reset strategy statistics."""
        with self._lock:
            self._stats = StrategyStats(self.name)
    
    def _record_attempt(
        self, 
        success: bool, 
        recovery_time_ms: float,
        error_context: ErrorContext
    ) -> None:
        """Record recovery attempt in statistics."""
        with self._lock:
            self._stats.update_attempt(
                success, 
                recovery_time_ms,
                error_context.error_type,
                error_context.category.value
            )
        
        if self._telemetry:
            self._telemetry.increment_counter(
                "recovery_strategy.attempts",
                tags={
                    "strategy": self.name,
                    "success": str(success).lower(),
                    "error_type": error_context.error_type,
                    "category": error_context.category.value
                }
            )
            
            if success:
                self._telemetry.record_histogram(
                    "recovery_strategy.recovery_time_ms",
                    recovery_time_ms,
                    tags={"strategy": self.name}
                )


class RetryStrategy(BaseRecoveryStrategy):
    """Retry strategy with configurable backoff policies."""
    
    def __init__(
        self, 
        name: str = "retry_strategy",
        config: Optional[RetryConfig] = None,
        enable_telemetry: bool = True
    ):
        super().__init__(name, enable_telemetry)
        self.config = config or RetryConfig()
        self._circuit_breaker: Optional[CircuitBreaker] = None
        
        # Initialize circuit breaker if enabled
        if self.config.enable_circuit_breaker:
            cb_name = self.config.circuit_breaker_name or f"{name}_circuit_breaker"
            cb_config = CircuitBreakerConfig(
                failure_threshold=self.config.max_attempts,
                recovery_timeout_seconds=self.config.max_delay_seconds
            )
            self._circuit_breaker = get_circuit_breaker(cb_name, cb_config)
    
    def should_apply(self, error_context: ErrorContext) -> bool:
        """Check if retry strategy should be applied."""
        # Check if error is retryable
        if not error_context.is_recoverable:
            return False
        
        # Check attempt limit
        if error_context.attempt_number >= self.config.max_attempts:
            return False
        
        # Check specific exception types
        if self.config.retry_on_exceptions:
            if type(error_context.error) not in self.config.retry_on_exceptions:
                return False
        
        # Check custom predicate
        if self.config.retry_predicate:
            if not self.config.retry_predicate(error_context.error):
                return False
        
        # Check categories that are generally retryable
        retryable_categories = {
            ErrorCategory.NETWORK,
            ErrorCategory.TIMEOUT,
            ErrorCategory.EXTERNAL_SERVICE
        }
        
        return error_context.category in retryable_categories
    
    async def execute_recovery(
        self, 
        error_context: ErrorContext,
        original_func: Callable[..., Any],
        *args,
        **kwargs
    ) -> ErrorRecoveryResult:
        """Execute retry strategy."""
        start_time = time.perf_counter()
        
        for attempt in range(error_context.attempt_number, self.config.max_attempts + 1):
            try:
                # Calculate delay for this attempt
                if attempt > 1:
                    delay = self._calculate_delay(attempt - 1)
                    if delay > 0:
                        await asyncio.sleep(delay)
                
                # Execute function with circuit breaker if enabled
                if self._circuit_breaker:
                    result = await self._circuit_breaker.call(original_func, *args, **kwargs)
                else:
                    if asyncio.iscoroutinefunction(original_func):
                        result = await original_func(*args, **kwargs)
                    else:
                        result = original_func(*args, **kwargs)
                
                # Success!
                recovery_time_ms = (time.perf_counter() - start_time) * 1000
                
                recovery_result = ErrorRecoveryResult(
                    success=True,
                    action_taken=RecoveryAction.RETRY,
                    recovery_time_ms=recovery_time_ms,
                    attempts_made=attempt,
                    strategy_used=self.name,
                    recovered_value=result,
                    metadata={
                        "total_attempts": attempt,
                        "final_delay_seconds": delay if attempt > 1 else 0
                    }
                )
                
                self._record_attempt(True, recovery_time_ms, error_context)
                return recovery_result
                
            except Exception as retry_error:
                if attempt == self.config.max_attempts:
                    # Final attempt failed
                    recovery_time_ms = (time.perf_counter() - start_time) * 1000
                    
                    recovery_result = ErrorRecoveryResult(
                        success=False,
                        action_taken=RecoveryAction.RETRY,
                        recovery_time_ms=recovery_time_ms,
                        attempts_made=attempt,
                        strategy_used=self.name,
                        new_error=retry_error,
                        metadata={"total_attempts": attempt}
                    )
                    
                    self._record_attempt(False, recovery_time_ms, error_context)
                    return recovery_result
                
                # Continue to next attempt
                continue
        
        # Should not reach here, but handle gracefully
        recovery_time_ms = (time.perf_counter() - start_time) * 1000
        recovery_result = ErrorRecoveryResult(
            success=False,
            action_taken=RecoveryAction.RETRY,
            recovery_time_ms=recovery_time_ms,
            attempts_made=self.config.max_attempts,
            strategy_used=self.name
        )
        
        self._record_attempt(False, recovery_time_ms, error_context)
        return recovery_result
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        if self.config.policy == RetryPolicy.FIXED_DELAY:
            return self.config.base_delay_seconds
        
        elif self.config.policy == RetryPolicy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay_seconds * (self.config.backoff_multiplier ** attempt)
            return min(delay, self.config.max_delay_seconds)
        
        elif self.config.policy == RetryPolicy.LINEAR_BACKOFF:
            delay = self.config.base_delay_seconds * (attempt + 1)
            return min(delay, self.config.max_delay_seconds)
        
        elif self.config.policy == RetryPolicy.JITTERED_BACKOFF:
            base_delay = self.config.base_delay_seconds * (self.config.backoff_multiplier ** attempt)
            jitter = base_delay * self.config.jitter_factor * random.random()
            delay = base_delay + jitter
            return min(delay, self.config.max_delay_seconds)
        
        elif self.config.policy == RetryPolicy.FIBONACCI_BACKOFF:
            # Calculate Fibonacci number for attempt
            fib = self._fibonacci(attempt + 1)
            delay = self.config.base_delay_seconds * fib
            return min(delay, self.config.max_delay_seconds)
        
        else:
            return self.config.base_delay_seconds
    
    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number."""
        if n <= 1:
            return n
        
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        
        return b


class FallbackStrategy(BaseRecoveryStrategy):
    """Fallback strategy with configurable fallback values or functions."""
    
    def __init__(
        self, 
        name: str = "fallback_strategy",
        config: Optional[FallbackConfig] = None,
        enable_telemetry: bool = True
    ):
        super().__init__(name, enable_telemetry)
        self.config = config or FallbackConfig()
    
    def should_apply(self, error_context: ErrorContext) -> bool:
        """Check if fallback strategy should be applied."""
        # Check specific exception types
        if self.config.fallback_on_exceptions:
            if type(error_context.error) not in self.config.fallback_on_exceptions:
                return False
        
        # Check custom predicate
        if self.config.fallback_predicate:
            if not self.config.fallback_predicate(error_context.error):
                return False
        
        # Check if we have a fallback mechanism
        has_fallback = (
            self.config.fallback_value is not None or
            self.config.fallback_function is not None or
            self.config.fallback_exception_handler is not None
        )
        
        if not has_fallback:
            return False
        
        # Categories that benefit from fallback
        fallback_categories = {
            ErrorCategory.VALIDATION,
            ErrorCategory.EXTERNAL_SERVICE,
            ErrorCategory.CONFIGURATION,
            ErrorCategory.NETWORK
        }
        
        return error_context.category in fallback_categories
    
    async def execute_recovery(
        self, 
        error_context: ErrorContext,
        original_func: Callable[..., Any],
        *args,
        **kwargs
    ) -> ErrorRecoveryResult:
        """Execute fallback strategy."""
        start_time = time.perf_counter()
        
        try:
            fallback_value = None
            
            # Try fallback function first
            if self.config.fallback_function:
                if asyncio.iscoroutinefunction(self.config.fallback_function):
                    fallback_value = await self.config.fallback_function(*args, **kwargs)
                else:
                    fallback_value = self.config.fallback_function(*args, **kwargs)
            
            # Try exception handler
            elif self.config.fallback_exception_handler:
                if asyncio.iscoroutinefunction(self.config.fallback_exception_handler):
                    fallback_value = await self.config.fallback_exception_handler(error_context.error)
                else:
                    fallback_value = self.config.fallback_exception_handler(error_context.error)
            
            # Use static fallback value
            else:
                fallback_value = self.config.fallback_value
            
            recovery_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Log fallback usage if enabled
            if self.config.log_fallback_usage:
                logger.info(
                    f"Fallback strategy '{self.name}' used for error: {error_context.error_type}"
                )
            
            recovery_result = ErrorRecoveryResult(
                success=True,
                action_taken=RecoveryAction.FALLBACK,
                recovery_time_ms=recovery_time_ms,
                attempts_made=1,
                strategy_used=self.name,
                recovered_value=fallback_value,
                metadata={
                    "fallback_type": "function" if self.config.fallback_function else "value",
                    "original_error": str(error_context.error)
                }
            )
            
            self._record_attempt(True, recovery_time_ms, error_context)
            return recovery_result
            
        except Exception as fallback_error:
            recovery_time_ms = (time.perf_counter() - start_time) * 1000
            
            recovery_result = ErrorRecoveryResult(
                success=False,
                action_taken=RecoveryAction.FALLBACK,
                recovery_time_ms=recovery_time_ms,
                attempts_made=1,
                strategy_used=self.name,
                new_error=fallback_error,
                metadata={"fallback_error": str(fallback_error)}
            )
            
            self._record_attempt(False, recovery_time_ms, error_context)
            return recovery_result


class CircuitBreakerStrategy(BaseRecoveryStrategy):
    """Circuit breaker strategy for preventing cascade failures."""
    
    def __init__(
        self, 
        name: str = "circuit_breaker_strategy",
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        enable_telemetry: bool = True
    ):
        super().__init__(name, enable_telemetry)
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    def should_apply(self, error_context: ErrorContext) -> bool:
        """Check if circuit breaker strategy should be applied."""
        # Categories that benefit from circuit breaking
        circuit_breaker_categories = {
            ErrorCategory.NETWORK,
            ErrorCategory.TIMEOUT,
            ErrorCategory.EXTERNAL_SERVICE,
            ErrorCategory.RESOURCE_EXHAUSTION
        }
        
        return error_context.category in circuit_breaker_categories
    
    async def execute_recovery(
        self, 
        error_context: ErrorContext,
        original_func: Callable[..., Any],
        *args,
        **kwargs
    ) -> ErrorRecoveryResult:
        """Execute circuit breaker strategy."""
        start_time = time.perf_counter()
        
        # Get or create circuit breaker for this context
        cb_name = f"{self.name}_{error_context.step_name or 'unknown'}"
        circuit_breaker = self._get_circuit_breaker(cb_name)
        
        try:
            # Execute function with circuit breaker protection
            result = await circuit_breaker.call(original_func, *args, **kwargs)
            
            recovery_time_ms = (time.perf_counter() - start_time) * 1000
            
            recovery_result = ErrorRecoveryResult(
                success=True,
                action_taken=RecoveryAction.CIRCUIT_BREAK,
                recovery_time_ms=recovery_time_ms,
                attempts_made=1,
                strategy_used=self.name,
                recovered_value=result,
                metadata={
                    "circuit_breaker_name": cb_name,
                    "circuit_state": circuit_breaker.state.value
                }
            )
            
            self._record_attempt(True, recovery_time_ms, error_context)
            return recovery_result
            
        except Exception as cb_error:
            recovery_time_ms = (time.perf_counter() - start_time) * 1000
            
            recovery_result = ErrorRecoveryResult(
                success=False,
                action_taken=RecoveryAction.CIRCUIT_BREAK,
                recovery_time_ms=recovery_time_ms,
                attempts_made=1,
                strategy_used=self.name,
                new_error=cb_error,
                metadata={
                    "circuit_breaker_name": cb_name,
                    "circuit_state": circuit_breaker.state.value
                }
            )
            
            self._record_attempt(False, recovery_time_ms, error_context)
            return recovery_result
    
    def _get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create circuit breaker for name."""
        if name not in self._circuit_breakers:
            self._circuit_breakers[name] = CircuitBreaker(name, self.circuit_breaker_config)
        return self._circuit_breakers[name]
    
    def get_circuit_breaker_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        return {
            name: cb.get_health_info()
            for name, cb in self._circuit_breakers.items()
        }


class CompositeStrategy(BaseRecoveryStrategy):
    """Composite strategy that combines multiple recovery strategies."""
    
    def __init__(
        self, 
        name: str = "composite_strategy",
        strategies: Optional[List[BaseRecoveryStrategy]] = None,
        enable_telemetry: bool = True
    ):
        super().__init__(name, enable_telemetry)
        self.strategies = strategies or []
        self._strategy_selector: Optional[Callable[[ErrorContext, List[BaseRecoveryStrategy]], BaseRecoveryStrategy]] = None
    
    def add_strategy(self, strategy: BaseRecoveryStrategy) -> None:
        """Add strategy to composite."""
        self.strategies.append(strategy)
    
    def remove_strategy(self, strategy_name: str) -> bool:
        """Remove strategy from composite."""
        for i, strategy in enumerate(self.strategies):
            if strategy.name == strategy_name:
                del self.strategies[i]
                return True
        return False
    
    def set_strategy_selector(
        self, 
        selector: Callable[[ErrorContext, List[BaseRecoveryStrategy]], BaseRecoveryStrategy]
    ) -> None:
        """Set custom strategy selector function."""
        self._strategy_selector = selector
    
    def should_apply(self, error_context: ErrorContext) -> bool:
        """Check if any strategy should be applied."""
        return any(strategy.should_apply(error_context) for strategy in self.strategies)
    
    async def execute_recovery(
        self, 
        error_context: ErrorContext,
        original_func: Callable[..., Any],
        *args,
        **kwargs
    ) -> ErrorRecoveryResult:
        """Execute composite recovery strategy."""
        start_time = time.perf_counter()
        
        # Find applicable strategies
        applicable_strategies = [
            strategy for strategy in self.strategies
            if strategy.should_apply(error_context)
        ]
        
        if not applicable_strategies:
            recovery_time_ms = (time.perf_counter() - start_time) * 1000
            recovery_result = ErrorRecoveryResult(
                success=False,
                action_taken=RecoveryAction.ABORT,
                recovery_time_ms=recovery_time_ms,
                strategy_used=self.name,
                metadata={"reason": "no_applicable_strategies"}
            )
            
            self._record_attempt(False, recovery_time_ms, error_context)
            return recovery_result
        
        # Select strategy to use
        if self._strategy_selector:
            selected_strategy = self._strategy_selector(error_context, applicable_strategies)
        else:
            # Default selection: prioritize by strategy type
            selected_strategy = self._default_strategy_selection(applicable_strategies)
        
        # Execute selected strategy
        try:
            result = await selected_strategy.execute_recovery(
                error_context, original_func, *args, **kwargs
            )
            
            recovery_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Update result with composite information
            result.strategy_used = f"{self.name}:{selected_strategy.name}"
            result.metadata.update({
                "composite_strategy": self.name,
                "selected_strategy": selected_strategy.name,
                "applicable_strategies": [s.name for s in applicable_strategies]
            })
            
            self._record_attempt(result.success, recovery_time_ms, error_context)
            return result
            
        except Exception as composite_error:
            recovery_time_ms = (time.perf_counter() - start_time) * 1000
            
            recovery_result = ErrorRecoveryResult(
                success=False,
                action_taken=RecoveryAction.ABORT,
                recovery_time_ms=recovery_time_ms,
                strategy_used=f"{self.name}:{selected_strategy.name}",
                new_error=composite_error,
                metadata={
                    "composite_strategy": self.name,
                    "selected_strategy": selected_strategy.name,
                    "composite_error": str(composite_error)
                }
            )
            
            self._record_attempt(False, recovery_time_ms, error_context)
            return recovery_result
    
    def _default_strategy_selection(
        self, 
        strategies: List[BaseRecoveryStrategy]
    ) -> BaseRecoveryStrategy:
        """Default strategy selection logic."""
        # Priority order: Circuit Breaker > Retry > Fallback
        for strategy_type in [CircuitBreakerStrategy, RetryStrategy, FallbackStrategy]:
            for strategy in strategies:
                if isinstance(strategy, strategy_type):
                    return strategy
        
        # Return first available strategy
        return strategies[0]
    
    def get_all_stats(self) -> Dict[str, StrategyStats]:
        """Get statistics for all strategies."""
        stats = {"composite": self.get_stats()}
        
        for strategy in self.strategies:
            stats[strategy.name] = strategy.get_stats()
        
        return stats


class StrategyRegistry:
    """Registry for managing recovery strategies."""
    
    def __init__(self):
        self._strategies: Dict[str, BaseRecoveryStrategy] = {}
        self._lock = RLock()
        
        # Register default strategies
        self._register_default_strategies()
    
    def register(self, strategy: BaseRecoveryStrategy) -> None:
        """Register a recovery strategy."""
        with self._lock:
            self._strategies[strategy.name] = strategy
    
    def unregister(self, strategy_name: str) -> bool:
        """Unregister a recovery strategy."""
        with self._lock:
            if strategy_name in self._strategies:
                del self._strategies[strategy_name]
                return True
            return False
    
    def get(self, strategy_name: str) -> Optional[BaseRecoveryStrategy]:
        """Get strategy by name."""
        with self._lock:
            return self._strategies.get(strategy_name)
    
    def list_strategies(self) -> List[str]:
        """List all registered strategy names."""
        with self._lock:
            return list(self._strategies.keys())
    
    def find_applicable_strategies(
        self, 
        error_context: ErrorContext
    ) -> List[BaseRecoveryStrategy]:
        """Find all strategies applicable to error context."""
        with self._lock:
            return [
                strategy for strategy in self._strategies.values()
                if strategy.should_apply(error_context)
            ]
    
    def get_best_strategy(
        self, 
        error_context: ErrorContext
    ) -> Optional[BaseRecoveryStrategy]:
        """Get best strategy for error context."""
        applicable = self.find_applicable_strategies(error_context)
        
        if not applicable:
            return None
        
        # Select strategy with highest success rate for this error type
        best_strategy = None
        best_score = -1.0
        
        for strategy in applicable:
            stats = strategy.get_stats()
            
            # Calculate score based on success rate and usage for this error type
            success_rate = stats.success_rate
            usage_count = stats.error_type_usage.get(error_context.error_type, 0)
            
            # Boost score for strategies with experience with this error type
            experience_boost = min(usage_count / 10.0, 0.2)  # Max 20% boost
            score = success_rate + experience_boost
            
            if score > best_score:
                best_score = score
                best_strategy = strategy
        
        return best_strategy
    
    def get_all_stats(self) -> Dict[str, StrategyStats]:
        """Get statistics for all strategies."""
        with self._lock:
            return {
                name: strategy.get_stats()
                for name, strategy in self._strategies.items()
            }
    
    def _register_default_strategies(self) -> None:
        """Register default recovery strategies."""
        # Network retry strategy
        network_retry = RetryStrategy(
            "network_retry",
            RetryConfig(
                max_attempts=3,
                base_delay_seconds=1.0,
                policy=RetryPolicy.EXPONENTIAL_BACKOFF,
                retry_on_exceptions=[ConnectionError, TimeoutError]
            )
        )
        self.register(network_retry)
        
        # Validation fallback strategy
        validation_fallback = FallbackStrategy(
            "validation_fallback",
            FallbackConfig(
                fallback_value=None,
                fallback_on_exceptions=[ValueError, TypeError]
            )
        )
        self.register(validation_fallback)
        
        # External service circuit breaker
        external_circuit_breaker = CircuitBreakerStrategy(
            "external_service_circuit_breaker",
            CircuitBreakerConfig(
                failure_threshold=5,
                failure_rate_threshold=0.5,
                recovery_timeout_seconds=30
            )
        )
        self.register(external_circuit_breaker)
        
        # Composite strategy for comprehensive recovery
        comprehensive_strategy = CompositeStrategy(
            "comprehensive_recovery",
            [network_retry, validation_fallback, external_circuit_breaker]
        )
        self.register(comprehensive_strategy)


# Global strategy registry
_global_strategy_registry: Optional[StrategyRegistry] = None


def get_global_strategy_registry() -> StrategyRegistry:
    """Get the global strategy registry."""
    global _global_strategy_registry
    if _global_strategy_registry is None:
        _global_strategy_registry = StrategyRegistry()
    return _global_strategy_registry


# Convenience functions
def register_strategy(strategy: BaseRecoveryStrategy) -> None:
    """Register a recovery strategy."""
    registry = get_global_strategy_registry()
    registry.register(strategy)


def get_strategy(strategy_name: str) -> Optional[BaseRecoveryStrategy]:
    """Get strategy by name."""
    registry = get_global_strategy_registry()
    return registry.get(strategy_name)


def find_best_strategy(error_context: ErrorContext) -> Optional[BaseRecoveryStrategy]:
    """Find best strategy for error context."""
    registry = get_global_strategy_registry()
    return registry.get_best_strategy(error_context)


async def execute_recovery(
    error_context: ErrorContext,
    original_func: Callable[..., Any],
    *args,
    strategy_name: Optional[str] = None,
    **kwargs
) -> ErrorRecoveryResult:
    """Execute recovery using specified or best strategy."""
    registry = get_global_strategy_registry()
    
    if strategy_name:
        strategy = registry.get(strategy_name)
        if not strategy:
            raise ValueError(f"Strategy '{strategy_name}' not found")
    else:
        strategy = registry.get_best_strategy(error_context)
        if not strategy:
            return ErrorRecoveryResult(
                success=False,
                action_taken=RecoveryAction.ABORT,
                recovery_time_ms=0.0,
                metadata={"reason": "no_applicable_strategy"}
            )
    
    return await strategy.execute_recovery(error_context, original_func, *args, **kwargs)


def get_recovery_stats() -> Dict[str, StrategyStats]:
    """Get statistics for all recovery strategies."""
    registry = get_global_strategy_registry()
    return registry.get_all_stats()