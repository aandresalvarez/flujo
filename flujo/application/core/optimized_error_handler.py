"""
Optimized error handler with caching and fast recovery.

This module provides error handling with caching, fast recovery, recovery strategy
registration, automatic error classification, and circuit breaker integration
for robust error management with minimal performance impact.
"""

import asyncio
import time
import traceback
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Callable, Union, Type, Tuple
from threading import RLock
from enum import Enum
import hashlib

from .optimized_telemetry import get_global_telemetry, MetricType
from .optimization.memory.memory_utils import track_object_creation


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    NETWORK = "network"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    EXTERNAL_SERVICE = "external_service"
    CONFIGURATION = "configuration"
    DATA_CORRUPTION = "data_corruption"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class RecoveryAction(Enum):
    """Types of recovery actions."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    ESCALATE = "escalate"
    IGNORE = "ignore"
    ABORT = "abort"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    
    error: Exception
    error_type: str
    error_message: str
    error_hash: str
    
    # Context information
    step_name: Optional[str] = None
    execution_id: Optional[str] = None
    attempt_number: int = 1
    
    # Timing information
    timestamp: float = field(default_factory=time.time)
    execution_duration_ms: float = 0.0
    
    # Classification
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    category: ErrorCategory = ErrorCategory.UNKNOWN
    
    # Recovery information
    is_recoverable: bool = True
    recovery_suggestions: List[RecoveryAction] = field(default_factory=list)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_exception(
        cls, 
        error: Exception, 
        step_name: Optional[str] = None,
        execution_id: Optional[str] = None,
        attempt_number: int = 1
    ) -> 'ErrorContext':
        """Create ErrorContext from exception."""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Generate error hash for caching
        error_content = f"{error_type}:{error_message}"
        error_hash = hashlib.md5(error_content.encode()).hexdigest()
        
        return cls(
            error=error,
            error_type=error_type,
            error_message=error_message,
            error_hash=error_hash,
            step_name=step_name,
            execution_id=execution_id,
            attempt_number=attempt_number
        )


@dataclass
class RecoveryStrategy:
    """Recovery strategy for specific error types."""
    
    name: str
    error_types: Set[Type[Exception]]
    error_patterns: List[str] = field(default_factory=list)
    
    # Recovery configuration
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    exponential_backoff: bool = True
    backoff_multiplier: float = 2.0
    max_delay_seconds: float = 60.0
    
    # Recovery actions
    primary_action: RecoveryAction = RecoveryAction.RETRY
    fallback_actions: List[RecoveryAction] = field(default_factory=list)
    
    # Conditions
    applies_to_categories: Set[ErrorCategory] = field(default_factory=set)
    applies_to_severities: Set[ErrorSeverity] = field(default_factory=set)
    
    # Custom recovery function
    custom_recovery_func: Optional[Callable[[ErrorContext], Any]] = None
    
    def matches_error(self, error_context: ErrorContext) -> bool:
        """Check if strategy matches the error context."""
        # Check error type
        if self.error_types and type(error_context.error) not in self.error_types:
            return False
        
        # Check error patterns
        if self.error_patterns:
            message_lower = error_context.error_message.lower()
            if not any(pattern.lower() in message_lower for pattern in self.error_patterns):
                return False
        
        # Check category
        if self.applies_to_categories and error_context.category not in self.applies_to_categories:
            return False
        
        # Check severity
        if self.applies_to_severities and error_context.severity not in self.applies_to_severities:
            return False
        
        return True
    
    def calculate_retry_delay(self, attempt_number: int) -> float:
        """Calculate retry delay for given attempt number."""
        if not self.exponential_backoff:
            return self.retry_delay_seconds
        
        delay = self.retry_delay_seconds * (self.backoff_multiplier ** (attempt_number - 1))
        return min(delay, self.max_delay_seconds)


@dataclass
class ErrorRecoveryResult:
    """Result of error recovery attempt."""
    
    success: bool
    action_taken: RecoveryAction
    recovery_time_ms: float
    
    # Recovery details
    attempts_made: int = 1
    strategy_used: Optional[str] = None
    
    # Result data
    recovered_value: Any = None
    new_error: Optional[Exception] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorStats:
    """Statistics for error handling."""
    
    total_errors: int = 0
    recovered_errors: int = 0
    unrecovered_errors: int = 0
    
    # Error type breakdown
    error_type_counts: Dict[str, int] = field(default_factory=dict)
    error_category_counts: Dict[ErrorCategory, int] = field(default_factory=dict)
    error_severity_counts: Dict[ErrorSeverity, int] = field(default_factory=dict)
    
    # Recovery statistics
    recovery_action_counts: Dict[RecoveryAction, int] = field(default_factory=dict)
    strategy_usage_counts: Dict[str, int] = field(default_factory=dict)
    
    # Performance statistics
    total_recovery_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    
    @property
    def recovery_rate(self) -> float:
        """Calculate error recovery rate."""
        if self.total_errors == 0:
            return 0.0
        return self.recovered_errors / self.total_errors
    
    @property
    def average_recovery_time_ms(self) -> float:
        """Calculate average recovery time."""
        if self.recovered_errors == 0:
            return 0.0
        return self.total_recovery_time_ms / self.recovered_errors
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.cache_hits + self.cache_misses
        return self.cache_hits / total_requests if total_requests > 0 else 0.0


class ErrorClassifier:
    """Classifies errors into categories and severity levels."""
    
    def __init__(self):
        # Error type mappings
        self._category_mappings = {
            # Network errors
            'ConnectionError': ErrorCategory.NETWORK,
            'TimeoutError': ErrorCategory.TIMEOUT,
            'ConnectTimeout': ErrorCategory.TIMEOUT,
            'ReadTimeout': ErrorCategory.TIMEOUT,
            'HTTPError': ErrorCategory.NETWORK,
            'URLError': ErrorCategory.NETWORK,
            
            # Authentication/Authorization
            'AuthenticationError': ErrorCategory.AUTHENTICATION,
            'PermissionError': ErrorCategory.AUTHORIZATION,
            'Forbidden': ErrorCategory.AUTHORIZATION,
            'Unauthorized': ErrorCategory.AUTHENTICATION,
            
            # Validation
            'ValidationError': ErrorCategory.VALIDATION,
            'ValueError': ErrorCategory.VALIDATION,
            'TypeError': ErrorCategory.VALIDATION,
            
            # Resource exhaustion
            'MemoryError': ErrorCategory.RESOURCE_EXHAUSTION,
            'OSError': ErrorCategory.SYSTEM,
            'IOError': ErrorCategory.SYSTEM,
            
            # Configuration
            'ConfigurationError': ErrorCategory.CONFIGURATION,
            'KeyError': ErrorCategory.CONFIGURATION,
            'AttributeError': ErrorCategory.CONFIGURATION,
        }
        
        self._severity_mappings = {
            # Critical errors
            'MemoryError': ErrorSeverity.CRITICAL,
            'SystemExit': ErrorSeverity.CRITICAL,
            'KeyboardInterrupt': ErrorSeverity.CRITICAL,
            
            # High severity
            'AuthenticationError': ErrorSeverity.HIGH,
            'PermissionError': ErrorSeverity.HIGH,
            'SecurityError': ErrorSeverity.HIGH,
            
            # Medium severity (default)
            'ConnectionError': ErrorSeverity.MEDIUM,
            'TimeoutError': ErrorSeverity.MEDIUM,
            'ValidationError': ErrorSeverity.MEDIUM,
            
            # Low severity
            'Warning': ErrorSeverity.LOW,
            'UserWarning': ErrorSeverity.LOW,
        }
        
        # Pattern-based classification
        self._category_patterns = {
            ErrorCategory.NETWORK: [
                'connection', 'network', 'socket', 'dns', 'ssl', 'tls'
            ],
            ErrorCategory.TIMEOUT: [
                'timeout', 'deadline', 'expired', 'time limit'
            ],
            ErrorCategory.AUTHENTICATION: [
                'auth', 'login', 'credential', 'token', 'unauthorized'
            ],
            ErrorCategory.AUTHORIZATION: [
                'permission', 'access', 'forbidden', 'privilege'
            ],
            ErrorCategory.VALIDATION: [
                'invalid', 'malformed', 'format', 'schema', 'validation'
            ],
            ErrorCategory.RESOURCE_EXHAUSTION: [
                'memory', 'disk', 'quota', 'limit', 'exhausted', 'full'
            ],
            ErrorCategory.EXTERNAL_SERVICE: [
                'service', 'api', 'endpoint', 'server', 'upstream'
            ],
            ErrorCategory.CONFIGURATION: [
                'config', 'setting', 'parameter', 'missing', 'not found'
            ]
        }
    
    def classify_error(self, error_context: ErrorContext) -> None:
        """Classify error context with category and severity."""
        # Classify category
        error_context.category = self._classify_category(error_context)
        
        # Classify severity
        error_context.severity = self._classify_severity(error_context)
        
        # Determine recoverability
        error_context.is_recoverable = self._is_recoverable(error_context)
        
        # Generate recovery suggestions
        error_context.recovery_suggestions = self._suggest_recovery_actions(error_context)
    
    def _classify_category(self, error_context: ErrorContext) -> ErrorCategory:
        """Classify error category."""
        # Check direct type mapping
        if error_context.error_type in self._category_mappings:
            return self._category_mappings[error_context.error_type]
        
        # Check pattern matching
        message_lower = error_context.error_message.lower()
        
        for category, patterns in self._category_patterns.items():
            if any(pattern in message_lower for pattern in patterns):
                return category
        
        return ErrorCategory.UNKNOWN
    
    def _classify_severity(self, error_context: ErrorContext) -> ErrorSeverity:
        """Classify error severity."""
        # Check direct type mapping
        if error_context.error_type in self._severity_mappings:
            return self._severity_mappings[error_context.error_type]
        
        # Pattern-based severity classification
        message_lower = error_context.error_message.lower()
        
        if any(word in message_lower for word in ['critical', 'fatal', 'emergency']):
            return ErrorSeverity.CRITICAL
        elif any(word in message_lower for word in ['error', 'failed', 'exception']):
            return ErrorSeverity.HIGH
        elif any(word in message_lower for word in ['warning', 'deprecated']):
            return ErrorSeverity.LOW
        
        return ErrorSeverity.MEDIUM
    
    def _is_recoverable(self, error_context: ErrorContext) -> bool:
        """Determine if error is recoverable."""
        # Critical errors are generally not recoverable
        if error_context.severity == ErrorSeverity.CRITICAL:
            return False
        
        # Some categories are generally recoverable
        recoverable_categories = {
            ErrorCategory.NETWORK,
            ErrorCategory.TIMEOUT,
            ErrorCategory.EXTERNAL_SERVICE
        }
        
        if error_context.category in recoverable_categories:
            return True
        
        # Some error types are not recoverable
        non_recoverable_types = {
            'MemoryError',
            'SystemExit',
            'KeyboardInterrupt',
            'SyntaxError',
            'ImportError'
        }
        
        return error_context.error_type not in non_recoverable_types
    
    def _suggest_recovery_actions(self, error_context: ErrorContext) -> List[RecoveryAction]:
        """Suggest recovery actions based on error classification."""
        suggestions = []
        
        if not error_context.is_recoverable:
            suggestions.append(RecoveryAction.ABORT)
            return suggestions
        
        # Category-based suggestions
        if error_context.category in {ErrorCategory.NETWORK, ErrorCategory.TIMEOUT}:
            suggestions.extend([RecoveryAction.RETRY, RecoveryAction.CIRCUIT_BREAK])
        elif error_context.category == ErrorCategory.EXTERNAL_SERVICE:
            suggestions.extend([RecoveryAction.RETRY, RecoveryAction.FALLBACK])
        elif error_context.category == ErrorCategory.VALIDATION:
            suggestions.extend([RecoveryAction.FALLBACK, RecoveryAction.ESCALATE])
        elif error_context.category == ErrorCategory.RESOURCE_EXHAUSTION:
            suggestions.extend([RecoveryAction.CIRCUIT_BREAK, RecoveryAction.ESCALATE])
        else:
            suggestions.append(RecoveryAction.RETRY)
        
        # Severity-based adjustments
        if error_context.severity == ErrorSeverity.HIGH:
            if RecoveryAction.ESCALATE not in suggestions:
                suggestions.append(RecoveryAction.ESCALATE)
        elif error_context.severity == ErrorSeverity.LOW:
            if RecoveryAction.IGNORE not in suggestions:
                suggestions.append(RecoveryAction.IGNORE)
        
        return suggestions


class OptimizedErrorHandler:
    """
    Optimized error handler with caching and fast recovery.
    
    Features:
    - Error classification and categorization
    - Recovery strategy registration and matching
    - Error context caching for fast recovery
    - Performance monitoring and statistics
    - Circuit breaker integration
    - Automatic error analysis and suggestions
    """
    
    def __init__(
        self,
        enable_caching: bool = True,
        cache_size: int = 1000,
        enable_stats: bool = True,
        enable_telemetry: bool = True
    ):
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        self.enable_stats = enable_stats
        self.enable_telemetry = enable_telemetry
        
        # Core components
        self._classifier = ErrorClassifier()
        self._telemetry = get_global_telemetry() if enable_telemetry else None
        
        # Recovery strategies
        self._strategies: List[RecoveryStrategy] = []
        self._strategy_cache: Dict[str, RecoveryStrategy] = {}
        
        # Error caching
        self._error_cache: Dict[str, ErrorContext] = {}
        self._recovery_cache: Dict[str, ErrorRecoveryResult] = {}
        self._weak_refs: Set[weakref.ref] = set()
        
        # Statistics
        self._stats = ErrorStats() if enable_stats else None
        
        # Thread safety
        self._lock = RLock()
        
        # Default strategies
        self._register_default_strategies()
    
    def register_strategy(self, strategy: RecoveryStrategy) -> None:
        """Register a recovery strategy."""
        with self._lock:
            self._strategies.append(strategy)
            # Clear strategy cache to force re-evaluation
            self._strategy_cache.clear()
    
    def unregister_strategy(self, strategy_name: str) -> None:
        """Unregister a recovery strategy."""
        with self._lock:
            self._strategies = [s for s in self._strategies if s.name != strategy_name]
            self._strategy_cache.clear()
    
    async def handle_error(
        self,
        error: Exception,
        step_name: Optional[str] = None,
        execution_id: Optional[str] = None,
        attempt_number: int = 1,
        **kwargs: Any
    ) -> ErrorRecoveryResult:
        """
        Handle error with optimized recovery.
        
        Args:
            error: Exception to handle
            step_name: Name of step where error occurred
            execution_id: Unique execution identifier
            attempt_number: Current attempt number
            **kwargs: Additional context
            
        Returns:
            ErrorRecoveryResult with recovery outcome
        """
        start_time = time.perf_counter()
        
        try:
            # Create error context
            error_context = ErrorContext.from_exception(
                error, step_name, execution_id, attempt_number
            )
            
            # Add additional metadata
            error_context.metadata.update(kwargs)
            
            # Check cache first
            if self.enable_caching:
                cached_result = self._get_cached_recovery(error_context)
                if cached_result:
                    if self.enable_stats:
                        self._stats.cache_hits += 1
                    
                    if self._telemetry:
                        self._telemetry.increment_counter(
                            "error_handler.cache_hits",
                            tags={"error_type": error_context.error_type}
                        )
                    
                    return cached_result
                
                if self.enable_stats:
                    self._stats.cache_misses += 1
            
            # Classify error
            self._classifier.classify_error(error_context)
            
            # Find matching strategy
            strategy = self._find_matching_strategy(error_context)
            
            # Execute recovery
            recovery_result = await self._execute_recovery(error_context, strategy)
            
            # Cache result
            if self.enable_caching and recovery_result.success:
                self._cache_recovery_result(error_context, recovery_result)
            
            # Update statistics
            if self.enable_stats:
                self._update_stats(error_context, recovery_result)
            
            # Record telemetry
            if self._telemetry:
                self._record_telemetry(error_context, recovery_result)
            
            return recovery_result
            
        except Exception as recovery_error:
            # Recovery itself failed
            recovery_time = (time.perf_counter() - start_time) * 1000
            
            result = ErrorRecoveryResult(
                success=False,
                action_taken=RecoveryAction.ABORT,
                recovery_time_ms=recovery_time,
                new_error=recovery_error
            )
            
            if self.enable_stats:
                self._stats.unrecovered_errors += 1
            
            return result
    
    def _get_cached_recovery(self, error_context: ErrorContext) -> Optional[ErrorRecoveryResult]:
        """Get cached recovery result."""
        with self._lock:
            return self._recovery_cache.get(error_context.error_hash)
    
    def _cache_recovery_result(self, error_context: ErrorContext, result: ErrorRecoveryResult) -> None:
        """Cache recovery result."""
        with self._lock:
            # Remove oldest entries if cache is full
            if len(self._recovery_cache) >= self.cache_size:
                # Simple FIFO eviction
                oldest_key = next(iter(self._recovery_cache))
                del self._recovery_cache[oldest_key]
            
            self._recovery_cache[error_context.error_hash] = result
    
    def _find_matching_strategy(self, error_context: ErrorContext) -> Optional[RecoveryStrategy]:
        """Find matching recovery strategy."""
        # Check strategy cache first
        cache_key = f"{error_context.error_type}:{error_context.category.value}"
        
        with self._lock:
            if cache_key in self._strategy_cache:
                return self._strategy_cache[cache_key]
            
            # Find matching strategy
            for strategy in self._strategies:
                if strategy.matches_error(error_context):
                    self._strategy_cache[cache_key] = strategy
                    return strategy
            
            # No matching strategy found
            self._strategy_cache[cache_key] = None
            return None
    
    async def _execute_recovery(
        self, 
        error_context: ErrorContext, 
        strategy: Optional[RecoveryStrategy]
    ) -> ErrorRecoveryResult:
        """Execute recovery strategy."""
        start_time = time.perf_counter()
        
        if not strategy:
            # No strategy available, use default behavior
            return ErrorRecoveryResult(
                success=False,
                action_taken=RecoveryAction.ABORT,
                recovery_time_ms=(time.perf_counter() - start_time) * 1000,
                strategy_used="none"
            )
        
        # Execute custom recovery function if available
        if strategy.custom_recovery_func:
            try:
                recovered_value = await self._execute_custom_recovery(
                    strategy.custom_recovery_func, 
                    error_context
                )
                
                return ErrorRecoveryResult(
                    success=True,
                    action_taken=strategy.primary_action,
                    recovery_time_ms=(time.perf_counter() - start_time) * 1000,
                    strategy_used=strategy.name,
                    recovered_value=recovered_value
                )
                
            except Exception as recovery_error:
                return ErrorRecoveryResult(
                    success=False,
                    action_taken=strategy.primary_action,
                    recovery_time_ms=(time.perf_counter() - start_time) * 1000,
                    strategy_used=strategy.name,
                    new_error=recovery_error
                )
        
        # Execute standard recovery actions
        return await self._execute_standard_recovery(error_context, strategy, start_time)
    
    async def _execute_custom_recovery(
        self, 
        recovery_func: Callable[[ErrorContext], Any], 
        error_context: ErrorContext
    ) -> Any:
        """Execute custom recovery function."""
        if asyncio.iscoroutinefunction(recovery_func):
            return await recovery_func(error_context)
        else:
            return recovery_func(error_context)
    
    async def _execute_standard_recovery(
        self, 
        error_context: ErrorContext, 
        strategy: RecoveryStrategy,
        start_time: float
    ) -> ErrorRecoveryResult:
        """Execute standard recovery actions."""
        action = strategy.primary_action
        
        if action == RecoveryAction.RETRY:
            # Calculate retry delay
            delay = strategy.calculate_retry_delay(error_context.attempt_number)
            
            if delay > 0:
                await asyncio.sleep(delay)
            
            return ErrorRecoveryResult(
                success=True,  # Assume retry will be handled by caller
                action_taken=RecoveryAction.RETRY,
                recovery_time_ms=(time.perf_counter() - start_time) * 1000,
                strategy_used=strategy.name,
                metadata={"retry_delay_seconds": delay}
            )
        
        elif action == RecoveryAction.FALLBACK:
            return ErrorRecoveryResult(
                success=True,  # Assume fallback will be handled by caller
                action_taken=RecoveryAction.FALLBACK,
                recovery_time_ms=(time.perf_counter() - start_time) * 1000,
                strategy_used=strategy.name
            )
        
        elif action == RecoveryAction.CIRCUIT_BREAK:
            return ErrorRecoveryResult(
                success=True,  # Circuit breaker will handle the error
                action_taken=RecoveryAction.CIRCUIT_BREAK,
                recovery_time_ms=(time.perf_counter() - start_time) * 1000,
                strategy_used=strategy.name
            )
        
        else:
            # Other actions (ESCALATE, IGNORE, ABORT)
            return ErrorRecoveryResult(
                success=action != RecoveryAction.ABORT,
                action_taken=action,
                recovery_time_ms=(time.perf_counter() - start_time) * 1000,
                strategy_used=strategy.name
            )
    
    def _update_stats(self, error_context: ErrorContext, result: ErrorRecoveryResult) -> None:
        """Update error handling statistics."""
        if not self.enable_stats or not self._stats:
            return
        
        with self._lock:
            self._stats.total_errors += 1
            
            if result.success:
                self._stats.recovered_errors += 1
            else:
                self._stats.unrecovered_errors += 1
            
            # Update breakdowns
            self._stats.error_type_counts[error_context.error_type] = \
                self._stats.error_type_counts.get(error_context.error_type, 0) + 1
            
            self._stats.error_category_counts[error_context.category] = \
                self._stats.error_category_counts.get(error_context.category, 0) + 1
            
            self._stats.error_severity_counts[error_context.severity] = \
                self._stats.error_severity_counts.get(error_context.severity, 0) + 1
            
            self._stats.recovery_action_counts[result.action_taken] = \
                self._stats.recovery_action_counts.get(result.action_taken, 0) + 1
            
            if result.strategy_used:
                self._stats.strategy_usage_counts[result.strategy_used] = \
                    self._stats.strategy_usage_counts.get(result.strategy_used, 0) + 1
            
            self._stats.total_recovery_time_ms += result.recovery_time_ms
    
    def _record_telemetry(self, error_context: ErrorContext, result: ErrorRecoveryResult) -> None:
        """Record telemetry for error handling."""
        if not self._telemetry:
            return
        
        # Record error occurrence
        self._telemetry.increment_counter(
            "error_handler.errors_total",
            tags={
                "error_type": error_context.error_type,
                "category": error_context.category.value,
                "severity": error_context.severity.value,
                "step_name": error_context.step_name or "unknown"
            }
        )
        
        # Record recovery result
        self._telemetry.increment_counter(
            "error_handler.recovery_attempts",
            tags={
                "action": result.action_taken.value,
                "success": str(result.success).lower(),
                "strategy": result.strategy_used or "none"
            }
        )
        
        # Record recovery time
        self._telemetry.record_histogram(
            "error_handler.recovery_time_ms",
            result.recovery_time_ms,
            tags={
                "action": result.action_taken.value,
                "success": str(result.success).lower()
            }
        )
    
    def _register_default_strategies(self) -> None:
        """Register default recovery strategies."""
        # Network error strategy
        network_strategy = RecoveryStrategy(
            name="network_errors",
            error_types={ConnectionError, TimeoutError},
            error_patterns=["connection", "network", "timeout"],
            max_retries=3,
            retry_delay_seconds=1.0,
            exponential_backoff=True,
            primary_action=RecoveryAction.RETRY,
            fallback_actions=[RecoveryAction.CIRCUIT_BREAK],
            applies_to_categories={ErrorCategory.NETWORK, ErrorCategory.TIMEOUT}
        )
        self.register_strategy(network_strategy)
        
        # Validation error strategy
        validation_strategy = RecoveryStrategy(
            name="validation_errors",
            error_types={ValueError, TypeError},
            max_retries=1,
            primary_action=RecoveryAction.FALLBACK,
            applies_to_categories={ErrorCategory.VALIDATION}
        )
        self.register_strategy(validation_strategy)
        
        # Resource exhaustion strategy
        resource_strategy = RecoveryStrategy(
            name="resource_exhaustion",
            error_types={MemoryError, OSError},
            max_retries=0,
            primary_action=RecoveryAction.CIRCUIT_BREAK,
            applies_to_categories={ErrorCategory.RESOURCE_EXHAUSTION}
        )
        self.register_strategy(resource_strategy)
    
    def get_stats(self) -> Optional[ErrorStats]:
        """Get error handling statistics."""
        return self._stats
    
    def reset_stats(self) -> None:
        """Reset error handling statistics."""
        if self.enable_stats and self._stats:
            with self._lock:
                self._stats = ErrorStats()
    
    def clear_cache(self) -> None:
        """Clear error and recovery caches."""
        with self._lock:
            self._error_cache.clear()
            self._recovery_cache.clear()
            self._strategy_cache.clear()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information."""
        with self._lock:
            return {
                "error_cache_size": len(self._error_cache),
                "recovery_cache_size": len(self._recovery_cache),
                "strategy_cache_size": len(self._strategy_cache),
                "cache_hit_rate": self._stats.cache_hit_rate if self._stats else 0.0,
                "max_cache_size": self.cache_size
            }
    
    def analyze_error_patterns(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Analyze error patterns over time window."""
        if not self.enable_stats or not self._stats:
            return {}
        
        with self._lock:
            analysis = {
                "total_errors": self._stats.total_errors,
                "recovery_rate": self._stats.recovery_rate,
                "average_recovery_time_ms": self._stats.average_recovery_time_ms,
                "top_error_types": dict(
                    sorted(
                        self._stats.error_type_counts.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:10]
                ),
                "category_distribution": {
                    category.value: count
                    for category, count in self._stats.error_category_counts.items()
                },
                "severity_distribution": {
                    severity.value: count
                    for severity, count in self._stats.error_severity_counts.items()
                },
                "recovery_action_distribution": {
                    action.value: count
                    for action, count in self._stats.recovery_action_counts.items()
                },
                "strategy_effectiveness": {
                    strategy: count
                    for strategy, count in self._stats.strategy_usage_counts.items()
                }
            }
        
        return analysis
    
    def suggest_optimizations(self) -> List[Dict[str, Any]]:
        """Suggest optimizations based on error patterns."""
        suggestions = []
        
        if not self.enable_stats or not self._stats:
            return suggestions
        
        with self._lock:
            # Analyze cache performance
            if self._stats.cache_hit_rate < 0.5:
                suggestions.append({
                    "type": "cache_optimization",
                    "priority": "medium",
                    "description": f"Cache hit rate is low ({self._stats.cache_hit_rate:.2%}). Consider increasing cache size or improving error classification.",
                    "current_value": self._stats.cache_hit_rate,
                    "recommended_action": "increase_cache_size"
                })
            
            # Analyze recovery rate
            if self._stats.recovery_rate < 0.7:
                suggestions.append({
                    "type": "recovery_optimization",
                    "priority": "high",
                    "description": f"Recovery rate is low ({self._stats.recovery_rate:.2%}). Consider adding more recovery strategies.",
                    "current_value": self._stats.recovery_rate,
                    "recommended_action": "add_recovery_strategies"
                })
            
            # Analyze recovery time
            if self._stats.average_recovery_time_ms > 1000:
                suggestions.append({
                    "type": "performance_optimization",
                    "priority": "medium",
                    "description": f"Average recovery time is high ({self._stats.average_recovery_time_ms:.1f}ms). Consider optimizing recovery strategies.",
                    "current_value": self._stats.average_recovery_time_ms,
                    "recommended_action": "optimize_recovery_strategies"
                })
            
            # Analyze error patterns
            if self._stats.error_type_counts:
                most_common_error = max(
                    self._stats.error_type_counts.items(),
                    key=lambda x: x[1]
                )
                
                if most_common_error[1] > self._stats.total_errors * 0.3:
                    suggestions.append({
                        "type": "error_prevention",
                        "priority": "high",
                        "description": f"Error type '{most_common_error[0]}' accounts for {most_common_error[1]/self._stats.total_errors:.1%} of all errors. Consider addressing root cause.",
                        "error_type": most_common_error[0],
                        "count": most_common_error[1],
                        "recommended_action": "investigate_root_cause"
                    })
        
        return suggestions


# Global error handler instance
_global_error_handler: Optional[OptimizedErrorHandler] = None


def get_global_error_handler() -> OptimizedErrorHandler:
    """Get the global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = OptimizedErrorHandler()
    return _global_error_handler


# Convenience functions
async def handle_error(
    error: Exception,
    step_name: Optional[str] = None,
    execution_id: Optional[str] = None,
    attempt_number: int = 1,
    **kwargs: Any
) -> ErrorRecoveryResult:
    """Convenience function to handle error."""
    handler = get_global_error_handler()
    return await handler.handle_error(
        error, step_name, execution_id, attempt_number, **kwargs
    )


def register_recovery_strategy(strategy: RecoveryStrategy) -> None:
    """Convenience function to register recovery strategy."""
    handler = get_global_error_handler()
    handler.register_strategy(strategy)


def get_error_stats() -> Optional[ErrorStats]:
    """Convenience function to get error statistics."""
    handler = get_global_error_handler()
    return handler.get_stats()


def analyze_error_patterns(time_window_hours: int = 24) -> Dict[str, Any]:
    """Convenience function to analyze error patterns."""
    handler = get_global_error_handler()
    return handler.analyze_error_patterns(time_window_hours)


# Decorator for automatic error handling
def with_error_handling(
    step_name: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    fallback_value: Any = None
):
    """Decorator for automatic error handling."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            handler = get_global_error_handler()
            
            for attempt in range(max_retries + 1):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                        
                except Exception as error:
                    if attempt == max_retries:
                        # Last attempt, handle error and potentially return fallback
                        result = await handler.handle_error(
                            error,
                            step_name=step_name or func.__name__,
                            attempt_number=attempt + 1
                        )
                        
                        if result.success and result.action_taken == RecoveryAction.FALLBACK:
                            return fallback_value
                        else:
                            raise error
                    else:
                        # Not last attempt, handle error and potentially retry
                        result = await handler.handle_error(
                            error,
                            step_name=step_name or func.__name__,
                            attempt_number=attempt + 1
                        )
                        
                        if result.action_taken == RecoveryAction.RETRY:
                            if retry_delay > 0:
                                await asyncio.sleep(retry_delay)
                            continue
                        elif result.action_taken == RecoveryAction.FALLBACK:
                            return fallback_value
                        else:
                            raise error
        
        def sync_wrapper(*args, **kwargs):
            # For synchronous functions, create a simple retry loop
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as error:
                    if attempt == max_retries:
                        if fallback_value is not None:
                            return fallback_value
                        else:
                            raise error
                    else:
                        if retry_delay > 0:
                            time.sleep(retry_delay)
                        continue
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
    
    def clear_cache(self) -> None:
        """Clear error handling caches."""
        with self._lock:
            self._error_cache.clear()
            self._recovery_cache.clear()
            self._strategy_cache.clear()
            self._weak_refs.clear()
    
    def get_strategies(self) -> List[RecoveryStrategy]:
        """Get registered recovery strategies."""
        with self._lock:
            return list(self._strategies)


# Global optimized error handler instance
_global_error_handler: Optional[OptimizedErrorHandler] = None


def get_global_error_handler() -> OptimizedErrorHandler:
    """Get the global optimized error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = OptimizedErrorHandler()
    return _global_error_handler


# Convenience functions
async def handle_error_optimized(
    error: Exception,
    step_name: Optional[str] = None,
    execution_id: Optional[str] = None,
    attempt_number: int = 1,
    **kwargs: Any
) -> ErrorRecoveryResult:
    """Convenience function to handle error with optimization."""
    handler = get_global_error_handler()
    return await handler.handle_error(error, step_name, execution_id, attempt_number, **kwargs)


def register_recovery_strategy(strategy: RecoveryStrategy) -> None:
    """Convenience function to register recovery strategy."""
    handler = get_global_error_handler()
    handler.register_strategy(strategy)


def get_error_stats() -> Optional[ErrorStats]:
    """Convenience function to get error statistics."""
    handler = get_global_error_handler()
    return handler.get_stats()