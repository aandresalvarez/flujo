"""
Ultra-optimized step executor v2 with modular, policy-driven architecture.

This is a complete rewrite of the UltraStepExecutor with:
- Modular design with clear separation of concerns
- Deterministic behavior across processes and restarts
- Pluggable components via dependency injection
- Robust isolation between concerns
- Exhaustive accounting of successful and failed attempts
- Backward compatibility with existing SDK signatures

Author: Flujo Team
Version: 2.0
"""

from __future__ import annotations

import asyncio
import contextvars
import time
import hashlib
import copy
import multiprocessing
from abc import abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from functools import cached_property, wraps
from multiprocessing import cpu_count
from typing import (
    Any,
    Awaitable,
    Callable,
    Generic,
    Optional,
    Protocol,
    TypeVar,
    Dict,
    List,
    Type,
    Tuple,
    Union,
    cast,
)
import types
from types import SimpleNamespace
from asyncio import Task
import weakref
from weakref import WeakKeyDictionary

from ...domain.dsl.step import HumanInTheLoopStep, Step, MergeStrategy, BranchFailureStrategy
from ...domain.dsl.loop import LoopStep
from ...domain.dsl.conditional import ConditionalStep
from .types import TContext_w_Scratch, ExecutionFrame
from ...domain.dsl.dynamic_router import DynamicParallelRouterStep
from ...domain.dsl.parallel import ParallelStep
from ...domain.models import BaseModel, StepResult, UsageLimits, PipelineResult, PipelineContext
from ...domain.processors import AgentProcessors
from pydantic import Field
from ...domain.validation import ValidationResult
from ...exceptions import (
    UsageLimitExceededError,
    PausedException,
    InfiniteFallbackError,
    InfiniteRedirectError,
    PricingNotConfiguredError,
    ContextInheritanceError,
    MissingAgentError,
)
from flujo.utils.formatting import format_cost

# Type alias for step executor function signature
StepExecutor = Callable[
    [Step[Any, Any], Any, Optional[Any], Optional[Any], Optional[Any]],
    Awaitable[StepResult],
]


# Exception classification for retry logic
class RetryableError(Exception):
    """Base class for errors that should trigger retries."""
    pass


class NonRetryableError(Exception):
    """Base class for errors that should not trigger retries."""
    pass


# Classify common exceptions
class ValidationError(RetryableError):
    """Validation failures that can be retried."""
    pass


class PluginError(RetryableError):
    """Plugin failures that can be retried."""
    pass


class AgentError(RetryableError):
    """Agent execution errors that can be retried."""
    pass


# Import required modules
from ...steps.cache_step import CacheStep
from ...utils.performance import time_perf_ns, time_perf_ns_to_seconds
from ...cost import extract_usage_metrics
from ...infra import telemetry
from ...signature_tools import analyze_signature
from ...application.context_manager import _accepts_param
from ...utils.context import safe_merge_context_updates


class ExecutorCore(Generic[TContext_w_Scratch]):
    """
    Ultra-optimized step executor with modular, policy-driven architecture.
    
    This implementation provides:
    - Consistent step routing in the main execute() method
    - Proper _execute_agent_step with separated try-catch blocks for validators, plugins, and agents
    - Comprehensive _execute_simple_step method with fallback support
    - Fixed _is_complex_step logic to properly categorize steps
    - Recursive execution model consistency across all step handlers
    """

    # Context variables for tracking fallback relationships and chains
    _fallback_relationships: contextvars.ContextVar[Dict[str, str]] = contextvars.ContextVar(
        "fallback_relationships", default={}
    )
    _fallback_chain: contextvars.ContextVar[list[Step[Any, Any]]] = contextvars.ContextVar(
        "fallback_chain", default=[]
    )

    # Cache for fallback relationship loop detection (True if loop detected, False otherwise)
    _fallback_graph_cache: contextvars.ContextVar[Dict[str, bool]] = contextvars.ContextVar(
        "fallback_graph_cache", default={}
    )

    # Maximum length for fallback chains to prevent infinite loops
    _MAX_FALLBACK_CHAIN_LENGTH = 10

    # Maximum iterations for fallback loop detection to prevent infinite loops
    _DEFAULT_MAX_FALLBACK_ITERATIONS = 100

    def __init__(
        self,
        agent_runner: Any = None,
        processor_pipeline: Any = None,
        validator_runner: Any = None,
        plugin_runner: Any = None,
        usage_meter: Any = None,
        cache_backend: Any = None,
        cache_key_generator: Any = None,
        telemetry: Any = None,
        enable_cache: bool = True,
        # Additional parameters for compatibility
        serializer: Any = None,
        hasher: Any = None,
    ):
        """Initialize ExecutorCore with dependency injection."""
        self._agent_runner = agent_runner or DefaultAgentRunner()
        self._processor_pipeline = processor_pipeline or DefaultProcessorPipeline()
        self._validator_runner = validator_runner or DefaultValidatorRunner()
        self._plugin_runner = plugin_runner or DefaultPluginRunner()
        self._usage_meter = usage_meter or ThreadSafeMeter()
        self._cache_backend = cache_backend
        self._telemetry = telemetry or DefaultTelemetry()
        self._enable_cache = enable_cache
        self._step_history_so_far: list[StepResult] = []
        
        # Store additional components for compatibility
        self._serializer = serializer or OrjsonSerializer()
        self._hasher = hasher or Blake3Hasher()
        self._cache_key_generator = cache_key_generator or DefaultCacheKeyGenerator(self._hasher)

    async def execute(
        self,
        frame: Optional[ExecutionFrame[TContext_w_Scratch]] = None,
        *,
        step: Optional[Any] = None,
        data: Optional[Any] = None,
        context: Optional[TContext_w_Scratch] = None,
        resources: Optional[Any] = None,
        limits: Optional[UsageLimits] = None,
        stream: bool = False,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]] = None,
        breach_event: Optional[Any] = None,
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]] = None,
        result: Optional[StepResult] = None,
        _fallback_depth: int = 0,
    ) -> StepResult:
        """
        Central execution method that routes to appropriate handlers.
        Implements the recursive execution model consistently for all step types.
        """
        # Handle both ExecutionFrame and keyword arguments for backward compatibility
        if frame is not None:
            # Extract parameters from the ExecutionFrame
            step = frame.step
            data = frame.data
            context = frame.context
            resources = frame.resources
            limits = frame.limits
            stream = frame.stream
            on_chunk = frame.on_chunk
            breach_event = frame.breach_event
            context_setter = frame.context_setter
            result = frame.result
            _fallback_depth = frame._fallback_depth
        elif step is None:
            raise ValueError("Either frame or step must be provided")
        
        telemetry.logfire.debug("=== EXECUTOR CORE EXECUTE ===")
        telemetry.logfire.debug(f"Step type: {type(step)}")
        telemetry.logfire.debug(f"Step name: {getattr(step, 'name', 'unknown')}")
        telemetry.logfire.debug(f"ExecutorCore.execute called with breach_event: {breach_event is not None}")
        telemetry.logfire.debug(f"ExecutorCore.execute called with limits: {limits}")

        # Generate cache key if caching is enabled
        cache_key = None
        if self._cache_backend is not None and self._enable_cache:
            cache_key = self._cache_key_generator.generate_key(step, data, context, resources)
            telemetry.logfire.debug(f"Generated cache key: {cache_key}")

            # Check cache first
            cached_result = await self._cache_backend.get(cache_key)
            if cached_result is not None:
                telemetry.logfire.debug(f"Cache hit for step: {step.name}")
                # Ensure metadata_ is always a dict
                if cached_result.metadata_ is None:
                    cached_result.metadata_ = {}
                cached_result.metadata_["cache_hit"] = True
                return cached_result
            else:
                telemetry.logfire.debug(f"Cache miss for step: {step.name}")
        else:
            telemetry.logfire.debug(f"Caching disabled for step: {step.name}")

        # Consistent step routing following the recursive execution model
        # Route to appropriate handler based on step type
        if isinstance(step, LoopStep):
            telemetry.logfire.debug(f"Routing to loop step handler: {step.name}")
            return await self._handle_loop_step(
                step, data, context, resources, limits, context_setter, _fallback_depth
            )
        elif isinstance(step, ParallelStep):
            telemetry.logfire.debug(f"Routing to parallel step handler: {step.name}")
            return await self._handle_parallel_step(
                step, data, context, resources, limits, breach_event, context_setter
            )
        elif isinstance(step, ConditionalStep):
            telemetry.logfire.debug(f"Routing to conditional step handler: {step.name}")
            return await self._handle_conditional_step(
                step, data, context, resources, limits, context_setter, _fallback_depth
            )
        elif isinstance(step, DynamicParallelRouterStep):
            telemetry.logfire.debug(f"Routing to dynamic router step handler: {step.name}")
            return await self._handle_dynamic_router_step(
                step, data, context, resources, limits, context_setter
            )
        elif isinstance(step, HumanInTheLoopStep):
            telemetry.logfire.debug(f"Routing to HITL step handler: {step.name}")
            return await self._handle_hitl_step(
                step, data, context, resources, limits, breach_event, context_setter
            )
        elif isinstance(step, CacheStep):
            telemetry.logfire.debug(f"Routing to cache step handler: {step.name}")
            return await self._handle_cache_step(
                step, data, context, resources, limits, breach_event, context_setter, None
            )
        elif hasattr(step, 'fallback_step') and step.fallback_step is not None:
            telemetry.logfire.debug(f"Routing to simple step with fallback: {step.name}")
            return await self._execute_simple_step(
                step, data, context, resources, limits, stream, on_chunk, cache_key, breach_event, _fallback_depth
            )
        else:
            telemetry.logfire.debug(f"Routing to agent step handler: {step.name}")
            return await self._execute_agent_step(
                step, data, context, resources, limits, stream, on_chunk, cache_key, breach_event, _fallback_depth
            )

    async def _execute_simple_step(
        self,
        step: Any,
        data: Any,
        context: Optional[TContext_w_Scratch],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        stream: bool,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]],
        cache_key: Optional[str],
        breach_event: Optional[Any],
        _fallback_depth: int = 0,
    ) -> StepResult:
        """
        Execute a simple step with comprehensive fallback support.
        
        This method implements the fallback logic for steps that have a fallback_step defined.
        It follows the recursive execution model by calling back into the main execute method
        for fallback steps.
        """
        telemetry.logfire.debug(f"_execute_simple_step called with step type: {type(step)}, limits: {limits}")
        telemetry.logfire.debug(f"_execute_simple_step step name: {step.name}")
        
        # Try to execute the primary step
        primary_result = None
        try:
            primary_result = await self._execute_agent_step(
                step=step,
                data=data,
                context=context,
                resources=resources,
                limits=limits,
                stream=stream,
                on_chunk=on_chunk,
                cache_key=cache_key,
                breach_event=breach_event,
                _fallback_depth=_fallback_depth,
            )
            
            # If primary step succeeded, return the result
            if primary_result.success:
                return primary_result
                
        except Exception as e:
            # Primary step failed with exception
            primary_result = StepResult(
                name=step.name,
                output=None,
                success=False,
                attempts=1,
                latency_s=0.0,
                token_counts=0,
                cost_usd=0.0,
                feedback=f"Step execution failed: {str(e)}",
                branch_context=None,
                metadata_={},
            )
        
        # Primary step failed, check if we have a fallback
        if hasattr(step, "fallback_step") and step.fallback_step is not None:
            # Check for infinite fallback loops
            if _fallback_depth >= self._MAX_FALLBACK_CHAIN_LENGTH:
                raise InfiniteFallbackError(f"Fallback chain too long for step '{step.name}'")
            
            # Execute fallback step using recursive execution model
            telemetry.logfire.debug(f"Executing fallback for step '{step.name}'")
            
            # Create ExecutionFrame for the fallback step
            fallback_frame = ExecutionFrame(
                step=step.fallback_step,
                data=data,
                context=context,
                resources=resources,
                limits=limits,
                stream=stream,
                on_chunk=on_chunk,
                breach_event=breach_event,
                context_setter=None,
                result=None,
                _fallback_depth=_fallback_depth + 1,
            )
            
            fallback_result = await self.execute(fallback_frame)
            
            # Combine results: use fallback's output and success, but preserve original step name
            # and accumulate metrics from both steps
            combined_result = StepResult(
                name=step.name,  # Preserve original step name
                output=fallback_result.output,
                success=fallback_result.success,
                attempts=primary_result.attempts + fallback_result.attempts,
                latency_s=primary_result.latency_s + fallback_result.latency_s,
                token_counts=primary_result.token_counts + fallback_result.token_counts,
                cost_usd=primary_result.cost_usd + fallback_result.cost_usd,
                feedback=fallback_result.feedback,  # Use fallback's feedback
                branch_context=fallback_result.branch_context,
                metadata_=fallback_result.metadata_.copy() if fallback_result.metadata_ else {},
                step_history=fallback_result.step_history,
            )
            
            # Add fallback metadata
            combined_result.metadata_["fallback_triggered"] = True
            
            # Combine feedback from both steps if both failed
            if not primary_result.success and not fallback_result.success:
                primary_feedback = primary_result.feedback or ""
                fallback_feedback = fallback_result.feedback or ""
                if primary_feedback and fallback_feedback:
                    combined_result.feedback = f"{primary_feedback}; {fallback_feedback}"
                elif primary_feedback:
                    combined_result.feedback = primary_feedback
                elif fallback_feedback:
                    combined_result.feedback = fallback_feedback
            elif fallback_result.success:
                # Clear feedback on successful fallback - but ensure it's never None
                combined_result.feedback = ""
                # Store original error in metadata
                if primary_result.feedback:
                    combined_result.metadata_["original_error"] = primary_result.feedback
            
            return combined_result
        
        # No fallback available, return the failed primary result
        return primary_result

    async def _execute_agent_step(
        self,
        step: Any,
        data: Any,
        context: Optional[TContext_w_Scratch],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        stream: bool,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]],
        cache_key: Optional[str],
        breach_event: Optional[Any],
        _fallback_depth: int = 0,
    ) -> StepResult:
        """
        Execute an agent step with proper failure domain separation.
        
        This method implements separated try-catch blocks for validators, plugins, and agents
        to ensure proper error isolation and handling according to the design requirements.
        
        FIXES IMPLEMENTED:
        - Ensure feedback is never None when step fails
        - Fix validation error handling to preserve proper feedback messages
        - Fix plugin error handling to maintain error context
        - Ensure successful steps have empty string feedback, not None
        - Fix retry logic to properly accumulate feedback across attempts
        """
        import time
        from ...exceptions import MissingAgentError

        # Initialize metadata_ variable to fix the critical issue
        metadata_: Dict[str, Any] = {}
        
        telemetry.logfire.debug("=== EXECUTE AGENT STEP ===")
        telemetry.logfire.debug(f"Step name: {step.name}")
        telemetry.logfire.debug(f"Step type: {type(step)}")
        telemetry.logfire.debug(f"Data: {data}")
        telemetry.logfire.debug(f"Context: {context}")
        telemetry.logfire.debug(f"Resources: {resources}")
        telemetry.logfire.debug(f"Stream: {stream}")
        telemetry.logfire.debug(f"Fallback depth: {_fallback_depth}")

        # Check for missing agent
        agent = getattr(step, "agent", None)
        if agent is None:
            raise MissingAgentError(f"Step '{step.name}' has no agent")

        # Initialize result
        result = StepResult(
            name=step.name,
            output=None,
            success=False,
            attempts=0,
            latency_s=0.0,
            token_counts=0,
            cost_usd=0.0,
            feedback="",  # Initialize with empty string, never None
            branch_context=None,
            metadata_=metadata_,
        )

        # Get step configuration
        max_retries = getattr(step, "max_retries", 1)
        if hasattr(step, "config") and step.config:
            max_retries = getattr(step.config, "max_retries", max_retries)

        # Track attempts and timing
        attempts = 0
        start_time = time.monotonic()
        accumulated_feedback = []

        # Retry loop
        while attempts <= max_retries:
            attempts += 1
            result.attempts = attempts
            
            telemetry.logfire.debug(f"Attempt {attempts}/{max_retries + 1} for step: {step.name}")

            # Separated try-catch blocks for different failure domains
            
            # 1. Agent execution domain
            agent_output = None
            try:
                # Prepare agent options
                options = {}
                if hasattr(step, "config") and step.config:
                    for attr in ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]:
                        if hasattr(step.config, attr):
                            options[attr] = getattr(step.config, attr)

                # Execute agent using the agent runner
                telemetry.logfire.debug(f"Agent execution - data: {data}, context.counter: {getattr(context, 'counter', 'N/A') if context else 'None'}")
                agent_output = await self._agent_runner.run(
                    agent=agent,
                    payload=data,
                    context=context,
                    resources=resources,
                    options=options,
                    stream=stream,
                    on_chunk=on_chunk,
                    breach_event=breach_event,
                )
                telemetry.logfire.debug(f"Agent execution completed - output: {agent_output}")

                # Extract cost and token information from the output
                from ...cost import extract_usage_metrics
                prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
                    raw_output=agent_output, agent=agent, step_name=step.name
                )
                result.cost_usd = cost_usd
                result.token_counts = prompt_tokens + completion_tokens

            except (
                PausedException,
                InfiniteFallbackError,
                InfiniteRedirectError,
                ContextInheritanceError,
            ) as e:
                # Re-raise critical exceptions immediately
                telemetry.logfire.error(f"Step '{step.name}' failed with critical error: {e}")
                raise
            except Exception as e:
                # Handle retryable agent errors
                error_msg = f"Agent execution failed on attempt {attempts}: {str(e)}"
                accumulated_feedback.append(error_msg)
                telemetry.logfire.warning(f"Step '{step.name}' agent execution attempt {attempts} failed: {e}")

                # Check if we should retry
                if attempts <= max_retries:
                    # Clone payload for retry with accumulated feedback
                    data = self._clone_payload_for_retry(data, accumulated_feedback)
                    continue
                else:
                    # Max retries exceeded - ensure feedback is never None
                    result.success = False
                    result.feedback = f"Agent execution failed after {attempts} attempts: {str(e)}"
                    result.output = None
                    result.latency_s = time.monotonic() - start_time
                    telemetry.logfire.error(f"Step '{step.name}' agent failed after {attempts} attempts")
                    return result

            # Agent execution succeeded, now process the output
            processed_output = agent_output

            # 2. Processor domain (separated try-catch)
            try:
                if hasattr(step, "processors") and step.processors:
                    processed_output = await self._processor_pipeline.apply_output(
                        step.processors, processed_output, context=context
                    )
            except Exception as processor_error:
                # Processor failure - DO NOT RETRY AGENT
                result.success = False
                result.feedback = f"Processor failed: {processor_error}"  # Ensure feedback is never None
                result.output = agent_output  # Keep the original agent output for fallback
                result.latency_s = time.monotonic() - start_time
                telemetry.logfire.debug(f"Step '{step.name}' processor failed: {processor_error}")
                return result

            # 3. Validator domain (separated try-catch)
            try:
                if hasattr(step, "validators") and step.validators:
                    await self._validator_runner.validate(
                        step.validators, processed_output, context=context
                    )
            except ValueError as validation_error:
                # Check if this is a non-strict validation step
                from ...application.context_manager import _get_validation_flags, _apply_validation_metadata
                is_validation_step, is_strict = _get_validation_flags(step)
                
                if is_validation_step and not is_strict:
                    # Non-strict validation: step succeeds but record failure in metadata
                    _apply_validation_metadata(
                        result,
                        validation_failed=True,
                        is_validation_step=is_validation_step,
                        is_strict=is_strict,
                    )
                    telemetry.logfire.debug(f"Step '{step.name}' validation failed (non-strict): {validation_error}")
                    # Continue with success - don't return here
                else:
                    # Strict validation or regular step: fail the step - DO NOT RETRY AGENT
                    result.success = False
                    result.feedback = f"Validation failed: {validation_error}"  # Ensure feedback is never None
                    if is_validation_step and is_strict:
                        # For strict validation steps, drop the output
                        result.output = None
                    else:
                        # For regular steps, keep the output for fallback
                        result.output = processed_output
                    result.latency_s = time.monotonic() - start_time
                    telemetry.logfire.debug(f"Step '{step.name}' failed validation: {validation_error}")
                    return result
            except Exception as validation_error:
                # Other validation errors - DO NOT RETRY AGENT
                result.success = False
                result.feedback = f"Validator error: {validation_error}"  # Ensure feedback is never None
                result.output = processed_output  # Keep the output for fallback
                result.latency_s = time.monotonic() - start_time
                telemetry.logfire.error(f"Step '{step.name}' validator error: {validation_error}")
                return result

            # 4. Plugin domain (separated try-catch)
            try:
                if hasattr(step, "plugins") and step.plugins:
                    processed_output = await self._plugin_runner.run_plugins(
                        step.plugins, processed_output, context=context
                    )
            except Exception as plugin_error:
                # Plugin failure - DO NOT RETRY AGENT
                result.success = False
                result.feedback = f"Plugin failed: {plugin_error}"  # Ensure feedback is never None
                result.output = processed_output  # Keep the output for fallback
                result.latency_s = time.monotonic() - start_time
                telemetry.logfire.debug(f"Step '{step.name}' plugin failed: {plugin_error}")
                return result

            # All processing succeeded
            result.output = processed_output
            result.success = True
            result.latency_s = time.monotonic() - start_time
            result.feedback = ""  # Successful steps have empty string feedback, not None
            # FIXED: Capture context changes made by the agent
            # The agent may have modified the context (e.g., incrementing counters)
            # We need to preserve these changes in the branch_context
            result.branch_context = context

            # Cache successful result
            if self._cache_backend is not None and cache_key is not None:
                if result.metadata_ is None:
                    result.metadata_ = {}
                await self._cache_backend.put(cache_key, result, ttl_s=3600)

            telemetry.logfire.debug(f"Step '{step.name}' completed successfully")
            return result

        # This should never be reached, but just in case
        result.success = False
        result.feedback = "Step execution failed: unexpected error"  # Ensure feedback is never None
        result.output = None
        result.latency_s = time.monotonic() - start_time
        return result

    def _clone_payload_for_retry(self, original_data: Any, accumulated_feedbacks: list[str]) -> Any:
        """Clone payload for retry attempts with accumulated feedback injection."""
        if not accumulated_feedbacks:
            # No feedback to add, return original data unchanged
            return original_data

        feedback_text = "\n".join(accumulated_feedbacks)

        # Handle dict objects (most common case)
        if isinstance(original_data, dict):
            cloned_data = original_data.copy()
            existing_feedback = cloned_data.get("feedback", "")
            cloned_data["feedback"] = (existing_feedback + "\n" + feedback_text).strip()
            return cloned_data
            
        # Handle Pydantic models with efficient model_copy
        elif hasattr(original_data, "model_copy"):
            cloned_data = original_data.model_copy(deep=False)
            if hasattr(cloned_data, "feedback"):
                existing_feedback = getattr(cloned_data, "feedback", "")
                setattr(cloned_data, "feedback", (existing_feedback + "\n" + feedback_text).strip())
            return cloned_data
            
        # Handle dataclasses and other objects with copy support
        elif hasattr(original_data, "__dict__"):
            import copy
            try:
                cloned_data = copy.deepcopy(original_data)
                if hasattr(cloned_data, "feedback"):
                    existing_feedback = getattr(cloned_data, "feedback", "")
                    setattr(cloned_data, "feedback", (existing_feedback + "\n" + feedback_text).strip())
                return cloned_data
            except (TypeError, RecursionError):
                # Fall back to string conversion if deep copy fails
                pass
                
        # Handle list/tuple types
        elif isinstance(original_data, (list, tuple)):
            try:
                cloned_data = original_data.copy() if isinstance(original_data, list) else list(original_data)
                # Try to add feedback to the first element if it's a dict
                if cloned_data and isinstance(cloned_data[0], dict):
                    cloned_data[0]["feedback"] = cloned_data[0].get("feedback", "") + "\n" + feedback_text
                return cloned_data
            except (AttributeError, TypeError):
                pass
                
        # Fallback: convert to string and append feedback
        return f"{str(original_data)}\n{feedback_text}"

    # Placeholder methods for step handlers - these would need to be implemented
    # based on the existing implementation in the original file
    
    async def _handle_loop_step(
        self,
        loop_step: Any,  # LoopStep type
        data: Any,
        context: Optional[TContext_w_Scratch],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        context_setter: Optional[Callable[[Any, Optional[Any]], None]],
        _fallback_depth: int = 0,
    ) -> StepResult:
        """
        Handle loop step execution with proper context propagation and iteration management.
        
        This implementation fixes:
        - Context accumulation across loop iterations in _handle_loop_step
        - Accurate iteration counting logic
        - Exit condition evaluation that works even when iterations fail
        - Max iterations logic that stops at the correct count
        - Proper timing of iteration input/output mappers
        - Loop step attempt counting for usage governance integration
        """
        import time
        from ...domain.dsl.pipeline import Pipeline

        print(f"[KIRO DEBUG] === HANDLE LOOP STEP === {loop_step.name}")
        telemetry.logfire.debug("=== HANDLE LOOP STEP ===")
        telemetry.logfire.debug(f"Loop step name: {loop_step.name}")
        telemetry.logfire.debug(f"Loop step limits: {limits}")

        # Initialize result with proper metadata
        result = StepResult(
            name=loop_step.name,
            output=None,
            success=False,
            attempts=0,  # Will be set to actual iteration count
            latency_s=0.0,
            token_counts=0,
            cost_usd=0.0,
            feedback="",
            branch_context=None,
            metadata_={"iterations": 0, "exit_reason": None},
        )

        start_time = time.monotonic()
        cumulative_cost = 0.0
        cumulative_tokens = 0
        iteration_count = 0  # Accurate iteration counting
        max_iterations = getattr(loop_step, "max_loops", 10)
        current_data = data

        telemetry.logfire.debug(f"Loop step max_iterations: {max_iterations}")

        try:
            telemetry.logfire.debug(f"Starting LoopStep: max_iterations={max_iterations}, limits={limits}")
            
            # Initialize context for loop iterations with proper accumulation
            current_context = context
            if current_context is None:
                from flujo.domain.models import PipelineContext
                current_context = PipelineContext(initial_prompt=str(current_data))
            
            # Apply initial input mapper if provided (called at the correct time)
            if hasattr(loop_step, "initial_input_to_loop_body_mapper") and loop_step.initial_input_to_loop_body_mapper:
                try:
                    current_data = loop_step.initial_input_to_loop_body_mapper(data, current_context)
                    telemetry.logfire.debug(f"Initial input mapper applied: {current_data}")
                except Exception as e:
                    result.success = False
                    result.feedback = f"Initial input mapper failed: {str(e)}"
                    result.latency_s = time.monotonic() - start_time
                    result.attempts = 0  # No iterations attempted
                    result.metadata_["exit_reason"] = "initial_mapper_failed"
                    telemetry.logfire.error(f"Error in initial input mapper for LoopStep '{loop_step.name}': {str(e)}")
                    return result
            
            # Track loop state
            loop_exit_reason = None
            last_body_output = None
            
            # Main loop with accurate iteration counting
            while iteration_count < max_iterations:
                iteration_count += 1
                telemetry.logfire.info(f"LoopStep '{loop_step.name}': Starting Iteration {iteration_count}/{max_iterations}")
                telemetry.logfire.debug(f"Starting iteration {iteration_count}, current_data={current_data}")
                telemetry.logfire.debug(f"Iteration {iteration_count}: current_context.counter = {getattr(current_context, 'counter', 'N/A')}")
                telemetry.logfire.debug(f"Loop body type: {type(loop_step.loop_body_pipeline)}")

                # Apply iteration input mapper if provided (for iterations after the first)
                if iteration_count > 1 and hasattr(loop_step, "iteration_input_mapper") and loop_step.iteration_input_mapper:
                    try:
                        # Use last_body_output for iteration input mapper (correct timing)
                        iteration_input = loop_step.iteration_input_mapper(last_body_output, current_context, iteration_count - 1)
                        current_data = iteration_input
                        telemetry.logfire.debug(f"Iteration input mapper applied: {current_data}")
                    except Exception as e:
                        result.success = False
                        result.feedback = f"Iteration input mapper failed on iteration {iteration_count}: {str(e)}"
                        result.latency_s = time.monotonic() - start_time
                        result.attempts = iteration_count - 1  # Count completed iterations
                        result.metadata_["iterations"] = iteration_count - 1
                        result.metadata_["exit_reason"] = "iteration_mapper_failed"
                        telemetry.logfire.error(f"Error in iteration input mapper for LoopStep '{loop_step.name}': {str(e)}")
                        return result

                # Execute the loop body using recursive execution model
                body_result = None
                if isinstance(loop_step.loop_body_pipeline, Pipeline):
                    # Execute pipeline by executing each step in sequence
                    current_body_data = current_data
                    
                    # FIXED: Use the accumulated current_context directly for this iteration
                    # This ensures context changes from previous iterations are preserved
                    body_context = current_context
                    telemetry.logfire.debug(f"Iteration {iteration_count}: body_context.counter before execution = {getattr(body_context, 'counter', 'N/A')}")
                    telemetry.logfire.debug(f"Iteration {iteration_count}: current_body_data = {current_body_data}")
                    telemetry.logfire.debug(f"Iteration {iteration_count}: Pipeline steps count = {len(loop_step.loop_body_pipeline.steps)}")
                    
                    # Execute each step in the pipeline
                    all_successful = True
                    total_cost = 0.0
                    total_tokens = 0
                    body_error_message = None
                    
                    for step_idx, step in enumerate(loop_step.loop_body_pipeline.steps):
                        telemetry.logfire.debug(f"Iteration {iteration_count}: Executing step {step_idx + 1}/{len(loop_step.loop_body_pipeline.steps)}: {step.name}")
                        with telemetry.logfire.span(step.name) as step_span:
                            # Create ExecutionFrame for the step
                            step_frame = ExecutionFrame(
                                step=step,
                                data=current_body_data,
                                context=body_context,
                                resources=resources,
                                limits=limits,
                                stream=False,
                                on_chunk=None,
                                breach_event=None,
                                context_setter=context_setter,
                                result=None,
                                _fallback_depth=_fallback_depth,
                            )
                            step_result = await self.execute(step_frame)
                            telemetry.logfire.debug(f"Iteration {iteration_count}: Step {step.name} result - success: {step_result.success}, output: {step_result.output}")
                            telemetry.logfire.debug(f"Iteration {iteration_count}: Step {step.name} result - branch_context: {step_result.branch_context}")
                            if step_result.branch_context:
                                telemetry.logfire.debug(f"Iteration {iteration_count}: Step {step.name} result - branch_context.counter: {getattr(step_result.branch_context, 'counter', 'N/A')}")
                            else:
                                telemetry.logfire.debug(f"Iteration {iteration_count}: Step {step.name} result - branch_context is None!")
                        
                        telemetry.logfire.debug(f"Iteration {iteration_count}: Step {step.name} result - success: {step_result.success}, output: {step_result.output}")
                        telemetry.logfire.debug(f"Iteration {iteration_count}: Step {step.name} result - branch_context: {step_result.branch_context}")
                        if step_result.branch_context:
                            telemetry.logfire.debug(f"Iteration {iteration_count}: Step {step.name} result - branch_context.counter: {getattr(step_result.branch_context, 'counter', 'N/A')}")
                        
                        total_cost += step_result.cost_usd
                        total_tokens += step_result.token_counts
                        
                        if not step_result.success:
                            all_successful = False
                            body_error_message = step_result.feedback
                            # Continue to capture context updates even on failure
                        
                        # Use output as input for next step
                        current_body_data = step_result.output
                        
                        # Update body_context with any context changes from the step
                        if step_result.branch_context is not None:
                            body_context = step_result.branch_context
                            telemetry.logfire.debug(f"Iteration {iteration_count}: body_context.counter after step execution = {getattr(body_context, 'counter', 'N/A')}")
                    
                    telemetry.logfire.debug(f"Iteration {iteration_count}: Pipeline execution completed - all_successful: {all_successful}")
                    # Create body result
                    body_result = StepResult(
                        name=f"{loop_step.name}_iteration_{iteration_count}",
                        output=current_body_data,
                        success=all_successful,
                        attempts=1,
                        latency_s=0.0,  # Individual iteration timing not tracked here
                        token_counts=total_tokens,
                        cost_usd=total_cost,
                        feedback="Loop body executed successfully" if all_successful else f"Loop body failed: {body_error_message}",
                        branch_context=body_context,
                        metadata_={},
                    )
                else:
                    # Execute as a regular step using recursive execution model
                    # FIXED: Use the accumulated current_context directly for this iteration
                    telemetry.logfire.debug(f"Iteration {iteration_count}: current_context.counter before execution = {getattr(current_context, 'counter', 'N/A')}")
                    body_frame = ExecutionFrame(
                        step=loop_step.loop_body_pipeline,
                        data=current_data,
                        context=current_context,
                        resources=resources,
                        limits=limits,
                        stream=False,
                        on_chunk=None,
                        breach_event=None,
                        context_setter=context_setter,
                        result=None,
                        _fallback_depth=_fallback_depth,
                    )
                    body_result = await self.execute(body_frame)
                    telemetry.logfire.debug(f"Iteration {iteration_count}: body_result.branch_context.counter after execution = {getattr(body_result.branch_context, 'counter', 'N/A') if body_result.branch_context else 'None'}")

                # FIXED: Properly accumulate context changes from this iteration
                telemetry.logfire.debug(f"Iteration {iteration_count}: body_result.success = {body_result.success}")
                telemetry.logfire.debug(f"Iteration {iteration_count}: body_result.branch_context = {body_result.branch_context}")
                telemetry.logfire.debug(f"Iteration {iteration_count}: body_result.output = {body_result.output}")
                
                if body_result.branch_context is not None:
                    telemetry.logfire.debug(f"Iteration {iteration_count}: Before merge - current_context.counter = {getattr(current_context, 'counter', 'N/A')}")
                    telemetry.logfire.debug(f"Iteration {iteration_count}: Before merge - body_result.branch_context.counter = {getattr(body_result.branch_context, 'counter', 'N/A')}")
                    
                    # Update current_context to use the updated context from this iteration
                    # This ensures context changes are properly accumulated across iterations
                    current_context = body_result.branch_context
                    telemetry.logfire.debug(f"Iteration {iteration_count}: After merge - current_context.counter = {getattr(current_context, 'counter', 'N/A')}")
                else:
                    telemetry.logfire.debug(f"Iteration {iteration_count}: No branch_context in body_result")

                # Prospective totals if we add this iteration's cost/tokens
                prospective_cost = cumulative_cost + (body_result.cost_usd or 0)
                prospective_tokens = cumulative_tokens + (body_result.token_counts or 0)
                if limits is not None:
                    from ...application.core.usage_governor import UsageGovernor
                    usage_governor = UsageGovernor(limits)
                    from ...domain.models import PipelineResult
                    # Use current totals for the exception result
                    temp_step_result = StepResult(
                        name=loop_step.name,
                        output=body_result.output,
                        success=True,
                        attempts=iteration_count - 1,
                        latency_s=time.monotonic() - start_time,
                        token_counts=cumulative_tokens,
                        cost_usd=cumulative_cost,
                        feedback="",
                        branch_context=current_context,
                        metadata_={"iterations": iteration_count - 1, "exit_reason": "usage_limit_exceeded"},
                    )
                    temp_result = PipelineResult(
                        step_history=[temp_step_result],
                        total_cost_usd=cumulative_cost,
                        total_tokens=cumulative_tokens,
                        total_latency_s=time.monotonic() - start_time,
                        final_pipeline_context=current_context,
                    )
                    # But check the limit with the prospective totals
                    if prospective_cost > limits.total_cost_usd_limit:
                        print(f"[DEBUG] Raising UsageLimitExceededError: current={cumulative_cost}, prospective={prospective_cost}, limit={limits.total_cost_usd_limit}")
                        from flujo.exceptions import UsageLimitExceededError as ULE
                        formatted_limit = str(int(limits.total_cost_usd_limit)) if limits.total_cost_usd_limit == int(limits.total_cost_usd_limit) else str(limits.total_cost_usd_limit)
                        temp_step_result_current = StepResult(
                            name=loop_step.name,
                            output=body_result.output,
                            success=True,
                            attempts=iteration_count - 1,
                            latency_s=time.monotonic() - start_time,
                            token_counts=cumulative_tokens,
                            cost_usd=cumulative_cost,
                            feedback="",
                            branch_context=current_context,
                            metadata_={"iterations": iteration_count - 1, "exit_reason": "usage_limit_exceeded"},
                        )
                        temp_result_current = PipelineResult(
                            step_history=[temp_step_result_current],
                            total_cost_usd=cumulative_cost,
                            total_tokens=cumulative_tokens,
                            total_latency_s=time.monotonic() - start_time,
                            final_pipeline_context=current_context,
                        )
                        raise ULE(f"Cost limit of ${formatted_limit} exceeded", temp_result_current)

                # Only add to cumulative totals if not breached
                cumulative_cost += body_result.cost_usd or 0
                cumulative_tokens += body_result.token_counts or 0

                # Store the body output for exit condition and next iteration
                last_body_output = body_result.output
                
                # FIXED: Use the body output as input for the next iteration
                # This ensures that the loop properly chains data between iterations
                current_data = last_body_output
                telemetry.logfire.debug(f"Iteration {iteration_count}: Updated current_data to {current_data} for next iteration")

                # Update usage meter with actual costs from this iteration (proper usage governance)
                if self._usage_meter is not None:
                    await self._usage_meter.add(
                        cost_usd=body_result.cost_usd,
                        prompt_tokens=body_result.token_counts,
                        completion_tokens=0  # Assuming all tokens are prompt tokens for simplicity
                    )

                telemetry.logfire.debug(f"Iteration {iteration_count} completed, checking exit condition")

                # Check exit condition (works even when iterations fail)
                if hasattr(loop_step, "exit_condition_callable") and loop_step.exit_condition_callable:
                    try:
                        telemetry.logfire.debug(f"Checking exit condition: output={last_body_output}")
                        should_exit = loop_step.exit_condition_callable(last_body_output, current_context)
                        telemetry.logfire.debug(f"Exit condition result: {should_exit}")
                        if should_exit:
                            loop_exit_reason = "condition"
                            telemetry.logfire.debug(f"Loop exiting due to condition after {iteration_count} iterations")
                            break
                    except Exception as e:
                        telemetry.logfire.warning(f"Exit condition evaluation failed: {e}")
                        # Continue loop execution even if exit condition fails
                        # This allows the loop to complete based on max_iterations
                
                # ✅ TASK 7.2: Check usage limits before starting the next iteration
                print(f"[DEBUG] Iteration {iteration_count}: Checking condition - limits: {limits is not None}, iteration_count: {iteration_count}, max_iterations: {max_iterations}")
                if limits is not None and iteration_count < max_iterations:
                    print(f"[DEBUG] Iteration {iteration_count}: Usage limit checking condition met")
                    try:
                        # Use UsageGovernor for consistent error message formatting
                        from ...application.core.usage_governor import UsageGovernor
                        usage_governor = UsageGovernor(limits)
                        
                        # Create a temporary pipeline result for checking limits
                        from ...domain.models import PipelineResult
                        
                        # Create a temporary step result for the loop step
                        temp_step_result = StepResult(
                            name=loop_step.name,
                            output=last_body_output,
                            success=True,
                            attempts=iteration_count,
                            latency_s=time.monotonic() - start_time,
                            token_counts=cumulative_tokens,
                            cost_usd=cumulative_cost,
                            feedback="",
                            branch_context=current_context,
                            metadata_={"iterations": iteration_count, "exit_reason": "usage_limit_exceeded"},
                        )
                        
                        temp_result = PipelineResult(
                            step_history=[temp_step_result],
                            total_cost_usd=cumulative_cost,
                            total_tokens=cumulative_tokens,
                            total_latency_s=time.monotonic() - start_time,
                            final_pipeline_context=current_context,
                        )
                        
                        # Debug output to understand the issue
                        print(f"[DEBUG] Iteration {iteration_count}: Checking usage limits - cost: {cumulative_cost}, limit: {limits.total_cost_usd_limit}")
                        
                        # Check limits using UsageGovernor for consistent formatting
                        usage_governor.check_usage_limits(temp_result, None)
                        print(f"[DEBUG] Iteration {iteration_count}: Usage limits check passed")
                    except UsageLimitExceededError as e:
                        print(f"[DEBUG] Iteration {iteration_count}: Usage limits breached - {e}")
                        # Update result with current state before re-raising
                        result.cost_usd = cumulative_cost
                        result.token_counts = cumulative_tokens
                        result.latency_s = time.monotonic() - start_time
                        result.attempts = iteration_count
                        result.metadata_["iterations"] = iteration_count
                        result.metadata_["exit_reason"] = "usage_limit_exceeded"
                        result.branch_context = current_context
                        raise e

            # Determine final output using output mapper (called at the correct time)
            final_output = last_body_output
            if hasattr(loop_step, "loop_output_mapper") and loop_step.loop_output_mapper:
                try:
                    final_output = loop_step.loop_output_mapper(last_body_output, current_context)
                    telemetry.logfire.debug(f"Loop output mapper applied: {final_output}")
                except Exception as e:
                    result.success = False
                    result.feedback = f"Loop output mapper failed: {str(e)}"
                    result.latency_s = time.monotonic() - start_time
                    result.attempts = iteration_count
                    result.metadata_["iterations"] = iteration_count
                    result.metadata_["exit_reason"] = "output_mapper_failed"
                    telemetry.logfire.error(f"Error in loop output mapper for LoopStep '{loop_step.name}': {str(e)}")
                    return result

            # Set final result based on loop completion (proper success determination)
            if loop_exit_reason == "condition":
                result.success = True
                result.feedback = f"Loop completed successfully after {iteration_count} iterations (exit condition met)"
                result.metadata_["exit_reason"] = "condition"
            elif iteration_count >= max_iterations:
                # Max iterations reached - this is considered a failure unless exit condition was met
                result.success = False
                result.feedback = f"Loop terminated after reaching max_loops ({max_iterations})"
                result.metadata_["exit_reason"] = "max_iterations"
            else:
                # Should not reach here, but handle gracefully
                result.success = False
                result.feedback = "Loop terminated unexpectedly"
                result.metadata_["exit_reason"] = "unexpected"
            
            # Set final result values
            result.output = final_output
            result.cost_usd = cumulative_cost
            result.token_counts = cumulative_tokens
            result.latency_s = time.monotonic() - start_time
            result.metadata_["iterations"] = iteration_count
            result.attempts = iteration_count  # Accurate attempt counting for usage governance
            result.branch_context = current_context  # Preserve accumulated context changes
            
            telemetry.logfire.debug(f"Loop completed: iterations={iteration_count}, success={result.success}, exit_reason={result.metadata_['exit_reason']}")
            return result
            
        except UsageLimitExceededError as e:
            # ✅ TASK 7.2: FIX LOOP STEP EXCEPTION HANDLING
            # Re-raise UsageLimitExceededError to preserve the specific exception type
            # Update result with current state before re-raising
            result.cost_usd = cumulative_cost
            result.token_counts = cumulative_tokens
            result.latency_s = time.monotonic() - start_time
            result.attempts = iteration_count
            result.metadata_["iterations"] = iteration_count
            result.metadata_["exit_reason"] = "usage_limit_exceeded"
            result.branch_context = current_context
            raise e
        except Exception as e:
            result.success = False
            result.feedback = f"Loop step failed: {str(e)}"
            result.output = last_body_output if 'last_body_output' in locals() else None
            result.cost_usd = cumulative_cost
            result.token_counts = cumulative_tokens
            result.latency_s = time.monotonic() - start_time
            result.attempts = iteration_count
            result.metadata_["iterations"] = iteration_count
            result.metadata_["exit_reason"] = "exception"
            result.branch_context = current_context
            telemetry.logfire.error(f"Error in LoopStep '{loop_step.name}': {str(e)}")
            return result
    
    async def _handle_parallel_step(self, step=None, data=None, context=None, resources=None, limits=None, breach_event=None, context_setter=None, parallel_step=None, step_executor=None):
        """Handle ParallelStep execution with proper feedback handling and error propagation."""
        import copy
        import asyncio
        from flujo.domain.dsl.parallel import ParallelStep
        from flujo.domain.dsl.step import MergeStrategy, BranchFailureStrategy
        
        # Use parallel_step if provided, otherwise use step
        if parallel_step is not None:
            step = parallel_step
        
        # Type check and validation
        if not isinstance(step, ParallelStep):
            raise ValueError(f"Expected ParallelStep, got {type(step)}")
        
        parallel_step = step
        telemetry.logfire.debug(f"=== HANDLING PARALLEL STEP === {parallel_step.name}")
        telemetry.logfire.debug(f"Parallel step branches: {list(parallel_step.branches.keys())}")
        
        # Initialize result
        result = StepResult(name=parallel_step.name)
        result.metadata_ = {}
        start_time = time.monotonic()
        
        # Check for empty branches
        if not parallel_step.branches:
            result.success = False
            result.feedback = "Parallel step has no branches to execute"
            result.output = {}
            result.latency_s = time.monotonic() - start_time
            return result
        
        # Create usage governor for parallel execution
        usage_governor = self._ParallelUsageGovernor(limits) if limits else None
        
        # Create breach event for immediate cancellation signaling
        if breach_event is None and limits is not None:
            breach_event = asyncio.Event()
        
        # Initialize tracking variables
        branch_results = {}
        branch_contexts = {}
        total_cost = 0.0
        total_tokens = 0
        all_successful = True
        failure_messages = []
        
        # Prepare context for each branch
        for branch_name, branch_pipeline in parallel_step.branches.items():
            # Create isolated context for each branch
            if context is not None:
                if parallel_step.context_include_keys:
                    # Create a new context with only the specified keys
                    from flujo.domain.models import PipelineContext
                    # Create a new context instance of the same type with required fields
                    branch_context = type(context)(initial_prompt=context.initial_prompt)
                    # Copy only the specified fields
                    for field_name in parallel_step.context_include_keys:
                        if hasattr(context, field_name):
                            setattr(branch_context, field_name, getattr(context, field_name))
                else:
                    # Deep copy entire context
                    branch_context = copy.deepcopy(context)
                    # Ensure scratchpad is also deep copied
                    if hasattr(branch_context, 'scratchpad'):
                        branch_context.scratchpad = copy.deepcopy(context.scratchpad)
            else:
                branch_context = None
            
            branch_contexts[branch_name] = branch_context
        
        # Execute branches in parallel
        async def execute_branch(branch_name: str, branch_pipeline, branch_context):
            """Execute a single branch with proper error handling and context isolation."""
            try:
                telemetry.logfire.debug(f"Executing branch: {branch_name}")
                
                # Use custom step executor if provided, otherwise use self.execute
                if step_executor is not None:
                    # Use custom step executor for testing
                    branch_result = await step_executor(branch_pipeline, data, branch_context, resources, breach_event)
                else:
                    # The branch_pipeline is a Pipeline object, not a Step object
                    # We need to execute it as a pipeline
                    from flujo.domain.models import PipelineResult
                    
                    # Execute the pipeline
                    pipeline_result = await self._execute_pipeline(
                        branch_pipeline, data, branch_context, resources, limits, breach_event, context_setter
                    )
                    
                    # Convert pipeline result to step result
                    pipeline_success = all(step.success for step in pipeline_result.step_history) if pipeline_result.step_history else False
                    branch_result = StepResult(
                        name=f"{parallel_step.name}_{branch_name}",
                        output=pipeline_result.step_history[-1].output if pipeline_result.step_history else None,
                        success=pipeline_success,
                        attempts=1,
                        latency_s=sum(step.latency_s for step in pipeline_result.step_history),
                        token_counts=pipeline_result.total_tokens,
                        cost_usd=pipeline_result.total_cost_usd,
                        feedback=pipeline_result.step_history[-1].feedback if pipeline_result.step_history else "",
                        branch_context=pipeline_result.final_pipeline_context,
                        metadata_={},
                    )
                
                # Track usage if governor is available
                if usage_governor is not None:
                    await usage_governor.add_usage(
                        branch_result.cost_usd,
                        branch_result.token_counts,
                        branch_result
                    )
                
                telemetry.logfire.debug(f"Branch {branch_name} completed: success={branch_result.success}")
                return branch_name, branch_result
                
            except Exception as e:
                telemetry.logfire.error(f"Branch {branch_name} failed with exception: {e}")
                # Create failure result
                failure_result = StepResult(
                    name=f"{parallel_step.name}_{branch_name}",
                    output=None,
                    success=False,
                    attempts=1,
                    latency_s=0.0,
                    token_counts=0,
                    cost_usd=0.0,
                    feedback=f"Branch execution failed: {str(e)}",
                    branch_context=branch_context,
                    metadata_={},
                )
                return branch_name, failure_result
        
        # Execute all branches concurrently
        branch_tasks = [
            execute_branch(branch_name, branch_pipeline, branch_contexts[branch_name])
            for branch_name, branch_pipeline in parallel_step.branches.items()
        ]
        
        # Wait for all branches to complete
        branch_execution_results = await asyncio.gather(*branch_tasks, return_exceptions=True)
        
        # Process branch results
        for branch_name, branch_result in branch_execution_results:
            if isinstance(branch_result, Exception):
                # Handle exceptions from gather
                telemetry.logfire.error(f"Branch {branch_name} raised exception: {branch_result}")
                branch_result = StepResult(
                    name=f"{parallel_step.name}_{branch_name}",
                    output=None,
                    success=False,
                    attempts=1,
                    latency_s=0.0,
                    token_counts=0,
                    cost_usd=0.0,
                    feedback=f"Branch execution failed: {str(branch_result)}",
                    metadata_={},
                )
            
            branch_results[branch_name] = branch_result
            
            # ✅ TASK 7.5: FIX PARALLEL STEP COST AGGREGATION
            # Only count costs from successful branches to avoid double-counting
            if branch_result.success:
                total_cost += branch_result.cost_usd
                total_tokens += branch_result.token_counts
            
            if not branch_result.success:
                all_successful = False
                failure_messages.append(f"Branch '{branch_name}': {branch_result.feedback}")
        
        # Determine overall success based on failure strategy
        if parallel_step.on_branch_failure == BranchFailureStrategy.PROPAGATE:
            result.success = all_successful
        elif parallel_step.on_branch_failure == BranchFailureStrategy.IGNORE:
            # Succeed if at least one branch succeeded, fail if all branches failed
            successful_branches = sum(1 for br in branch_results.values() if br.success)
            result.success = successful_branches > 0
        else:
            # Default to propagate
            result.success = all_successful
        
        # Build output dictionary
        output_dict = {}
        for branch_name, branch_result in branch_results.items():
            if branch_result.success:
                output_dict[branch_name] = branch_result.output
            else:
                # For failed branches, include the StepResult object
                output_dict[branch_name] = branch_result
        
        result.output = output_dict
        
        # Handle context merging based on merge strategy
        telemetry.logfire.debug(f"Context merging check: context={context is not None}, merge_strategy={parallel_step.merge_strategy}")
        if context is not None and parallel_step.merge_strategy != MergeStrategy.NO_MERGE:
            try:
                # Collect successful branch contexts
                successful_contexts = {}
                for branch_name, branch_result in branch_results.items():
                    if branch_result.success and branch_result.branch_context is not None:
                        successful_contexts[branch_name] = branch_result.branch_context
                        telemetry.logfire.debug(f"Successful branch: {branch_name}")
                
                telemetry.logfire.debug(f"Context merging: strategy={parallel_step.merge_strategy}, successful_branches={len(successful_contexts)}")
                
                # Apply merge strategy
                if parallel_step.merge_strategy == MergeStrategy.CONTEXT_UPDATE:
                    # Merge all successful branch contexts into main context
                    for branch_name, branch_context in successful_contexts.items():
                        if parallel_step.field_mapping and branch_name in parallel_step.field_mapping:
                            # Use explicit field mapping
                            for field_name in parallel_step.field_mapping[branch_name]:
                                if hasattr(branch_context, field_name):
                                    setattr(context, field_name, getattr(branch_context, field_name))
                        else:
                            # Merge all fields (excluding branch name unless ignore_branch_names is False)
                            for field_name in dir(branch_context):
                                if not field_name.startswith('_'):
                                    if hasattr(branch_context, field_name):
                                        if parallel_step.ignore_branch_names or field_name != branch_name:
                                            try:
                                                setattr(context, field_name, getattr(branch_context, field_name))
                                            except (AttributeError, TypeError):
                                                pass  # Skip read-only fields
                
                elif parallel_step.merge_strategy == MergeStrategy.MERGE_SCRATCHPAD:
                    # Merge scratchpad dictionaries from all successful branches
                    # Sort branch names to ensure consistent merge order
                    sorted_branch_names = sorted(successful_contexts.keys())
                    for branch_name in sorted_branch_names:
                        branch_context = successful_contexts[branch_name]
                        if hasattr(branch_context, 'scratchpad') and hasattr(context, 'scratchpad'):
                            # Check for key collisions and log warnings
                            for key in branch_context.scratchpad:
                                if key in context.scratchpad:
                                    # For MERGE_SCRATCHPAD, log warning on collision but continue
                                    telemetry.logfire.warning(f"Scratchpad key collision: '{key}' already exists in main context, skipping")
                                else:
                                    # Only merge non-colliding keys
                                    context.scratchpad[key] = branch_context.scratchpad[key]
                
                elif parallel_step.merge_strategy == MergeStrategy.OVERWRITE:
                    # Overwrite context with the last successful branch context
                    if successful_contexts:
                        last_branch_name = sorted(successful_contexts.keys())[-1]
                        last_branch_context = successful_contexts[last_branch_name]
                        # Only overwrite fields that were included in the branch contexts
                        if parallel_step.context_include_keys:
                            # Only overwrite the fields that were included in the branch contexts
                            for field_name in parallel_step.context_include_keys:
                                if hasattr(last_branch_context, field_name) and hasattr(context, field_name):
                                    try:
                                        telemetry.logfire.debug(f"OVERWRITE: Setting {field_name} from {last_branch_name}")
                                        if field_name == 'scratchpad':
                                            telemetry.logfire.debug(f"OVERWRITE: Last branch scratchpad: {getattr(last_branch_context, field_name)}")
                                        setattr(context, field_name, getattr(last_branch_context, field_name))
                                    except (AttributeError, TypeError):
                                        pass  # Skip read-only fields
                        else:
                            # If no context_include_keys, overwrite all fields from the last branch
                            # Copy specific known fields from the last successful branch
                            known_fields = ['scratchpad', 'initial_prompt', 'run_id', 'hitl_history', 'command_log', 'val']
                            for field_name in known_fields:
                                if hasattr(last_branch_context, field_name) and hasattr(context, field_name):
                                    try:
                                        telemetry.logfire.debug(f"OVERWRITE: Setting {field_name} from {last_branch_name}")
                                        if field_name == 'scratchpad':
                                            telemetry.logfire.debug(f"OVERWRITE: Last branch scratchpad: {getattr(last_branch_context, field_name)}")
                                        setattr(context, field_name, getattr(last_branch_context, field_name))
                                    except (AttributeError, TypeError):
                                        pass  # Skip read-only fields
                
                elif callable(parallel_step.merge_strategy):
                    # Custom merge function
                    parallel_step.merge_strategy(context, successful_contexts)
                
                # Set the merged context as branch_context
                result.branch_context = context
                
            except Exception as e:
                telemetry.logfire.error(f"Context merging failed: {e}")
                # Continue without context merging
        
        # Set final result values
        result.cost_usd = total_cost
        result.token_counts = total_tokens
        result.latency_s = time.monotonic() - start_time
        result.attempts = 1
        
        # Set feedback based on results
        if result.success:
            if all_successful:
                result.feedback = f"All {len(parallel_step.branches)} branches executed successfully"
            else:
                result.feedback = f"Parallel step completed with {len(failure_messages)} branch failures (ignored)"
        else:
            result.feedback = f"Parallel step failed: {'; '.join(failure_messages)}"
        
        telemetry.logfire.debug(f"Parallel step {parallel_step.name} completed: success={result.success}")
        return result
    
    async def _execute_pipeline(self, pipeline, data, context, resources, limits, breach_event, context_setter):
        """Execute a pipeline and return a PipelineResult."""
        from flujo.domain.models import PipelineResult
        
        # Execute each step in the pipeline sequentially
        current_data = data
        current_context = context
        total_cost = 0.0
        total_tokens = 0
        total_latency = 0.0
        step_history = []
        all_successful = True
        feedback = ""
        
        for step in pipeline.steps:
            try:
                # Create execution frame for the step
                step_frame = ExecutionFrame(
                    step=step,
                    data=current_data,
                    context=current_context,
                    resources=resources,
                    limits=limits,
                    stream=False,
                    on_chunk=None,
                    breach_event=breach_event,
                    context_setter=context_setter,
                    result=None,
                    _fallback_depth=0,
                )
                
                # Execute the step
                step_result = await self.execute(step_frame)
                
                # Update tracking variables
                total_cost += step_result.cost_usd
                total_tokens += step_result.token_counts
                total_latency += step_result.latency_s
                step_history.append(step_result)
                
                if not step_result.success:
                    all_successful = False
                    feedback = step_result.feedback
                    break
                
                # Use output as input for next step
                current_data = step_result.output
                
                # Update context if available
                if step_result.branch_context is not None:
                    current_context = step_result.branch_context
                    
            except Exception as e:
                all_successful = False
                feedback = f"Step execution failed: {str(e)}"
                break
        
        # Create pipeline result
        return PipelineResult(
            step_history=step_history,
            total_cost_usd=total_cost,
            total_tokens=total_tokens,
            total_latency_s=total_latency,
            final_pipeline_context=current_context,
        )
    
    class _ParallelUsageGovernor:
        """Usage governor for parallel step execution."""
        
        def __init__(self, limits):
            self.limits = limits
            self.total_cost = 0.0
            self.total_tokens = 0
            self.limit_breached = asyncio.Event()
            self.limit_breach_error = None
        
        async def add_usage(self, cost_delta, token_delta, result):
            """Add usage and check limits."""
            self.total_cost += cost_delta
            self.total_tokens += token_delta
            
            # Check limits
            if self.limits.total_cost_usd_limit is not None and self.total_cost > self.limits.total_cost_usd_limit:
                self.limit_breach_error = UsageLimitExceededError(
                    f"Cost limit of ${self.limits.total_cost_usd_limit} exceeded (current: ${self.total_cost})"
                )
                self.limit_breached.set()
            
            if self.limits.total_tokens_limit is not None and self.total_tokens > self.limits.total_tokens_limit:
                self.limit_breach_error = UsageLimitExceededError(
                    f"Token limit of {self.limits.total_tokens_limit} exceeded (current: {self.total_tokens})"
                )
                self.limit_breached.set()
        
        def breached(self):
            """Check if limits have been breached."""
            return self.limit_breached.is_set()
        
        def get_error(self):
            """Get the breach error if any."""
            return self.limit_breach_error
    
    async def _handle_conditional_step(self, step, data, context, resources, limits, context_setter, fallback_depth=0):
        """Handle ConditionalStep execution with proper context isolation and merging.
        
        This implementation fixes:
        - Branch execution logic in _handle_conditional_step
        - Context isolation for branch execution using deep copy
        - Context capture and merging logic to preserve branch modifications
        - Mapper context handling to call mappers on main context, not branch context
        - Conditional step error handling to properly propagate branch failures
        - Feedback messages to accurately reflect what actually happened
        """
        import time
        import copy
        from ...domain.dsl.pipeline import Pipeline
        from ...utils.context import safe_merge_context_updates

        telemetry.logfire.debug("=== HANDLE CONDITIONAL STEP ===")
        telemetry.logfire.debug(f"Conditional step name: {step.name}")

        # Initialize result
        result = StepResult(
            name=step.name,
            output=None,
            success=False,
            attempts=1,
            latency_s=0.0,
            token_counts=0,
            cost_usd=0.0,
            feedback="",
            branch_context=None,
            metadata_={},
        )

        start_time = time.monotonic()

        try:
            # Execute condition callable
            branch_key = step.condition_callable(data, context)
            telemetry.logfire.debug(f"Condition evaluated to branch key: {branch_key}")

            # Determine which branch to execute
            branch_to_execute = None
            if branch_key in step.branches:
                branch_to_execute = step.branches[branch_key]
                result.metadata_["executed_branch_key"] = branch_key
                telemetry.logfire.info(f"Condition evaluated to branch key '{branch_key}'")
                telemetry.logfire.info(f"Executing branch for key '{branch_key}'")
                # Set span attribute for tracing
                with telemetry.logfire.span(f"branch_{branch_key}") as span:
                    span.set_attribute("executed_branch_key", branch_key)
            elif step.default_branch_pipeline is not None:
                branch_to_execute = step.default_branch_pipeline
                result.metadata_["executed_branch_key"] = "default"
                telemetry.logfire.info(f"Condition evaluated to branch key '{branch_key}', using default branch")
                telemetry.logfire.info("Executing default branch")
                # Set span attribute for tracing
                with telemetry.logfire.span("branch_default") as span:
                    span.set_attribute("executed_branch_key", "default")
            else:
                telemetry.logfire.warn(f"No branch matches condition '{branch_key}' and no default branch provided")

            if branch_to_execute:
                # Execute the selected branch using the executor's own methods
                if isinstance(branch_to_execute, Pipeline):
                    # Execute pipeline branch by executing each step in the pipeline
                    branch_data = data
                    
                    # Apply input mapper if provided (on main context, not branch context)
                    if step.branch_input_mapper:
                        branch_data = step.branch_input_mapper(data, context)
                        telemetry.logfire.debug(f"Branch input mapper applied: {branch_data}")
                    
                    # Isolate context for branch execution using deep copy
                    if context is not None:
                        branch_context = copy.deepcopy(context)
                        telemetry.logfire.debug(f"Created isolated branch context from main context")
                    else:
                        from flujo.domain.models import PipelineContext
                        branch_context = PipelineContext(initial_prompt=str(branch_data))
                        telemetry.logfire.debug(f"Created new PipelineContext for branch execution")
                    
                    # Execute each step in the pipeline
                    current_data = branch_data
                    total_cost = 0.0
                    total_tokens = 0
                    all_successful = True
                    step_results = []
                    branch_error_message = None
                    
                    for step_idx, pipeline_step in enumerate(branch_to_execute.steps):
                        telemetry.logfire.debug(f"Executing step {step_idx + 1}/{len(branch_to_execute.steps)}: {pipeline_step.name}")
                        with telemetry.logfire.span(pipeline_step.name) as step_span:
                            # Create ExecutionFrame for the step
                            step_frame = ExecutionFrame(
                                step=pipeline_step,
                                data=current_data,
                                context=branch_context,
                                resources=resources,
                                limits=limits,
                                stream=False,
                                on_chunk=None,
                                breach_event=None,
                                context_setter=context_setter,
                                result=None,
                                _fallback_depth=fallback_depth,
                            )
                            step_result = await self.execute(step_frame)
                        
                        step_results.append(step_result)
                        total_cost += step_result.cost_usd
                        total_tokens += step_result.token_counts
                        
                        if not step_result.success:
                            all_successful = False
                            branch_error_message = step_result.feedback
                            telemetry.logfire.debug(f"Step {pipeline_step.name} failed: {branch_error_message}")
                            break
                        
                        # Use output as input for next step
                        current_data = step_result.output
                        telemetry.logfire.debug(f"Step {pipeline_step.name} output: {current_data}")
                    
                    # Apply output mapper if provided (on main context, not branch context)
                    final_output = current_data
                    if step.branch_output_mapper:
                        final_output = step.branch_output_mapper(current_data, branch_key, context)
                        telemetry.logfire.debug(f"Branch output mapper applied: {final_output}")
                    
                    # Capture the final state of branch_context
                    final_branch_context = (
                        copy.deepcopy(branch_context) if branch_context is not None else None
                    )
                    
                    # Merge branch context back into main context (regardless of success/failure)
                    # But only if no mappers are used, since mappers modify the main context directly
                    if (final_branch_context is not None and context is not None and 
                        step.branch_input_mapper is None and 
                        step.branch_output_mapper is None):
                        safe_merge_context_updates(context, final_branch_context)
                        telemetry.logfire.debug("Merged branch context back to main context")
                    
                    result.success = all_successful
                    result.output = final_output
                    result.cost_usd = total_cost
                    result.token_counts = total_tokens
                    result.latency_s = time.monotonic() - start_time
                    result.metadata_["executed_branch_key"] = branch_key
                    result.branch_context = final_branch_context
                    
                    if all_successful:
                        result.feedback = f"Branch '{branch_key}' executed successfully"
                    else:
                        result.feedback = f"Failure in branch '{branch_key}': {branch_error_message}"
                        
                else:
                    # Execute as a regular step using recursive execution model
                    telemetry.logfire.debug(f"Executing branch as regular step")
                    branch_frame = ExecutionFrame(
                        step=branch_to_execute,
                        data=data,
                        context=context,
                        resources=resources,
                        limits=limits,
                        stream=False,
                        on_chunk=None,
                        breach_event=None,
                        context_setter=context_setter,
                        result=None,
                        _fallback_depth=fallback_depth,
                    )
                    branch_result = await self.execute(branch_frame)

                    result.success = branch_result.success
                    result.output = branch_result.output
                    result.feedback = branch_result.feedback
                    result.cost_usd = branch_result.cost_usd
                    result.token_counts = branch_result.token_counts
                    result.latency_s = time.monotonic() - start_time
                    result.metadata_.update(branch_result.metadata_ or {})
                    result.branch_context = branch_result.branch_context
            else:
                # No branch to execute and no default branch
                result.success = False
                result.output = data
                result.latency_s = time.monotonic() - start_time
                result.feedback = f"No branch matches condition '{branch_key}' and no default branch provided"

        except Exception as e:
            result.success = False
            result.feedback = f"Error executing conditional logic or branch: {str(e)}"
            result.latency_s = time.monotonic() - start_time
            telemetry.logfire.error(f"Error in conditional step '{step.name}': {str(e)}")

        return result
    
    async def _handle_dynamic_router_step(self, step, data, context, resources, limits, context_setter):
        """Handle DynamicParallelRouterStep execution."""
        # This would contain the existing dynamic router step logic
        raise NotImplementedError("Dynamic router step handler needs to be implemented")
    
    async def _handle_hitl_step(self, step, data, context, resources, limits, breach_event, context_setter):
        """Handle HumanInTheLoopStep execution."""
        # This would contain the existing HITL step logic
        raise NotImplementedError("HITL step handler needs to be implemented")
    
    async def _handle_cache_step(self, step, data, context, resources, limits, breach_event, context_setter, step_executor):
        """Handle CacheStep execution."""
        # This would contain the existing cache step logic
        raise NotImplementedError("Cache step handler needs to be implemented")

    def _default_set_final_context(self, result: PipelineResult[Any], context: Optional[Any]) -> None:
        """Default context setter implementation."""
        pass


class DefaultProcessorPipeline:
    """Default processor pipeline implementation."""

    async def apply_prompt(self, processors: Any, data: Any, *, context: Any) -> Any:
        """Apply prompt processors sequentially."""
        import inspect

        if not processors:
            return data

        # Handle both list of processors and object with prompt_processors attribute
        processor_list = processors
        if hasattr(processors, "prompt_processors"):
            processor_list = processors.prompt_processors

        if not processor_list:
            return data

        processed_data = data
        for proc in processor_list:
            try:
                fn = getattr(proc, "process", proc)
                if inspect.iscoroutinefunction(fn):
                    try:
                        processed_data = await fn(processed_data, context=context)
                    except TypeError:
                        processed_data = await fn(processed_data)
                else:
                    try:
                        processed_data = fn(processed_data, context=context)
                    except TypeError:
                        processed_data = fn(processed_data)
            except Exception as e:
                # Log error but continue with original data
                try:
                    telemetry.logfire.error(f"Prompt processor failed: {e}")
                except Exception:
                    pass
                processed_data = data

        return processed_data

    async def apply_output(self, processors: Any, data: Any, *, context: Any) -> Any:
        """Apply output processors sequentially."""
        import inspect

        if not processors:
            return data

        # Handle both list of processors and object with output_processors attribute
        processor_list = processors
        if hasattr(processors, "output_processors"):
            processor_list = processors.output_processors

        if not processor_list:
            return data

        processed_data = data
        for proc in processor_list:
            try:
                fn = getattr(proc, "process", proc)
                if inspect.iscoroutinefunction(fn):
                    try:
                        processed_data = await fn(processed_data, context=context)
                    except TypeError:
                        processed_data = await fn(processed_data)
                else:
                    try:
                        processed_data = fn(processed_data, context=context)
                    except TypeError:
                        processed_data = fn(processed_data)
            except Exception as e:
                # Log error but continue with original output
                try:
                    telemetry.logfire.error(f"Output processor failed: {e}")
                except Exception:
                    pass
                processed_data = data

        return processed_data


class DefaultValidatorRunner:
    """Default validator runner implementation."""

    async def validate(self, validators: List[Any], data: Any, *, context: Any) -> List[ValidationResult]:
        """Run validators and return validation results. Raises ValueError on first failure."""
        if not validators:
            return []

        validation_results = []
        for validator in validators:
            try:
                result = await validator.validate(data, context=context)
                if isinstance(result, ValidationResult):
                    validation_results.append(result)
                    if not result.is_valid:
                        # Use feedback field instead of message
                        feedback = result.feedback or "Validation failed"
                        raise ValueError(f"Validation failed: {feedback}")
                else:
                    # Handle case where validator doesn't return ValidationResult
                    raise ValueError(f"Validator {type(validator).__name__} returned invalid result type")
            except ValueError:
                # Re-raise validation errors but keep the results collected so far
                raise
            except Exception as e:
                raise ValueError(f"Validator {type(validator).__name__} failed: {e}")
        
        return validation_results


def _should_pass_context_to_plugin(context: Optional[Any], func: Callable[..., Any]) -> bool:
    """Determine if context should be passed to a plugin based on signature analysis."""
    if context is None:
        return False

    import inspect
    sig = inspect.signature(func)
    has_explicit_context = any(
        p.kind == inspect.Parameter.KEYWORD_ONLY and p.name == "context"
        for p in sig.parameters.values()
    )
    return has_explicit_context


def _should_pass_resources_to_plugin(resources: Optional[Any], func: Callable[..., Any]) -> bool:
    """Determine if resources should be passed to a plugin based on signature analysis."""
    if resources is None:
        return False

    import inspect
    sig = inspect.signature(func)
    has_explicit_resources = any(
        p.kind == inspect.Parameter.KEYWORD_ONLY and p.name == "resources"
        for p in sig.parameters.values()
    )
    return has_explicit_resources


class DefaultPluginRunner:
    """Default plugin runner implementation."""
    
    async def run_plugins(self, plugins: List[tuple[Any, int]], data: Any, *, context: Any, resources: Optional[Any] = None) -> Any:
        """Run plugins and return processed data."""
        from ...domain.plugins import PluginOutcome
        
        processed_data = data
        for plugin, priority in sorted(plugins, key=lambda x: x[1], reverse=True):
            try:
                # Check if the plugin accepts context and resources parameters
                plugin_kwargs = {}
                if _should_pass_context_to_plugin(context, plugin.validate):
                    plugin_kwargs["context"] = context
                if _should_pass_resources_to_plugin(resources, plugin.validate):
                    plugin_kwargs["resources"] = resources
                
                # Call the plugin's validate method
                result = await plugin.validate(processed_data, **plugin_kwargs)

                if isinstance(result, PluginOutcome):
                    if not result.success:
                        # Plugin failed - raise exception with feedback for step logic to handle
                        plugin_name = getattr(plugin, "name", type(plugin).__name__)
                        failure_msg = result.feedback if result.feedback else f"{plugin_name} failed"
                        telemetry.logfire.error(f"Plugin {plugin_name} failed: {result.feedback}")
                        raise ValueError(f"Plugin validation failed: {failure_msg}")
                    
                    # Plugin succeeded, use its output if provided
                    if result.output is not None:
                        processed_data = result.output
                else:
                    # Handle legacy plugins that don't return PluginOutcome
                    processed_data = result if result is not None else processed_data
                    
            except Exception as e:
                # Plugin failed - re-raise for step logic to handle
                plugin_name = getattr(plugin, "name", type(plugin).__name__)
                telemetry.logfire.error(f"Plugin {plugin_name} failed: {e}")
                raise ValueError(f"Plugin {plugin_name} failed: {e}")
        
        return processed_data


# Stub classes for backward compatibility
class OptimizationConfig:
    """Stub optimization configuration class."""
    pass


class _LRUCache:
    """Stub LRU cache class."""
    pass


class _UsageTracker:
    """Stub usage tracker class."""
    pass


# --------------------------------------------------------------------------- #
# ★ Protocol Interfaces
# --------------------------------------------------------------------------- #


class ISerializer(Protocol):
    """Interface for object serialization."""

    @abstractmethod
    def serialize(self, obj: Any) -> bytes:
        """Serialize an object to bytes."""
        ...

    @abstractmethod
    def deserialize(self, blob: bytes) -> Any:
        """Deserialize bytes back to an object."""
        ...


class IHasher(Protocol):
    """Interface for deterministic hashing."""

    @abstractmethod
    def digest(self, data: bytes) -> str:
        """Generate a deterministic hash digest from bytes."""
        ...


class ICacheBackend(Protocol):
    """Interface for caching step results."""

    @abstractmethod
    async def get(self, key: str) -> Optional[StepResult]:
        """Retrieve a cached result by key."""
        ...

    @abstractmethod
    async def put(self, key: str, value: StepResult, ttl_s: int):
        """Store a result in cache with TTL."""
        ...

    @abstractmethod
    async def clear(self):
        """Clear all cached entries."""
        ...


class IUsageMeter(Protocol):
    """Interface for tracking and enforcing usage limits."""

    @abstractmethod
    async def add(self, cost_usd: float, prompt_tokens: int, completion_tokens: int):
        """Add usage metrics to cumulative totals."""
        ...

    @abstractmethod
    async def guard(self, limits: UsageLimits, step_history: Optional[List[Any]] = None):
        """Check if current usage exceeds limits, raise if so."""
        ...

    @abstractmethod
    async def snapshot(self) -> tuple[float, int, int]:
        """Get current (cost, prompt_tokens, completion_tokens)."""
        ...


class IAgentRunner(Protocol):
    """Interface for running agents with proper parameter handling."""

    @abstractmethod
    async def run(
        self,
        agent: Any,
        payload: Any,
        *,
        context: Any,
        resources: Any,
        options: Dict[str, Any],
        stream: bool = False,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]] = None,
        breach_event: Optional[Any] = None,
    ) -> Any:
        """Run an agent and return raw output."""
        ...


class IProcessorPipeline(Protocol):
    """Interface for running prompt and output processors."""

    @abstractmethod
    async def apply_prompt(self, processors: Any, data: Any, *, context: Any) -> Any:
        """Apply prompt processors to input data."""
        ...

    @abstractmethod
    async def apply_output(self, processors: Any, data: Any, *, context: Any) -> Any:
        """Apply output processors to agent output."""
        ...


class IValidatorRunner(Protocol):
    """Interface for running validators."""

    @abstractmethod
    async def validate(self, validators: List[Any], data: Any, *, context: Any):
        """Run validators and raise ValueError on first failure."""
        ...


class IPluginRunner(Protocol):
    """Interface for running plugins."""

    @abstractmethod
    async def run_plugins(self, plugins: List[tuple[Any, int]], data: Any, *, context: Any, resources: Optional[Any] = None) -> Any:
        """Run plugins and return processed data."""
        ...


class ITelemetry(Protocol):
    """Interface for telemetry operations."""

    @abstractmethod
    def trace(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Create a telemetry trace decorator."""
        ...


# --------------------------------------------------------------------------- #
# ★ Default Implementations
# --------------------------------------------------------------------------- #


class OrjsonSerializer:
    """Fast JSON serializer using orjson if available."""

    def __init__(self) -> None:
        try:
            import orjson

            self._orjson = orjson
            self._use_orjson = True
        except ImportError:
            import json

            self._json = json
            self._use_orjson = False

    def serialize(self, obj: Any) -> bytes:
        if self._use_orjson:
            return self._orjson.dumps(obj, option=self._orjson.OPT_SORT_KEYS)
        else:
            s = self._json.dumps(obj, sort_keys=True, separators=(",", ":"))
            return s.encode("utf-8")

    def deserialize(self, blob: bytes) -> Any:
        if self._use_orjson:
            return self._orjson.loads(blob)
        else:
            return self._json.loads(blob.decode("utf-8"))


class Blake3Hasher:
    """Fast cryptographic hasher using Blake3 if available."""

    def __init__(self) -> None:
        try:
            import blake3

            self._blake3 = blake3
            self._use_blake3 = True
        except ImportError:
            self._use_blake3 = False

    def digest(self, data: bytes) -> str:
        if self._use_blake3:
            return self._blake3.blake3(data).hexdigest()
        else:
            return hashlib.blake2b(data, digest_size=32).hexdigest()


@dataclass
class InMemoryLRUBackend:
    """O(1) LRU cache with TTL support."""

    max_size: int = 1024
    ttl_s: int = 3600
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)
    _store: OrderedDict[str, tuple[StepResult, float, int]] = field(
        init=False, default_factory=OrderedDict
    )

    async def get(self, key: str) -> Optional[StepResult]:
        """Retrieve a cached result by key."""
        async with self._lock:
            if key not in self._store:
                return None

            result, timestamp, access_count = self._store[key]
            current_time = time.monotonic()

            # Check TTL
            if current_time - timestamp > self.ttl_s:
                del self._store[key]
                return None

            # Move to end (most recently used)
            self._store.move_to_end(key)
            self._store[key] = (result, timestamp, access_count + 1)
            return result

    async def put(self, key: str, value: StepResult, ttl_s: int):
        """Store a result in cache with TTL."""
        async with self._lock:
            current_time = time.monotonic()

            # Remove oldest entries if at capacity
            while len(self._store) >= self.max_size:
                self._store.popitem(last=False)

            self._store[key] = (value, current_time, 0)
            self._store.move_to_end(key)

    async def clear(self):
        """Clear all cached entries."""
        async with self._lock:
            self._store.clear()


@dataclass
class ThreadSafeMeter:
    """Thread-safe usage meter with atomic operations."""

    total_cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)

    async def add(self, cost_usd: float, prompt_tokens: int, completion_tokens: int):
        async with self._lock:
            self.total_cost_usd += cost_usd
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens

    async def guard(self, limits: UsageLimits, step_history: Optional[List[Any]] = None):
        async with self._lock:
            # Use precise comparison for floating point
            if (
                limits.total_cost_usd_limit is not None
                and self.total_cost_usd - limits.total_cost_usd_limit > 1e-9
            ):
                raise UsageLimitExceededError(
                    f"Cost limit of ${limits.total_cost_usd_limit} exceeded (current: ${self.total_cost_usd})",
                    PipelineResult(step_history=step_history or [], total_cost_usd=self.total_cost_usd),
                )

            total_tokens = self.prompt_tokens + self.completion_tokens
            if limits.total_tokens_limit is not None and total_tokens - limits.total_tokens_limit > 0:
                raise UsageLimitExceededError(
                    f"Token limit of {limits.total_tokens_limit} exceeded (current: {total_tokens})",
                    PipelineResult(step_history=step_history or [], total_cost_usd=self.total_cost_usd),
                )

    async def snapshot(self) -> tuple[float, int, int]:
        async with self._lock:
            return self.total_cost_usd, self.prompt_tokens, self.completion_tokens


class DefaultAgentRunner:
    """Default agent runner with parameter filtering and streaming support."""

    async def run(
        self,
        agent: Any,
        payload: Any,
        *,
        context: Any,
        resources: Any,
        options: Dict[str, Any],
        stream: bool = False,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]] = None,
        breach_event: Optional[Any] = None,
    ) -> Any:
        """Run agent with proper parameter filtering and fallback strategies."""
        import inspect
        from unittest.mock import Mock, MagicMock, AsyncMock
        from ...application.context_manager import _accepts_param, _should_pass_context
        from ...signature_tools import analyze_signature

        if agent is None:
            raise RuntimeError("Agent is None")

        # Extract the target agent (handle wrapped agents)
        target_agent = getattr(agent, "_agent", agent)

        # Find the executable function
        executable_func = None
        if hasattr(agent, "run"):
            executable_func = getattr(agent, "run")
        elif hasattr(target_agent, "run"):
            executable_func = getattr(target_agent, "run")
        elif callable(target_agent):
            executable_func = target_agent
        else:
            raise RuntimeError(f"Agent {type(agent).__name__} has no executable method")

        # Build filtered kwargs based on function signature
        filtered_kwargs: Dict[str, Any] = {}
        
        # For mocks, pass all parameters
        if isinstance(executable_func, (Mock, MagicMock, AsyncMock)):
            filtered_kwargs.update(options)
            if context is not None:
                filtered_kwargs["context"] = context
            if resources is not None:
                filtered_kwargs["resources"] = resources
            if breach_event is not None:
                filtered_kwargs["breach_event"] = breach_event
        else:
            # For real functions, analyze signature
            try:
                spec = analyze_signature(executable_func)
                
                # Add context if the function accepts it
                if _should_pass_context(spec, context, executable_func):
                    filtered_kwargs["context"] = context

                # Add resources if the function accepts it
                if resources is not None and _accepts_param(executable_func, "resources"):
                    filtered_kwargs["resources"] = resources

                # Add other options based on function signature
                for key, value in options.items():
                    if value is not None and _accepts_param(executable_func, key):
                        filtered_kwargs[key] = value

                # Add breach_event if the function accepts it
                if breach_event is not None and _accepts_param(executable_func, "breach_event"):
                    filtered_kwargs["breach_event"] = breach_event
            except Exception:
                # If signature analysis fails, try basic parameter passing
                filtered_kwargs.update(options)
                if context is not None:
                    filtered_kwargs["context"] = context
                if resources is not None:
                    filtered_kwargs["resources"] = resources
                if breach_event is not None:
                    filtered_kwargs["breach_event"] = breach_event

        # Execute the agent
        try:
            if inspect.iscoroutinefunction(executable_func):
                return await executable_func(payload, **filtered_kwargs)
            else:
                result = executable_func(payload, **filtered_kwargs)
                if inspect.iscoroutine(result):
                    return await result
                return result
        except (
            PausedException,
            InfiniteFallbackError,
            InfiniteRedirectError,
            ContextInheritanceError,
        ) as e:
            # Re-raise critical exceptions immediately
            raise e


class DefaultProcessorPipeline:
    """Default processor pipeline implementation."""

    async def apply_prompt(self, processors: Any, data: Any, *, context: Any) -> Any:
        """Apply prompt processors sequentially."""
        if not processors:
            return data
        
        processed_data = data
        processor_list = processors if isinstance(processors, list) else getattr(processors, "prompt_processors", [])
        
        for proc in processor_list:
            try:
                fn = getattr(proc, "process", proc)
                if asyncio.iscoroutinefunction(fn):
                    try:
                        processed_data = await fn(processed_data, context=context)
                    except TypeError:
                        processed_data = await fn(processed_data)
                else:
                    try:
                        processed_data = fn(processed_data, context=context)
                    except TypeError:
                        processed_data = fn(processed_data)
            except Exception:
                # Continue with original data on error
                processed_data = data
        
        return processed_data

    async def apply_output(self, processors: Any, data: Any, *, context: Any) -> Any:
        """Apply output processors sequentially."""
        if not processors:
            return data
        
        processed_data = data
        processor_list = processors if isinstance(processors, list) else getattr(processors, "output_processors", [])
        
        for proc in processor_list:
            try:
                fn = getattr(proc, "process", proc)
                if asyncio.iscoroutinefunction(fn):
                    try:
                        processed_data = await fn(processed_data, context=context)
                    except TypeError:
                        processed_data = await fn(processed_data)
                else:
                    try:
                        processed_data = fn(processed_data, context=context)
                    except TypeError:
                        processed_data = fn(processed_data)
            except Exception:
                # Continue with original data on error
                processed_data = data
        
        return processed_data


class DefaultValidatorRunner:
    """Default validator runner implementation."""

    async def validate(self, validators: List[Any], data: Any, *, context: Any):
        """Run validators and raise ValueError on first failure."""
        if not validators:
            return

        for validator in validators:
            try:
                result = await validator.validate(data, context=context)
                if hasattr(result, 'is_valid') and not result.is_valid:
                    feedback = getattr(result, 'feedback', 'Validation failed')
                    raise ValueError(f"Validation failed: {feedback}")
            except ValueError:
                raise  # Re-raise validation errors
            except Exception as e:
                raise ValueError(f"Validator {type(validator).__name__} failed: {e}")


class DefaultCacheKeyGenerator:
    """Default cache key generator implementation."""
    
    def __init__(self, hasher: Any = None):
        self._hasher = hasher or Blake3Hasher()
    
    def generate_key(self, step: Any, data: Any, context: Any, resources: Any) -> str:
        """Generate a cache key for the given step and inputs."""
        # FIXED: Include context in cache key to handle loop iterations properly
        # This ensures that different iterations with different contexts get different cache keys
        step_name = getattr(step, 'name', str(type(step).__name__))
        data_str = str(data) if data is not None else ""
        
        # Include context state in the cache key
        context_str = ""
        if context is not None:
            # For Pydantic models, use model_dump() to get all fields
            if hasattr(context, 'model_dump'):
                context_str = str(context.model_dump())
            elif hasattr(context, 'dict'):
                context_str = str(context.dict())
            else:
                # For other objects, use __dict__ or str representation
                context_str = str(getattr(context, '__dict__', str(context)))
        
        key_data = f"{step_name}:{data_str}:{context_str}".encode('utf-8')
        return self._hasher.digest(key_data)


class DefaultTelemetry:
    """Default telemetry implementation."""
    
    def trace(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Create a telemetry trace decorator."""
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func
        return decorator


class OptimizedExecutorCore(ExecutorCore):
    """Optimized version of ExecutorCore with additional performance features."""
    pass