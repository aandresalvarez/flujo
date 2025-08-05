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
        agent_runner: Any,
        processor_pipeline: Any,
        validator_runner: Any,
        plugin_runner: Any,
        usage_meter: Any,
        cache_backend: Any = None,
        cache_key_generator: Any = None,
        telemetry: Any = None,
        enable_cache: bool = True,
    ):
        """Initialize ExecutorCore with dependency injection."""
        self._agent_runner = agent_runner
        self._processor_pipeline = processor_pipeline
        self._validator_runner = validator_runner
        self._plugin_runner = plugin_runner
        self._usage_meter = usage_meter
        self._cache_backend = cache_backend
        self._cache_key_generator = cache_key_generator
        self._telemetry = telemetry
        self._enable_cache = enable_cache
        self._step_history_so_far: list[StepResult] = []

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
                # Clear feedback on successful fallback
                combined_result.feedback = None
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
            feedback="",
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
                    # Max retries exceeded
                    result.success = False
                    result.feedback = f"Agent execution failed: {str(e)}"
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
                result.feedback = f"Processor failed: {processor_error}"
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
                    result.feedback = f"Validation failed: {validation_error}"
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
                result.feedback = f"Validator error: {validation_error}"
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
                result.feedback = f"Plugin failed: {plugin_error}"
                result.output = processed_output  # Keep the output for fallback
                result.latency_s = time.monotonic() - start_time
                telemetry.logfire.debug(f"Step '{step.name}' plugin failed: {plugin_error}")
                return result

            # All processing succeeded
            result.output = processed_output
            result.success = True
            result.latency_s = time.monotonic() - start_time
            result.feedback = ""

            # Cache successful result
            if self._cache_backend is not None and cache_key is not None:
                if result.metadata_ is None:
                    result.metadata_ = {}
                await self._cache_backend.put(cache_key, result, ttl_s=3600)

            telemetry.logfire.debug(f"Step '{step.name}' completed successfully")
            return result

        # This should never be reached, but just in case
        result.success = False
        result.feedback = "Step execution failed: unexpected error"
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

        telemetry.logfire.debug("=== HANDLE LOOP STEP ===")
        telemetry.logfire.debug(f"Loop step name: {loop_step.name}")

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
                    
                    # Create isolated context for body execution (proper context isolation)
                    if current_context is not None:
                        body_context = copy.deepcopy(current_context)
                    else:
                        from flujo.domain.models import PipelineContext
                        body_context = PipelineContext(initial_prompt=str(current_body_data))
                    
                    # Execute each step in the pipeline
                    all_successful = True
                    total_cost = 0.0
                    total_tokens = 0
                    body_error_message = None
                    
                    for step in loop_step.loop_body_pipeline.steps:
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

                # Accumulate context changes from this iteration (proper context accumulation)
                if body_result.branch_context is not None:
                    # Use safe_merge_context_updates for proper context accumulation
                    current_context = safe_merge_context_updates(current_context, body_result.branch_context)

                # Update cumulative costs and tokens
                cumulative_cost += body_result.cost_usd
                cumulative_tokens += body_result.token_counts

                # Store the body output for exit condition and next iteration
                last_body_output = body_result.output

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
    
    async def _handle_parallel_step(self, step, data, context, resources, limits, breach_event, context_setter):
        """Handle ParallelStep execution."""
        # This would contain the existing parallel step logic
        raise NotImplementedError("Parallel step handler needs to be implemented")
    
    async def _handle_conditional_step(self, step, data, context, resources, limits, context_setter, fallback_depth):
        """Handle ConditionalStep execution."""
        # This would contain the existing conditional step logic
        raise NotImplementedError("Conditional step handler needs to be implemented")
    
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