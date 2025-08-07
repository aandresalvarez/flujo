"""
Policy classes for executing different step types.

This module contains policy classes that implement the logic for executing 
specific types of steps, moving business logic out of the monolithic ExecutorCore
into specialized, testable policies.

Each policy class is responsible for executing a specific step type and should
be fully self-contained and testable in isolation.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional, Callable, Awaitable

from ...domain.dsl.step import Step, HumanInTheLoopStep
from ...domain.dsl.loop import LoopStep
from ...domain.dsl.conditional import ConditionalStep
from ...domain.dsl.dynamic_router import DynamicParallelRouterStep
from ...domain.dsl.parallel import ParallelStep
from ...domain.models import StepResult, UsageLimits, PipelineResult
from ...exceptions import (
    MissingAgentError,
    InfiniteFallbackError,
    PausedException,
)
from ...steps.cache_step import CacheStep


class DefaultSimpleStepExecutor:
    """Policy for executing simple steps with plugins, validators, and fallbacks."""
    
    async def execute(
        self,
        core: Any,  # ExecutorCore instance
        step: Step,
        data: Any,
        context: Optional[Any] = None,
        resources: Optional[Any] = None,
        usage_limits: Optional[UsageLimits] = None,
        stream: bool = False,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]] = None,
        breach_event: Optional[Any] = None,
        context_setter: Optional[Callable[["PipelineResult[Any]", Optional[Any]], None]] = None,
    ) -> StepResult:
        """Execute a simple step with full plugin+validator orchestration."""
        # For now, delegate to core step logic - will be migrated in Task 1.2
        from .step_logic import _run_step_logic
        
        # Create step executor wrapper for recursion
        async def step_executor_wrapper(
            s: Any,
            d: Any,
            c: Optional[Any],
            r: Optional[Any],
            breach_event: Optional[Any] = None,
        ) -> StepResult:
            return await core.execute_step(
                s, d, c, r, usage_limits, stream, on_chunk, breach_event, context_setter=context_setter
            )
        
        return await _run_step_logic(
            step=step,
            data=data,
            context=context,
            resources=resources,
            step_executor=step_executor_wrapper,
            context_model_defined=True,
            usage_limits=usage_limits,
            context_setter=context_setter,
            stream=stream,
            on_chunk=on_chunk,
        )


class DefaultAgentStepExecutor:
    """Policy for executing agent steps."""
    
    async def execute(
        self,
        core: Any,  # ExecutorCore instance
        step: Step,
        data: Any,
        context: Optional[Any] = None,
        resources: Optional[Any] = None,
        usage_limits: Optional[UsageLimits] = None,
        stream: bool = False,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]] = None,
        breach_event: Optional[Any] = None,
        context_setter: Optional[Callable[["PipelineResult[Any]", Optional[Any]], None]] = None,
    ) -> StepResult:
        """Execute an agent step."""
        # For now, delegate to simple step executor - agent steps are handled via _execute_complex_step
        simple_executor = DefaultSimpleStepExecutor()
        return await simple_executor.execute(
            core, step, data, context, resources, usage_limits, stream, on_chunk, breach_event, context_setter
        )


class DefaultLoopStepExecutor:
    """Policy for executing loop steps."""
    
    async def execute(
        self,
        core: Any,  # ExecutorCore instance
        step: LoopStep,
        data: Any,
        context: Optional[Any] = None,
        resources: Optional[Any] = None,
        usage_limits: Optional[UsageLimits] = None,
        stream: bool = False,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]] = None,
        breach_event: Optional[Any] = None,
        context_setter: Optional[Callable[["PipelineResult[Any]", Optional[Any]], None]] = None,
    ) -> StepResult:
        """Execute a loop step."""
        # For now, delegate to step logic handler - will be migrated in Task 2.1
        from .step_logic import _handle_loop_step
        
        # Create special step executor for loop steps that bypasses state persistence
        async def loop_step_executor(
            s: Any,
            d: Any,
            c: Optional[Any],
            r: Optional[Any],
            breach_event: Optional[Any] = None,
        ) -> StepResult:
            # For loop steps, call step logic directly to avoid state persistence
            from .step_logic import _run_step_logic

            # Create a proper wrapper that matches StepExecutor signature
            async def step_executor_wrapper(
                step: Any,
                data: Any,
                context: Optional[Any],
                resources: Optional[Any],
                breach_event: Optional[Any] = None,
            ) -> StepResult:
                return await core.execute_step(
                    step,
                    data,
                    context,
                    resources,
                    usage_limits,
                    stream,
                    on_chunk,
                    breach_event,
                    context_setter=context_setter,
                )

            return await _run_step_logic(
                step=s,
                data=d,
                context=c,
                resources=r,
                step_executor=step_executor_wrapper,
                context_model_defined=True,
                usage_limits=usage_limits,
                context_setter=context_setter,
                stream=stream,
                on_chunk=on_chunk,
            )
        
        return await _handle_loop_step(
            step,
            data,
            context,
            resources,
            loop_step_executor,
            context_model_defined=True,
            usage_limits=usage_limits,
            context_setter=context_setter,
        )


class DefaultParallelStepExecutor:
    """Policy for executing parallel steps."""
    
    async def execute(
        self,
        core: Any,  # ExecutorCore instance
        step: ParallelStep,
        data: Any,
        context: Optional[Any] = None,
        resources: Optional[Any] = None,
        usage_limits: Optional[UsageLimits] = None,
        stream: bool = False,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]] = None,
        breach_event: Optional[Any] = None,
        context_setter: Optional[Callable[["PipelineResult[Any]", Optional[Any]], None]] = None,
    ) -> StepResult:
        """Execute a parallel step."""
        from .step_logic import _handle_parallel_step
        
        # Create step executor wrapper for recursion
        async def step_executor_wrapper(
            s: Any,
            d: Any,
            c: Optional[Any],
            r: Optional[Any],
            breach_event: Optional[Any] = None,
        ) -> StepResult:
            return await core.execute_step(
                s, d, c, r, usage_limits, stream, on_chunk, breach_event, context_setter=context_setter
            )
        
        return await _handle_parallel_step(
            step,
            data,
            context,
            resources,
            step_executor_wrapper,
            context_model_defined=True,
            usage_limits=usage_limits,
            context_setter=context_setter,
        )


class DefaultConditionalStepExecutor:
    """Policy for executing conditional steps."""
    
    async def execute(
        self,
        core: Any,  # ExecutorCore instance
        step: ConditionalStep,
        data: Any,
        context: Optional[Any] = None,
        resources: Optional[Any] = None,
        usage_limits: Optional[UsageLimits] = None,
        stream: bool = False,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]] = None,
        breach_event: Optional[Any] = None,
        context_setter: Optional[Callable[["PipelineResult[Any]", Optional[Any]], None]] = None,
    ) -> StepResult:
        """Execute a conditional step."""
        from .step_logic import _handle_conditional_step
        
        # Create step executor wrapper for recursion
        async def step_executor_wrapper(
            s: Any,
            d: Any,
            c: Optional[Any],
            r: Optional[Any],
            breach_event: Optional[Any] = None,
        ) -> StepResult:
            return await core.execute_step(
                s, d, c, r, usage_limits, stream, on_chunk, breach_event, context_setter=context_setter
            )
        
        return await _handle_conditional_step(
            step,
            data,
            context,
            resources,
            step_executor_wrapper,
            context_model_defined=True,
            usage_limits=usage_limits,
            context_setter=context_setter,
        )


class DefaultDynamicRouterStepExecutor:
    """Policy for executing dynamic router steps."""
    
    async def execute(
        self,
        core: Any,  # ExecutorCore instance
        step: DynamicParallelRouterStep,
        data: Any,
        context: Optional[Any] = None,
        resources: Optional[Any] = None,
        usage_limits: Optional[UsageLimits] = None,
        stream: bool = False,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]] = None,
        breach_event: Optional[Any] = None,
        context_setter: Optional[Callable[["PipelineResult[Any]", Optional[Any]], None]] = None,
    ) -> StepResult:
        """Execute a dynamic router step."""
        from .step_logic import _handle_dynamic_router_step
        
        # Create step executor wrapper for recursion
        async def step_executor_wrapper(
            s: Any,
            d: Any,
            c: Optional[Any],
            r: Optional[Any],
            breach_event: Optional[Any] = None,
        ) -> StepResult:
            return await core.execute_step(
                s, d, c, r, usage_limits, stream, on_chunk, breach_event, context_setter=context_setter
            )
        
        return await _handle_dynamic_router_step(
            step,
            data,
            context,
            resources,
            step_executor_wrapper,
            context_model_defined=True,
            usage_limits=usage_limits,
            context_setter=context_setter,
        )


class DefaultCacheStepExecutor:
    """Policy for executing cache steps."""
    
    async def execute(
        self,
        core: Any,  # ExecutorCore instance
        step: CacheStep,
        data: Any,
        context: Optional[Any] = None,
        resources: Optional[Any] = None,
        usage_limits: Optional[UsageLimits] = None,
        stream: bool = False,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]] = None,
        breach_event: Optional[Any] = None,
        context_setter: Optional[Callable[["PipelineResult[Any]", Optional[Any]], None]] = None,
    ) -> StepResult:
        """Execute a cache step."""
        from .step_logic import _handle_cache_step
        
        # Create step executor wrapper for recursion
        async def step_executor_wrapper(
            s: Any,
            d: Any,
            c: Optional[Any],
            r: Optional[Any],
            breach_event: Optional[Any] = None,
        ) -> StepResult:
            return await core.execute_step(
                s, d, c, r, usage_limits, stream, on_chunk, breach_event, context_setter=context_setter
            )
        
        return await _handle_cache_step(step, data, context, resources, step_executor_wrapper)


class DefaultHitlStepExecutor:
    """Policy for executing human-in-the-loop steps."""
    
    async def execute(
        self,
        core: Any,  # ExecutorCore instance
        step: HumanInTheLoopStep,
        data: Any,
        context: Optional[Any] = None,
        resources: Optional[Any] = None,
        usage_limits: Optional[UsageLimits] = None,
        stream: bool = False,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]] = None,
        breach_event: Optional[Any] = None,
        context_setter: Optional[Callable[["PipelineResult[Any]", Optional[Any]], None]] = None,
    ) -> StepResult:
        """Execute a human-in-the-loop step."""
        from .step_logic import _handle_hitl_step
        
        return await _handle_hitl_step(step, data, context)


# Additional helper policies for specific functionality

class DefaultPluginRedirector:
    """Policy for redirecting and running plugin operations."""
    
    async def run(self, plugin: Any, data: Any, context: Optional[Any] = None, **kwargs) -> Any:
        """Run a plugin and raise generic exception on failure."""
        try:
            if hasattr(plugin, 'run'):
                return await plugin.run(data, context=context, **kwargs)
            else:
                return await plugin(data, context=context, **kwargs)
        except Exception as e:
            # Raise generic exception for core to re-wrap as PluginError
            raise Exception(f"Plugin execution failed: {str(e)}") from e


class DefaultValidatorInvoker:
    """Policy for invoking validators."""
    
    async def validate(self, validators: list[Any], data: Any, context: Optional[Any] = None) -> bool:
        """Validate data against all validators and raise generic exception on first failure."""
        for validator in validators:
            try:
                if hasattr(validator, 'validate'):
                    result = await validator.validate(data, context=context)
                else:
                    result = await validator(data, context=context)
                
                if not result:
                    raise Exception(f"Validation failed for validator: {validator}")
            except Exception as e:
                # Raise generic exception for core to re-wrap as ValidationError
                raise Exception(f"Validator execution failed: {str(e)}") from e
        
        return True