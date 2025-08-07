from __future__ import annotations
import asyncio
from typing import Any, Awaitable, Optional, Protocol, Callable, Dict, List
from pydantic import BaseModel
from flujo.domain.models import StepResult, UsageLimits
from flujo.exceptions import MissingAgentError, InfiniteFallbackError, UsageLimitExceededError, NonRetryableError, MockDetectionError
from flujo.cost import extract_usage_metrics
from flujo.utils.performance import time_perf_ns, time_perf_ns_to_seconds
from flujo.infra import telemetry
from flujo.application.core.context_manager import ContextManager
from flujo.domain.dsl.parallel import ParallelStep
from flujo.domain.models import PipelineResult
from flujo.domain.dsl.conditional import ConditionalStep
from flujo.domain.dsl.pipeline import Pipeline

import copy
import time
from flujo.domain.dsl.step import MergeStrategy, BranchFailureStrategy

# --- Timeout runner policy ---
class TimeoutRunner(Protocol):
    async def run_with_timeout(self, coro: Awaitable[Any], timeout_s: Optional[float]) -> Any:
        ...

class DefaultTimeoutRunner:
    async def run_with_timeout(self, coro: Awaitable[Any], timeout_s: Optional[float]) -> Any:
        if timeout_s is None:
            return await coro
        return await asyncio.wait_for(coro, timeout_s)

# --- Agent result unpacker policy ---
class AgentResultUnpacker(Protocol):
    def unpack(self, output: Any) -> Any:
        ...

class DefaultAgentResultUnpacker:
    def unpack(self, output: Any) -> Any:
        if isinstance(output, BaseModel):
            return output
        for attr in ("output", "content", "result", "data", "text", "message", "value"):
            if hasattr(output, attr):
                return getattr(output, attr)
        return output

# --- Plugin redirector policy ---
class PluginRedirector(Protocol):
    async def run(
        self,
        initial: Any,
        step: Any,
        data: Any,
        context: Any,
        resources: Any,
        timeout_s: Optional[float]
    ) -> Any:
        ...

class DefaultPluginRedirector:
    def __init__(self, plugin_runner: Any, agent_runner: Any):
        self._plugin_runner = plugin_runner
        self._agent_runner = agent_runner

    async def run(
        self,
        initial: Any,
        step: Any,
        data: Any,
        context: Any,
        resources: Any,
        timeout_s: Optional[float]
    ) -> Any:
        from flujo.exceptions import InfiniteRedirectError, PluginError
        redirect_chain: list[Any] = []
        processed = initial
        unpacker = DefaultAgentResultUnpacker()
        while True:
            outcome = await asyncio.wait_for(
                self._plugin_runner.run_plugins(step.plugins, processed, context=context, resources=resources),
                timeout_s,
            )
            # Handle redirect_to
            if hasattr(outcome, 'redirect_to') and outcome.redirect_to is not None:
                if outcome.redirect_to in redirect_chain:
                    raise InfiniteRedirectError(f"Redirect loop detected at agent {outcome.redirect_to}")
                redirect_chain.append(outcome.redirect_to)
                raw = await asyncio.wait_for(
                    self._agent_runner.run(
                        agent=outcome.redirect_to,
                        payload=data,
                        context=context,
                        resources=resources,
                        options={},
                        stream=False
                    ),
                    timeout_s,
                )
                processed = unpacker.unpack(raw)
                continue
            # Failure
            if hasattr(outcome, 'success') and not outcome.success:
                raise PluginError(outcome.feedback or "Plugin failed without feedback")
            # New solution
            if hasattr(outcome, 'new_solution') and outcome.new_solution is not None:
                processed = outcome.new_solution
                continue
            return outcome

# --- Validator invocation policy ---
class ValidatorInvoker(Protocol):
    async def validate(self, output: Any, step: Any, context: Any, timeout_s: Optional[float]) -> None:
        ...

class DefaultValidatorInvoker:
    def __init__(self, validator_runner: Any):
        self._validator_runner = validator_runner

    async def validate(self, output: Any, step: Any, context: Any, timeout_s: Optional[float]) -> None:
        from flujo.exceptions import ValidationError
        # No validators
        if not getattr(step, 'validators', []):
            return
        results = await asyncio.wait_for(
            self._validator_runner.validate(step.validators, output, context=context),
            timeout_s,
        )
        for r in results:
            if not getattr(r, 'is_valid', False):
                raise ValidationError(r.feedback)

# --- Simple Step Executor policy ---
class SimpleStepExecutor(Protocol):
    async def execute(
        self,
        step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        stream: bool,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]],
        cache_key: Optional[str],
        breach_event: Optional[Any],
        _fallback_depth: int = 0,
    ) -> StepResult:
        ...

class DefaultSimpleStepExecutor:
    async def execute(
        self,
        step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        stream: bool,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]],
        cache_key: Optional[str],
        breach_event: Optional[Any],
        _fallback_depth: int = 0,
    ) -> StepResult:
        # Delegate to the original ExecutorCore's implementation for legacy behavior
        from flujo.application.core.ultra_executor import ExecutorCore as _OriginalExecutorCore
        return await _OriginalExecutorCore._execute_simple_step(
            self,
            step,
            data,
            context,
            resources,
            limits,
            stream,
            on_chunk,
            cache_key,
            breach_event,
            _fallback_depth,
        )

# --- Agent Step Executor policy ---
class AgentStepExecutor(Protocol):
    async def execute(
        self,
        core: Any,
        step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        stream: bool,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]],
        cache_key: Optional[str],
        breach_event: Optional[Any],
        _fallback_depth: int = 0,
    ) -> StepResult:
        ...

class DefaultAgentStepExecutor:
    async def execute(
        self,
        core,
        step,
        data,
        context,
        resources,
        limits,
        stream,
        on_chunk,
        cache_key,
        breach_event,
        _fallback_depth: int = 0,
    ) -> StepResult:
        # Delegate to the original ExecutorCore implementation for legacy behavior
        from flujo.application.core.ultra_executor import ExecutorCore as _OriginalExecutorCore
        return await _OriginalExecutorCore._execute_agent_step(
            core,
            step,
            data,
            context,
            resources,
            limits,
            stream,
            on_chunk,
            cache_key,
            breach_event,
            _fallback_depth,
        )

# --- Loop Step Executor policy ---
class LoopStepExecutor(Protocol):
    async def execute(
        self,
        core: Any,
        loop_step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        context_setter: Optional[Callable[[Any, Optional[Any]], None]],
        _fallback_depth: int = 0,
    ) -> StepResult:
        ...

class DefaultLoopStepExecutor:
    async def execute(
        self,
        core,
        loop_step,
        data,
        context,
        resources,
        limits,
        context_setter,
        _fallback_depth: int = 0,
    ) -> StepResult:
        # Delegate to the core instance's loop implementation
        return await core._execute_loop(
            loop_step,
            data,
            context,
            resources,
            limits,
            context_setter,
            _fallback_depth,
        )

# --- Parallel Step Executor policy ---
class ParallelStepExecutor(Protocol):
    async def execute(
        self,
        core: Any,
        step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        breach_event: Optional[Any],
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
        parallel_step: Optional[ParallelStep] = None,
        step_executor: Optional[Callable[..., Awaitable[StepResult]]] = None,
    ) -> StepResult:
        ...

class DefaultParallelStepExecutor:
    async def execute(
        self,
        core,
        step,
        data,
        context,
        resources,
        limits,
        breach_event,
        context_setter,
        parallel_step=None,
        step_executor=None,
    ) -> StepResult:
        # Actual parallel-step execution logic extracted from legacy `_handle_parallel_step`
        if parallel_step is not None:
            step = parallel_step
        if not isinstance(step, ParallelStep):
            raise ValueError(f"Expected ParallelStep, got {type(step)}")
        parallel_step = step
        telemetry.logfire.debug(f"=== HANDLING PARALLEL STEP === {parallel_step.name}")
        telemetry.logfire.debug(f"Parallel step branches: {list(parallel_step.branches.keys())}")
        result = StepResult(name=parallel_step.name)
        result.metadata_ = {}
        start_time = time.monotonic()
        # Handle empty branches
        if not parallel_step.branches:
            result.success = False
            result.feedback = "Parallel step has no branches to execute"
            result.output = {}
            result.latency_s = time.monotonic() - start_time
            return result
        # Set up usage governor
        usage_governor = core._ParallelUsageGovernor(limits) if limits else None
        if breach_event is None and limits is not None:
            breach_event = asyncio.Event()
        # Tracking variables
        branch_results: Dict[str, StepResult] = {}
        branch_contexts: Dict[str, Any] = {}
        total_cost = 0.0
        total_tokens = 0
        all_successful = True
        failure_messages: List[str] = []
        # Prepare branch contexts with proper isolation
        for branch_name, branch_pipeline in parallel_step.branches.items():
            # Use ContextManager for proper deep isolation
            branch_context = ContextManager.isolate(context, include_keys=parallel_step.context_include_keys) if context is not None else None
            branch_contexts[branch_name] = branch_context
        # Branch executor
        async def execute_branch(branch_name: str, branch_pipeline: Any, branch_context: Any):
            try:
                telemetry.logfire.debug(f"Executing branch: {branch_name}")
                if step_executor is not None:
                    branch_result = await step_executor(branch_pipeline, data, branch_context, resources, breach_event)
                else:
                    pipeline_result = await core._execute_pipeline(
                        branch_pipeline, data, branch_context, resources, limits, breach_event, context_setter
                    )
                    pipeline_success = all(s.success for s in pipeline_result.step_history) if pipeline_result.step_history else False
                    branch_result = StepResult(
                        name=f"{parallel_step.name}_{branch_name}",
                        output=(pipeline_result.step_history[-1].output if pipeline_result.step_history else None),
                        success=pipeline_success,
                        attempts=1,
                        latency_s=sum(s.latency_s for s in pipeline_result.step_history),
                        token_counts=pipeline_result.total_tokens,
                        cost_usd=pipeline_result.total_cost_usd,
                        feedback=(pipeline_result.step_history[-1].feedback if pipeline_result.step_history else ""),
                        branch_context=pipeline_result.final_pipeline_context,
                        metadata_={},
                    )
                if usage_governor is not None:
                    breached = await usage_governor.add_usage(branch_result.cost_usd, branch_result.token_counts, branch_result)
                    if breached and breach_event is not None:
                        breach_event.set()
                telemetry.logfire.debug(f"Branch {branch_name} completed: success={branch_result.success}")
                return branch_name, branch_result
            except Exception as e:
                telemetry.logfire.error(f"Branch {branch_name} failed with exception: {e}")
                failure = StepResult(
                    name=f"{parallel_step.name}_{branch_name}",
                    output=None,
                    success=False,
                    attempts=1,
                    latency_s=0.0,
                    token_counts=0,
                    cost_usd=0.0,
                    feedback=f"Branch execution failed: {e}",
                    branch_context=context,
                    metadata_={},
                )
                return branch_name, failure
        # Execute branches concurrently
        tasks = [execute_branch(n, p, branch_contexts[n]) for n, p in parallel_step.branches.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for branch_name, branch_result in results:
            if isinstance(branch_result, Exception):
                telemetry.logfire.error(f"Branch {branch_name} raised exception: {branch_result}")
                branch_result = StepResult(
                    name=f"{parallel_step.name}_{branch_name}",
                    output=None,
                    success=False,
                    attempts=1,
                    latency_s=0.0,
                    token_counts=0,
                    cost_usd=0.0,
                    feedback=f"Branch execution failed: {branch_result}",
                    metadata_={},
                )
            branch_results[branch_name] = branch_result
            if branch_result.success:
                total_cost += branch_result.cost_usd
                total_tokens += branch_result.token_counts
            else:
                all_successful = False
                failure_messages.append(f"Branch '{branch_name}': {branch_result.feedback}")
        # Post-usage check
        if usage_governor is not None and usage_governor.breached():
            err = usage_governor.get_error()
            if err:
                telemetry.logfire.error(f"Parallel step usage limit breached: {err}")
                pipeline_result = PipelineResult(
                    step_history=list(branch_results.values()),
                    total_cost_usd=total_cost,
                    total_tokens=total_tokens,
                    final_pipeline_context=context,
                )
                raise UsageLimitExceededError(str(err), pipeline_result)
        # Overall success
        if parallel_step.on_branch_failure == BranchFailureStrategy.PROPAGATE:
            result.success = all_successful
        elif parallel_step.on_branch_failure == BranchFailureStrategy.IGNORE:
            result.success = any(br.success for br in branch_results.values())
        else:
            result.success = all_successful
        # Build output
        output_dict: Dict[str, Any] = {}
        for bn, br in branch_results.items():
            output_dict[bn] = br.output if br.success else br
        result.output = output_dict
        result.metadata_["executed_branches"] = list(branch_results.keys())
        # Context merging using ContextManager
        if context is not None and parallel_step.merge_strategy != MergeStrategy.NO_MERGE:
            try:
                branch_ctxs = {n: br.branch_context for n, br in branch_results.items() if br.branch_context is not None}
                if parallel_step.merge_strategy == MergeStrategy.CONTEXT_UPDATE:
                    for n, bc in branch_ctxs.items():
                        if parallel_step.field_mapping and n in parallel_step.field_mapping:
                            for f in parallel_step.field_mapping[n]:
                                if hasattr(bc, f):
                                    setattr(context, f, getattr(bc, f))
                        else:
                            # Use ContextManager for safe merging
                            context = ContextManager.merge(context, bc)
                elif parallel_step.merge_strategy == MergeStrategy.MERGE_SCRATCHPAD:
                    if not hasattr(context, "scratchpad"):
                        setattr(context, "scratchpad", {})
                    for n in sorted(branch_ctxs):
                        bc = branch_ctxs[n]
                        if hasattr(bc, "scratchpad"):
                            for k in bc.scratchpad:
                                if k in context.scratchpad:
                                    telemetry.logfire.warning(f"Scratchpad key collision: '{k}', skipping")
                                else:
                                    context.scratchpad[k] = bc.scratchpad[k]
                elif parallel_step.merge_strategy == MergeStrategy.OVERWRITE and branch_ctxs:
                    last = sorted(branch_ctxs)[-1]
                    branch_ctx = branch_ctxs[last]
                    if parallel_step.context_include_keys:
                        for f in parallel_step.context_include_keys:
                            if hasattr(branch_ctx, f):
                                setattr(context, f, getattr(branch_ctx, f))
                    else:
                        if hasattr(context, "scratchpad"):
                            for bn in sorted(branch_ctxs):
                                bc = branch_ctxs[bn]
                                if hasattr(bc, "scratchpad"):
                                    for key, val in bc.scratchpad.items():
                                        context.scratchpad[key] = val
                elif callable(parallel_step.merge_strategy):
                    parallel_step.merge_strategy(context, branch_ctxs)
                
                # Special handling for executed_branches field - merge it back to context
                if hasattr(context, "executed_branches"):
                    # Get all executed branches from branch contexts
                    all_executed_branches = []
                    for bc in branch_ctxs.values():
                        if hasattr(bc, "executed_branches") and bc.executed_branches:
                            all_executed_branches.extend(bc.executed_branches)
                    
                    # Handle executed_branches based on merge strategy
                    if parallel_step.merge_strategy == MergeStrategy.OVERWRITE:
                        # For OVERWRITE, only keep the last successful branch
                        successful_branches = [name for name, br in branch_results.items() if br.success]
                        if successful_branches:
                            # Get the last successful branch (alphabetically sorted)
                            last_successful_branch = sorted(successful_branches)[-1]
                            context.executed_branches = [last_successful_branch]
                            
                            # Also handle branch_results for OVERWRITE strategy
                            if hasattr(context, "branch_results"):
                                # Get the branch_results from the last successful branch context
                                last_branch_ctx = branch_ctxs.get(last_successful_branch)
                                if last_branch_ctx and hasattr(last_branch_ctx, "branch_results"):
                                    context.branch_results = last_branch_ctx.branch_results.copy()
                                else:
                                    # If no branch_results in context, create from current results
                                    context.branch_results = {last_successful_branch: branch_results[last_successful_branch].output}
                        else:
                            context.executed_branches = []
                            if hasattr(context, "branch_results"):
                                context.branch_results = {}
                    else:
                        # For other strategies, add all successful branches
                        successful_branches = [name for name, br in branch_results.items() if br.success]
                        all_executed_branches.extend(successful_branches)
                        
                        # Remove duplicates while preserving order
                        seen = set()
                        unique_branches = []
                        for branch in all_executed_branches:
                            if branch not in seen:
                                seen.add(branch)
                                unique_branches.append(branch)
                        
                        # Update context with merged executed_branches
                        context.executed_branches = unique_branches
                        
                        # Handle branch_results for other strategies
                        if hasattr(context, "branch_results"):
                            # Merge branch_results from all successful branches
                            merged_branch_results = {}
                            for bc in branch_ctxs.values():
                                if hasattr(bc, "branch_results") and bc.branch_results:
                                    merged_branch_results.update(bc.branch_results)
                            context.branch_results = merged_branch_results
                
                result.branch_context = context
            except Exception as e:
                telemetry.logfire.error(f"Context merging failed: {e}")
        # Finalize result
        result.cost_usd = total_cost
        result.token_counts = total_tokens
        result.latency_s = time.monotonic() - start_time
        result.attempts = 1
        if result.success:
            result.feedback = (
                f"All {len(parallel_step.branches)} branches executed successfully"
                if all_successful
                else f"Parallel step completed with {len(failure_messages)} branch failures (ignored)"
            )
        else:
            result.feedback = f"Parallel step failed with {len(failure_messages)} branch failures"
        return result

class ConditionalStepExecutor(Protocol):
    async def execute(
        self,
        core: Any,
        conditional_step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
        _fallback_depth: int = 0,
    ) -> StepResult:
        ...

class DefaultConditionalStepExecutor:
    async def execute(
        self,
        core,
        conditional_step,
        data,
        context,
        resources,
        limits,
        context_setter,
        _fallback_depth: int = 0,
    ) -> StepResult:
        """Handle ConditionalStep execution with proper context isolation and merging."""
        import time
        import copy
        from flujo.domain.dsl.pipeline import Pipeline
        from flujo.application.core.context_manager import ContextManager
        from ...utils.context import safe_merge_context_updates

        telemetry.logfire.debug("=== HANDLE CONDITIONAL STEP ===")
        telemetry.logfire.debug(f"Conditional step name: {conditional_step.name}")

        # Initialize result
        result = StepResult(
            name=conditional_step.name,
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
            branch_key = conditional_step.condition_callable(data, context)
            telemetry.logfire.debug(f"Condition evaluated to branch key: {branch_key}")
            # Determine branch
            branch_to_execute = None
            if branch_key in conditional_step.branches:
                branch_to_execute = conditional_step.branches[branch_key]
                result.metadata_["executed_branch_key"] = branch_key
            elif conditional_step.default_branch_pipeline is not None:
                branch_to_execute = conditional_step.default_branch_pipeline
                result.metadata_["executed_branch_key"] = "default"
            else:
                telemetry.logfire.warn(f"No branch matches condition '{branch_key}' and no default branch provided")
            # Execute selected branch
            if branch_to_execute:
                branch_data = data
                if conditional_step.branch_input_mapper:
                    branch_data = conditional_step.branch_input_mapper(data, context)
                # Use ContextManager for proper deep isolation
                branch_context = ContextManager.isolate(context) if context is not None else None
                # Execute pipeline
                total_cost = 0.0
                total_tokens = 0
                step_history = []
                for pipeline_step in (branch_to_execute.steps if isinstance(branch_to_execute, Pipeline) else [branch_to_execute]):
                    step_result = await core.execute(
                        pipeline_step,
                        branch_data,
                        context=branch_context,
                        resources=resources,
                        limits=limits,
                        context_setter=context_setter,
                        _fallback_depth=_fallback_depth,
                    )
                    total_cost += step_result.cost_usd
                    total_tokens += step_result.token_counts
                    branch_data = step_result.output
                    if not step_result.success:
                        result.feedback = step_result.feedback
                        result.success = False
                        result.latency_s = time.monotonic() - start_time
                        result.token_counts = total_tokens
                        result.cost_usd = total_cost
                        return result
                    step_history.append(step_result)
                result.success = True
                result.output = branch_data
                result.latency_s = time.monotonic() - start_time
                result.token_counts = total_tokens
                result.cost_usd = total_cost
                # Update branch context using ContextManager
                result.branch_context = ContextManager.merge(context, branch_context) if context is not None else branch_context
                return result
        except Exception as e:
            result.feedback = f"Error executing conditional step: {e}"
        result.latency_s = time.monotonic() - start_time
        return result

# --- Dynamic Router Step Executor policy ---
from flujo.domain.models import StepResult, PipelineResult
from flujo.domain.dsl.step import Step
from flujo.domain.dsl.parallel import ParallelStep
from .types import ExecutionFrame
from .context_manager import ContextManager

class DynamicRouterStepExecutor(Protocol):
    async def execute(
        self,
        core: Any,
        router_step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
    ) -> StepResult:
        ...

class DefaultDynamicRouterStepExecutor:
    async def execute(
        self,
        core,
        router_step,
        data,
        context,
        resources,
        limits,
        context_setter,
    ) -> StepResult:
        """Handle DynamicParallelRouterStep execution with proper branch selection and parallel delegation."""
        import time
        import asyncio
        telemetry.logfire.debug("=== HANDLE DYNAMIC ROUTER STEP ===")
        telemetry.logfire.debug(f"Dynamic router step name: {router_step.name}")

        # Phase 1: Execute the router agent to decide which branches to run
        router_agent_step = Step(name=f"{router_step.name}_router", agent=router_step.router_agent)
        router_frame = ExecutionFrame(
            step=router_agent_step,
            data=data,
            context=context,
            resources=resources,
            limits=limits,
            stream=False,
            on_chunk=None,
            breach_event=None,
            context_setter=context_setter,
        )
        router_result = await core.execute(router_frame)

        # Handle router failure
        if not router_result.success:
            result = StepResult(name=core._safe_step_name(router_step), success=False, feedback=f"Router agent failed: {router_result.feedback}")
            result.cost_usd = router_result.cost_usd
            result.token_counts = router_result.token_counts
            return result

        # Process router output to get branch names
        selected_branch_names = router_result.output
        if isinstance(selected_branch_names, str):
            selected_branch_names = [selected_branch_names]
        if not isinstance(selected_branch_names, list):
            return StepResult(name=core._safe_step_name(router_step), success=False, feedback=f"Router agent must return a list of branch names, got {type(selected_branch_names).__name__}")

        # Filter branches based on router's decision
        selected_branches = {
            name: router_step.branches[name]
            for name in selected_branch_names
            if name in router_step.branches
        }
        # Handle no selected branches
        if not selected_branches:
            return StepResult(name=core._safe_step_name(router_step), success=True, output={}, cost_usd=router_result.cost_usd, token_counts=router_result.token_counts)

        # Phase 2: Execute selected branches in parallel via policy
        temp_parallel_step = ParallelStep(
            name=router_step.name,
            branches=selected_branches,
            merge_strategy=router_step.merge_strategy,
            on_branch_failure=router_step.on_branch_failure,
            context_include_keys=router_step.context_include_keys,
            field_mapping=router_step.field_mapping,
        )
        parallel_result = await core.parallel_step_executor.execute(
            core,
            temp_parallel_step,
            data,
            context,
            resources,
            limits,
            None,
            context_setter,
        )

        # Add router usage metrics
        parallel_result.cost_usd += router_result.cost_usd
        parallel_result.token_counts += router_result.token_counts

        # Merge branch context into original context
        if parallel_result.branch_context is not None and context is not None:
            merged_ctx = ContextManager.merge(context, parallel_result.branch_context)
            parallel_result.branch_context = merged_ctx
            if context_setter is not None:
                try:
                    pipeline_result = PipelineResult(
                        step_history=[parallel_result],
                        total_cost_usd=parallel_result.cost_usd,
                        total_tokens=parallel_result.token_counts,
                        total_latency_s=parallel_result.latency_s,
                        final_pipeline_context=parallel_result.branch_context,
                    )
                    context_setter(pipeline_result, context)
                except Exception as e:
                    telemetry.logfire.warning(f"Context setter failed for DynamicParallelRouterStep: {e}")

        # Record executed branches
        parallel_result.metadata_["executed_branches"] = selected_branch_names
        return parallel_result

# --- End Dynamic Router Step Executor policy ---

# --- Human-In-The-Loop Step Executor policy ---
from flujo.domain.dsl.step import HumanInTheLoopStep
from flujo.exceptions import PausedException

class HitlStepExecutor(Protocol):
    async def execute(
        self,
        core: Any,
        step: HumanInTheLoopStep,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
    ) -> StepResult:
        ...

class DefaultHitlStepExecutor:
    async def execute(
        self,
        core,
        step,
        data,
        context,
        resources,
        limits,
        context_setter,
    ) -> StepResult:
        """Handle Human-In-The-Loop step execution."""
        import time

        telemetry.logfire.debug("=== HANDLE HITL STEP ===")
        telemetry.logfire.debug(f"HITL step name: {step.name}")

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

        if context is not None:
            try:
                if hasattr(context, 'scratchpad') and isinstance(context.scratchpad, dict):
                    context.scratchpad['status'] = 'paused'
                    context.scratchpad['last_state_update'] = time.monotonic()
                else:
                    core._update_context_state(context, 'paused')
            except Exception as e:
                telemetry.logfire.error(f"Failed to update context state: {e}")

        if context is not None and hasattr(context, 'scratchpad'):
            try:
                try:
                    hitl_message = step.message_for_user if step.message_for_user is not None else str(data)
                except Exception:
                    hitl_message = "Data conversion failed"
                context.scratchpad['hitl_message'] = hitl_message
                context.scratchpad['hitl_data'] = data
            except Exception as e:
                telemetry.logfire.error(f"Failed to update context scratchpad: {e}")

        try:
            message = step.message_for_user if step.message_for_user is not None else str(data)
        except Exception:
            message = "Data conversion failed"
        raise PausedException(message)
# --- End Human-In-The-Loop Step Executor policy ---

# --- Cache Step Executor policy ---
from flujo.steps.cache_step import CacheStep, _generate_cache_key
from .types import ExecutionFrame
from flujo.application.core.context_adapter import _build_context_update, _inject_context
import asyncio

class CacheStepExecutor(Protocol):
    async def execute(
        self,
        core: Any,
        cache_step: CacheStep[Any, Any],
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        breach_event: Optional[Any],
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
        step_executor: Optional[Callable[..., Awaitable[StepResult]]],
    ) -> StepResult:
        ...

class DefaultCacheStepExecutor:
    async def execute(
        self,
        core,
        cache_step,
        data,
        context,
        resources,
        limits,
        breach_event,
        context_setter,
        step_executor=None,
    ) -> StepResult:
        """Handle CacheStep execution with concurrency control and resilience."""
        try:
            cache_key = _generate_cache_key(cache_step.wrapped_step, data, context, resources)
        except Exception as e:
            telemetry.logfire.warning(f"Cache key generation failed for step '{cache_step.name}': {e}. Skipping cache.")
            cache_key = None
        if cache_key:
            async with core._cache_locks_lock:
                if cache_key not in core._cache_locks:
                    core._cache_locks[cache_key] = asyncio.Lock()
            async with core._cache_locks[cache_key]:
                try:
                    cached_result = await cache_step.cache_backend.get(cache_key)
                    if cached_result is not None:
                        if cached_result.metadata_ is None:
                            cached_result.metadata_ = {}
                        cached_result.metadata_["cache_hit"] = True
                        if cached_result.branch_context is not None and context is not None:
                            update_data = _build_context_update(cached_result.output)
                            if update_data:
                                validation_error = _inject_context(context, update_data, type(context))
                                if validation_error:
                                    cached_result.success = False
                                    cached_result.feedback = f"Context validation failed: {validation_error}"
                        return cached_result
                except Exception as e:
                    telemetry.logfire.error(f"Cache backend GET failed for step '{cache_step.name}': {e}")
                frame = ExecutionFrame(
                    step=cache_step.wrapped_step,
                    data=data,
                    context=context,
                    resources=resources,
                    limits=limits,
                    stream=False,
                    on_chunk=None,
                    breach_event=breach_event,
                    context_setter=context_setter,
                    _fallback_depth=0,
                )
                result = await core.execute(frame)
                if result.success:
                    try:
                        await cache_step.cache_backend.set(cache_key, result)
                    except Exception as e:
                        telemetry.logfire.error(f"Cache backend SET failed for step '{cache_step.name}': {e}")
                return result
        frame = ExecutionFrame(
            step=cache_step.wrapped_step,
            data=data,
            context=context,
            resources=resources,
            limits=limits,
            stream=False,
            on_chunk=None,
            breach_event=breach_event,
            context_setter=context_setter,
            _fallback_depth=0,
        )
        return await core.execute(frame)
# --- End Cache Step Executor policy ---