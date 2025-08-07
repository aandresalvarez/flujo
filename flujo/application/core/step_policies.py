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
        # Delegate to the original ExecutorCore's loop implementation
        from flujo.application.core.ultra_executor import ExecutorCore as _OriginalExecutorCore
        return await _OriginalExecutorCore._execute_loop(
            core,
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
        # Prepare branch contexts
        for branch_name, branch_pipeline in parallel_step.branches.items():
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
        # Context merging
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