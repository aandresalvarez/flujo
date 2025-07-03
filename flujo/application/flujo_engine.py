from __future__ import annotations

import asyncio
import inspect
import time
import weakref
import copy
from unittest.mock import Mock
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    Type,
    TypeVar,
    AsyncIterator,
    Awaitable,
    Union,
    cast,
    TypeAlias,
    get_type_hints,
    get_origin,
    get_args,
)

from pydantic import ValidationError

from ..infra.telemetry import logfire
from ..exceptions import (
    OrchestratorError,
    PipelineContextInitializationError,
    UsageLimitExceededError,
    PipelineAbortSignal,
    PausedException,
    MissingAgentError,
    TypeMismatchError,
    ContextInheritanceError,
)
from ..domain.pipeline_dsl import (
    Pipeline,
    Step,
    LoopStep,
    ConditionalStep,
    ParallelStep,
    MergeStrategy,
    BranchFailureStrategy,
    HumanInTheLoopStep,
    BranchKey,
)
from ..domain.plugins import PluginOutcome
from ..domain.validation import ValidationResult
from ..domain.models import (
    BaseModel,
    PipelineResult,
    StepResult,
    UsageLimits,
    PipelineContext,
    HumanInteraction,
)
from pydantic import BaseModel as PydanticBaseModel
from ..domain.commands import AgentCommand, ExecutedCommandLog
from pydantic import TypeAdapter
from ..domain.resources import AppResources
from ..domain.types import HookCallable
from ..domain.events import (
    HookPayload,
    PreRunPayload,
    PostRunPayload,
    PreStepPayload,
    PostStepPayload,
    OnStepFailurePayload,
)
from ..domain.backends import ExecutionBackend, StepExecutionRequest
from ..tracing import ConsoleTracer

_agent_command_adapter: TypeAdapter[AgentCommand] = TypeAdapter(AgentCommand)


class InfiniteRedirectError(OrchestratorError):
    """Raised when a redirect loop is detected."""


_accepts_param_cache_weak: "weakref.WeakKeyDictionary[Callable[..., Any], Dict[str, Optional[bool]]]" = weakref.WeakKeyDictionary()
_accepts_param_cache_id: weakref.WeakValueDictionary[int, Dict[str, Optional[bool]]] = (
    weakref.WeakValueDictionary()
)


def _get_validation_flags(step: Step[Any, Any]) -> tuple[bool, bool]:
    """Return (is_validation_step, is_strict) flags from step metadata."""
    is_validation_step = bool(step.meta.get("is_validation_step", False))
    is_strict = bool(step.meta.get("strict_validation", False)) if is_validation_step else False
    return is_validation_step, is_strict


def _apply_validation_metadata(
    result: StepResult, *, validation_failed: bool, is_validation_step: bool, is_strict: bool
) -> None:
    """Set result metadata when non-strict validation fails."""
    if validation_failed and is_validation_step and not is_strict:
        result.metadata_ = result.metadata_ or {}
        result.metadata_["validation_passed"] = False


def _accepts_param(func: Callable[..., Any], param: str) -> Optional[bool]:
    """
    Return True if callable's signature includes `param` or `**kwargs`.
    Caches the result to avoid repeated inspection.
    """
    try:
        cache = _accepts_param_cache_weak.setdefault(func, {})
    except TypeError:  # For unhashable callables
        func_id = id(func)
        cache = _accepts_param_cache_id.setdefault(func_id, {})
    if param in cache:
        return cache[param]

    try:
        sig = inspect.signature(func)
        if param in sig.parameters:
            result = True
        elif any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            result = True
        else:
            result = False
    except (TypeError, ValueError):
        result = None  # Cannot inspect signature

    cache[param] = result
    return result


def _extract_missing_fields(cause: Any) -> list[str]:
    """Return list of missing field names from a Pydantic ValidationError."""
    missing_fields: list[str] = []
    if cause is not None and hasattr(cause, "errors"):
        for err in cause.errors():
            if err.get("type") == "missing":
                loc = err.get("loc") or []
                if isinstance(loc, (list, tuple)) and loc:
                    field = loc[0]
                    if isinstance(field, str):
                        missing_fields.append(field)
    return missing_fields


def _types_compatible(a: Any, b: Any) -> bool:
    """Return ``True`` if type ``a`` is compatible with type ``b``."""
    if a is Any or b is Any:
        return True

    origin_a, origin_b = get_origin(a), get_origin(b)

    if origin_b is Union:
        return any(_types_compatible(a, arg) for arg in get_args(b))
    if origin_a is Union:
        return all(_types_compatible(arg, b) for arg in get_args(a))

    try:
        return issubclass(a, b)
    except Exception as e:
        import logging

        logging.warning(f"_types_compatible: issubclass({a}, {b}) raised {e}")
        return False


RunnerInT = TypeVar("RunnerInT")
RunnerOutT = TypeVar("RunnerOutT")
ContextT = TypeVar("ContextT", bound=BaseModel)

StepExecutor: TypeAlias = Callable[
    [Step[Any, Any], Any, Optional[ContextT], Optional[AppResources]],
    Awaitable[StepResult],
]


async def _execute_loop_step_logic(
    loop_step: LoopStep[ContextT],
    loop_step_initial_input: Any,
    context: Optional[ContextT],
    resources: Optional[AppResources],
    *,
    step_executor: StepExecutor[ContextT],
    context_model_defined: bool,
    usage_limits: UsageLimits | None = None,
) -> StepResult:
    """Logic for executing a LoopStep without engine coupling."""
    loop_overall_result = StepResult(name=loop_step.name)

    if loop_step.initial_input_to_loop_body_mapper:
        try:
            current_body_input = loop_step.initial_input_to_loop_body_mapper(
                loop_step_initial_input, context
            )
        except Exception as e:
            logfire.error(
                f"Error in initial_input_to_loop_body_mapper for LoopStep '{loop_step.name}': {e}"
            )
            loop_overall_result.success = False
            loop_overall_result.feedback = f"Initial input mapper raised an exception: {e}"
            return loop_overall_result
    else:
        current_body_input = loop_step_initial_input

    last_successful_iteration_body_output: Any = None
    final_body_output_of_last_iteration: Any = None
    loop_exited_successfully_by_condition = False

    for i in range(1, loop_step.max_loops + 1):
        loop_overall_result.attempts = i
        logfire.info(f"LoopStep '{loop_step.name}': Starting Iteration {i}/{loop_step.max_loops}")

        iteration_succeeded_fully = True
        current_iteration_data_for_body_step = current_body_input

        with logfire.span(f"Loop '{loop_step.name}' - Iteration {i}"):
            for body_s in loop_step.loop_body_pipeline.steps:
                body_step_result_obj = await step_executor(
                    body_s,
                    current_iteration_data_for_body_step,
                    context,
                    resources,
                )

                loop_overall_result.latency_s += body_step_result_obj.latency_s
                loop_overall_result.cost_usd += getattr(body_step_result_obj, "cost_usd", 0.0)
                loop_overall_result.token_counts += getattr(body_step_result_obj, "token_counts", 0)

                if usage_limits is not None:
                    if (
                        usage_limits.total_cost_usd_limit is not None
                        and loop_overall_result.cost_usd > usage_limits.total_cost_usd_limit
                    ):
                        logfire.warn(f"Cost limit of ${usage_limits.total_cost_usd_limit} exceeded")
                        loop_overall_result.success = False
                        loop_overall_result.feedback = (
                            f"Cost limit of ${usage_limits.total_cost_usd_limit} exceeded"
                        )
                        pr: PipelineResult[ContextT] = PipelineResult(
                            step_history=[loop_overall_result],
                            total_cost_usd=loop_overall_result.cost_usd,
                        )
                        Flujo._set_final_context(pr, context)
                        raise UsageLimitExceededError(
                            loop_overall_result.feedback,
                            pr,
                        )
                    if (
                        usage_limits.total_tokens_limit is not None
                        and loop_overall_result.token_counts > usage_limits.total_tokens_limit
                    ):
                        logfire.warn(f"Token limit of {usage_limits.total_tokens_limit} exceeded")
                        loop_overall_result.success = False
                        loop_overall_result.feedback = (
                            f"Token limit of {usage_limits.total_tokens_limit} exceeded"
                        )
                        pr_tokens: PipelineResult[ContextT] = PipelineResult(
                            step_history=[loop_overall_result],
                            total_cost_usd=loop_overall_result.cost_usd,
                        )
                        Flujo._set_final_context(pr_tokens, context)
                        raise UsageLimitExceededError(
                            loop_overall_result.feedback,
                            pr_tokens,
                        )

                if not body_step_result_obj.success:
                    logfire.warn(
                        f"Body Step '{body_s.name}' in LoopStep '{loop_step.name}' (Iteration {i}) failed."
                    )
                    iteration_succeeded_fully = False
                    final_body_output_of_last_iteration = body_step_result_obj.output
                    break

                current_iteration_data_for_body_step = body_step_result_obj.output

        if iteration_succeeded_fully:
            last_successful_iteration_body_output = current_iteration_data_for_body_step
        final_body_output_of_last_iteration = current_iteration_data_for_body_step

        try:
            should_exit = loop_step.exit_condition_callable(
                final_body_output_of_last_iteration, context
            )
        except Exception as e:
            logfire.error(f"Error in exit_condition_callable for LoopStep '{loop_step.name}': {e}")
            loop_overall_result.success = False
            loop_overall_result.feedback = f"Exit condition callable raised an exception: {e}"
            break

        if should_exit:
            logfire.info(f"LoopStep '{loop_step.name}' exit condition met at iteration {i}.")
            loop_overall_result.success = iteration_succeeded_fully
            if not iteration_succeeded_fully:
                loop_overall_result.feedback = (
                    "Loop exited by condition, but last iteration body failed."
                )
            loop_exited_successfully_by_condition = True
            break

        if i < loop_step.max_loops:
            if loop_step.iteration_input_mapper:
                try:
                    current_body_input = loop_step.iteration_input_mapper(
                        final_body_output_of_last_iteration, context, i
                    )
                except Exception as e:
                    logfire.error(
                        f"Error in iteration_input_mapper for LoopStep '{loop_step.name}': {e}"
                    )
                    loop_overall_result.success = False
                    loop_overall_result.feedback = (
                        f"Iteration input mapper raised an exception: {e}"
                    )
                    break
            else:
                current_body_input = final_body_output_of_last_iteration
    else:
        logfire.warn(
            f"LoopStep '{loop_step.name}' reached max_loops ({loop_step.max_loops}) without exit condition being met."
        )
        loop_overall_result.success = False
        loop_overall_result.feedback = (
            f"Reached max_loops ({loop_step.max_loops}) without meeting exit condition."
        )

    if loop_overall_result.success and loop_exited_successfully_by_condition:
        if loop_step.loop_output_mapper:
            try:
                loop_overall_result.output = loop_step.loop_output_mapper(
                    last_successful_iteration_body_output, context
                )
            except Exception as e:
                logfire.error(f"Error in loop_output_mapper for LoopStep '{loop_step.name}': {e}")
                loop_overall_result.success = False
                loop_overall_result.feedback = f"Loop output mapper raised an exception: {e}"
                loop_overall_result.output = None
        else:
            loop_overall_result.output = last_successful_iteration_body_output
    else:
        loop_overall_result.output = final_body_output_of_last_iteration
        if not loop_overall_result.feedback:
            loop_overall_result.feedback = (
                "Loop did not complete successfully or exit condition not met positively."
            )

    return loop_overall_result


async def _execute_conditional_step_logic(
    conditional_step: ConditionalStep[ContextT],
    conditional_step_input: Any,
    context: Optional[ContextT],
    resources: Optional[AppResources],
    *,
    step_executor: StepExecutor[ContextT],
    context_model_defined: bool,
    usage_limits: UsageLimits | None = None,
) -> StepResult:
    """Logic for executing a ConditionalStep without engine coupling."""
    conditional_overall_result = StepResult(name=conditional_step.name)
    executed_branch_key: BranchKey | None = None
    branch_output: Any = None
    branch_succeeded = False

    try:
        branch_key_to_execute = conditional_step.condition_callable(conditional_step_input, context)
        logfire.info(
            f"ConditionalStep '{conditional_step.name}': Condition evaluated to branch key '{branch_key_to_execute}'."
        )
        executed_branch_key = branch_key_to_execute

        selected_branch_pipeline = conditional_step.branches.get(branch_key_to_execute)
        if selected_branch_pipeline is None:
            selected_branch_pipeline = conditional_step.default_branch_pipeline
            if selected_branch_pipeline is None:
                err_msg = f"ConditionalStep '{conditional_step.name}': No branch found for key '{branch_key_to_execute}' and no default branch defined."
                logfire.warn(err_msg)
                conditional_overall_result.success = False
                conditional_overall_result.feedback = err_msg
                return conditional_overall_result
            logfire.info(f"ConditionalStep '{conditional_step.name}': Executing default branch.")
        else:
            logfire.info(
                f"ConditionalStep '{conditional_step.name}': Executing branch for key '{branch_key_to_execute}'."
            )

        if conditional_step.branch_input_mapper:
            input_for_branch = conditional_step.branch_input_mapper(conditional_step_input, context)
        else:
            input_for_branch = conditional_step_input

        current_branch_data = input_for_branch
        branch_pipeline_failed_internally = False

        for branch_s in selected_branch_pipeline.steps:
            with logfire.span(
                f"ConditionalStep '{conditional_step.name}' Branch '{branch_key_to_execute}' - Step '{branch_s.name}'"
            ) as span:
                if executed_branch_key is not None:
                    try:
                        span.set_attribute("executed_branch_key", str(executed_branch_key))
                    except Exception as e:  # pragma: no cover - defensive
                        logfire.error(f"Error setting span attribute: {e}")
                branch_step_result_obj = await step_executor(
                    branch_s,
                    current_branch_data,
                    context,
                    resources,
                )

            conditional_overall_result.latency_s += branch_step_result_obj.latency_s
            conditional_overall_result.cost_usd += getattr(branch_step_result_obj, "cost_usd", 0.0)
            conditional_overall_result.token_counts += getattr(
                branch_step_result_obj, "token_counts", 0
            )

            if not branch_step_result_obj.success:
                logfire.warn(
                    f"Step '{branch_s.name}' in branch '{branch_key_to_execute}' of ConditionalStep '{conditional_step.name}' failed."
                )
                branch_pipeline_failed_internally = True
                branch_output = branch_step_result_obj.output
                conditional_overall_result.feedback = f"Failure in branch '{branch_key_to_execute}', step '{branch_s.name}': {branch_step_result_obj.feedback}"
                break

            current_branch_data = branch_step_result_obj.output

        if not branch_pipeline_failed_internally:
            branch_output = current_branch_data
            branch_succeeded = True

    except Exception as e:
        logfire.error(
            f"Error during ConditionalStep '{conditional_step.name}' execution: {e}",
            exc_info=True,
        )
        conditional_overall_result.success = False
        conditional_overall_result.feedback = f"Error executing conditional logic or branch: {e}"
        return conditional_overall_result

    conditional_overall_result.success = branch_succeeded
    if branch_succeeded:
        if conditional_step.branch_output_mapper:
            try:
                conditional_overall_result.output = conditional_step.branch_output_mapper(
                    branch_output, executed_branch_key, context
                )
            except Exception as e:
                logfire.error(
                    f"Error in branch_output_mapper for ConditionalStep '{conditional_step.name}': {e}"
                )
                conditional_overall_result.success = False
                conditional_overall_result.feedback = (
                    f"Branch output mapper raised an exception: {e}"
                )
                conditional_overall_result.output = None
        else:
            conditional_overall_result.output = branch_output
    else:
        conditional_overall_result.output = branch_output

    conditional_overall_result.attempts = 1
    if executed_branch_key is not None:
        conditional_overall_result.metadata_ = conditional_overall_result.metadata_ or {}
        conditional_overall_result.metadata_["executed_branch_key"] = str(executed_branch_key)

    return conditional_overall_result


async def _execute_parallel_step_logic(
    parallel_step: ParallelStep[ContextT],
    parallel_input: Any,
    context: Optional[ContextT],
    resources: Optional[AppResources],
    *,
    step_executor: StepExecutor[ContextT],
    context_model_defined: bool,
    usage_limits: UsageLimits | None = None,
) -> StepResult:
    """Execute all branch pipelines concurrently and aggregate results."""

    result = StepResult(name=parallel_step.name)
    outputs: Dict[str, Any] = {}
    branch_results: Dict[str, StepResult] = {}

    # Shared state for proactive cancellation
    limit_breached = asyncio.Event()
    limit_breach_error: Optional[UsageLimitExceededError] = None

    async def run_branch(key: str, branch_pipe: Pipeline[Any, Any]) -> None:
        nonlocal limit_breach_error

        # Optimized context copying strategy
        if context is not None:
            if parallel_step.context_include_keys is not None:
                # Create a new context with only the specified fields
                branch_context_data = {}
                for field_key in parallel_step.context_include_keys:
                    if hasattr(context, field_key):
                        branch_context_data[field_key] = getattr(context, field_key)
                # Deepcopy only the selected data to maintain isolation
                ctx_copy = context.__class__(**copy.deepcopy(branch_context_data))
            else:
                # Fallback to the original safe behavior
                ctx_copy = copy.deepcopy(context)
        else:
            ctx_copy = None

        current = parallel_input
        branch_res = StepResult(name=f"{parallel_step.name}:{key}")

        try:
            for s in branch_pipe.steps:
                # Check if limit has been breached by another branch
                if limit_breached.is_set():
                    logfire.info(f"Branch '{key}' cancelled due to limit breach in sibling branch")
                    branch_res.success = False
                    branch_res.feedback = "Cancelled due to usage limit breach in sibling branch"
                    break

                sr = await step_executor(s, current, ctx_copy, resources)
                branch_res.latency_s += sr.latency_s
                branch_res.cost_usd += getattr(sr, "cost_usd", 0.0)
                branch_res.token_counts += getattr(sr, "token_counts", 0)
                branch_res.attempts += sr.attempts

                # Check usage limits after each step within the branch
                if usage_limits is not None:
                    # Calculate cumulative usage across all branches
                    total_cost = branch_res.cost_usd
                    total_tokens = branch_res.token_counts

                    # Add usage from other completed branches
                    for other_key, other_br in branch_results.items():
                        if other_key != key:
                            total_cost += other_br.cost_usd
                            total_tokens += other_br.token_counts

                    # Check if limits are breached
                    if (
                        usage_limits.total_cost_usd_limit is not None
                        and total_cost > usage_limits.total_cost_usd_limit
                    ):
                        limit_breach_error = UsageLimitExceededError(
                            f"Cost limit of ${usage_limits.total_cost_usd_limit} exceeded",
                            PipelineResult(
                                step_history=[result],
                                total_cost_usd=total_cost,
                            ),
                        )
                        limit_breached.set()
                        break

                    if (
                        usage_limits.total_tokens_limit is not None
                        and total_tokens > usage_limits.total_tokens_limit
                    ):
                        limit_breach_error = UsageLimitExceededError(
                            f"Token limit of {usage_limits.total_tokens_limit} exceeded",
                            PipelineResult(
                                step_history=[result],
                                total_cost_usd=total_cost,
                            ),
                        )
                        limit_breached.set()
                        break

                if not sr.success:
                    branch_res.success = False
                    branch_res.feedback = sr.feedback
                    branch_res.output = sr.output
                    break
                current = sr.output
            else:
                branch_res.success = True
                branch_res.output = current

        except Exception as e:
            logfire.error(f"Error in branch '{key}': {e}")
            branch_res.success = False
            branch_res.feedback = f"Branch execution error: {e}"
            branch_res.output = None

        branch_res.branch_context = ctx_copy

        outputs[key] = branch_res.output
        branch_results[key] = branch_res

    start = time.monotonic()

    # Create tasks for all branches
    tasks = {
        asyncio.create_task(run_branch(k, pipe)): k for k, pipe in parallel_step.branches.items()
    }

    # Monitor tasks for completion or cancellation
    while tasks:
        done, pending = await asyncio.wait(tasks.keys(), return_when=asyncio.FIRST_COMPLETED)

        # Check if any completed task was due to a limit breach
        if limit_breached.is_set():
            logfire.info("Usage limit breached, cancelling remaining tasks...")
            for task in pending:
                task.cancel()

            # Wait for cancellations to complete
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

            # Re-raise the usage limit error
            if limit_breach_error is not None:
                Flujo._set_final_context(limit_breach_error.result, context)
                raise limit_breach_error
            break

        # Process completed tasks
        for task in done:
            try:
                await task  # This will re-raise any exception from the task
            except Exception as e:
                logfire.error(f"Task failed: {e}")
            tasks.pop(task)

    result.latency_s = time.monotonic() - start

    # Aggregate usage metrics
    for br in branch_results.values():
        result.cost_usd += br.cost_usd
        result.token_counts += br.token_counts

    succeeded_branches: Dict[str, StepResult] = {}
    failed_branches: Dict[str, StepResult] = {}
    for name, br in branch_results.items():
        if br.success:
            succeeded_branches[name] = br
        else:
            failed_branches[name] = br

    # --- 1. Handle Failures ---
    if failed_branches and parallel_step.on_branch_failure == BranchFailureStrategy.PROPAGATE:
        result.success = False
        fail_name = next(iter(failed_branches))
        result.feedback = f"Branch '{fail_name}' failed. Propagating failure."
        result.output = {
            **{k: v.output for k, v in succeeded_branches.items()},
            **{k: v for k, v in failed_branches.items()},
        }
        result.attempts = 1
        return result

    # --- 2. Handle Merging for Successful Branches ---
    if parallel_step.merge_strategy != MergeStrategy.NO_MERGE and context is not None:
        if callable(parallel_step.merge_strategy):
            for res in succeeded_branches.values():
                if res.branch_context is not None:
                    parallel_step.merge_strategy(context, res.branch_context)
        elif parallel_step.merge_strategy == MergeStrategy.OVERWRITE:
            if succeeded_branches:
                last_ctx = list(succeeded_branches.values())[-1].branch_context
                if last_ctx is not None:
                    context.__dict__.update(last_ctx.__dict__)
        elif parallel_step.merge_strategy == MergeStrategy.MERGE_SCRATCHPAD:
            if hasattr(context, "scratchpad"):
                for res in succeeded_branches.values():
                    bc = res.branch_context
                    if bc is not None and hasattr(bc, "scratchpad"):
                        context.scratchpad.update(bc.scratchpad)

    # --- 3. Finalize the Result ---
    result.success = True
    final_output = {k: v.output for k, v in succeeded_branches.items()}
    if parallel_step.on_branch_failure == BranchFailureStrategy.IGNORE:
        final_output.update(failed_branches)

    result.output = final_output
    result.attempts = 1

    # Final usage limit check (for backward compatibility and safety)
    if usage_limits is not None:
        if (
            usage_limits.total_cost_usd_limit is not None
            and result.cost_usd > usage_limits.total_cost_usd_limit
        ):
            result.success = False
            result.feedback = f"Cost limit of ${usage_limits.total_cost_usd_limit} exceeded"
            pr_cost: PipelineResult[ContextT] = PipelineResult(
                step_history=[result],
                total_cost_usd=result.cost_usd,
            )
            Flujo._set_final_context(pr_cost, context)
            raise UsageLimitExceededError(result.feedback, pr_cost)
        if (
            usage_limits.total_tokens_limit is not None
            and result.token_counts > usage_limits.total_tokens_limit
        ):
            result.success = False
            result.feedback = f"Token limit of {usage_limits.total_tokens_limit} exceeded"
            pr_tokens: PipelineResult[ContextT] = PipelineResult(
                step_history=[result],
                total_cost_usd=result.cost_usd,
            )
            Flujo._set_final_context(pr_tokens, context)
            raise UsageLimitExceededError(result.feedback, pr_tokens)

    return result


async def _run_step_logic(
    step: Step[Any, Any],
    data: Any,
    context: Optional[ContextT],
    resources: Optional[AppResources],
    *,
    step_executor: StepExecutor[ContextT],
    context_model_defined: bool,
    usage_limits: UsageLimits | None = None,
) -> StepResult:
    """Core logic for executing a single step without engine coupling."""
    visited: set[Any] = set()
    if isinstance(step, LoopStep):
        return await _execute_loop_step_logic(
            step,
            data,
            context,
            resources,
            step_executor=step_executor,
            context_model_defined=context_model_defined,
            usage_limits=usage_limits,
        )
    if isinstance(step, ConditionalStep):
        return await _execute_conditional_step_logic(
            step,
            data,
            context,
            resources,
            step_executor=step_executor,
            context_model_defined=context_model_defined,
            usage_limits=usage_limits,
        )
    if isinstance(step, ParallelStep):
        return await _execute_parallel_step_logic(
            step,
            data,
            context,
            resources,
            step_executor=step_executor,
            context_model_defined=context_model_defined,
            usage_limits=usage_limits,
        )
    if isinstance(step, HumanInTheLoopStep):
        message = step.message_for_user if step.message_for_user is not None else str(data)
        if isinstance(context, PipelineContext):
            context.scratchpad["status"] = "paused"
        raise PausedException(message)

    result = StepResult(name=step.name)
    original_agent = step.agent
    current_agent = original_agent
    last_feedback = None
    last_raw_output = None
    last_unpacked_output = None
    validation_failed = False
    last_attempt_feedbacks: list[str] = []
    last_attempt_output = None
    for attempt in range(1, step.config.max_retries + 1):
        validation_failed = False
        result.attempts = attempt
        feedbacks: list[str] = []  # feedbacks for this attempt only
        plugin_failed_this_attempt = False  # Always initialize at start of attempt
        if current_agent is None:
            raise MissingAgentError(
                f"Step '{step.name}' is missing an agent. Assign one via `Step('name', agent=...)` "
                "or by using a step factory like `@step` or `Step.from_callable()`."
            )

        start = time.monotonic()
        agent_kwargs: Dict[str, Any] = {}
        # Apply prompt processors
        if step.processors.prompt_processors:
            logfire.info(
                f"Running {len(step.processors.prompt_processors)} prompt processors for step '{step.name}'..."
            )
            processed = data
            for proc in step.processors.prompt_processors:
                try:
                    processed = await proc.process(processed, context)
                except Exception as e:  # pragma: no cover - defensive
                    logfire.error(f"Processor {proc.name} failed: {e}")
                data = processed
        from ..signature_tools import analyze_signature

        target = getattr(current_agent, "_agent", current_agent)
        func = getattr(target, "_step_callable", target.run)
        spec = analyze_signature(func)

        if spec.needs_context:
            if context is None:
                raise TypeError(
                    f"Component in step '{step.name}' requires a context, but no context model was provided to the Flujo runner."
                )
            agent_kwargs["context"] = context

        if resources is not None:
            if spec.needs_resources:
                agent_kwargs["resources"] = resources
            elif _accepts_param(func, "resources"):
                agent_kwargs["resources"] = resources
        if step.config.temperature is not None and _accepts_param(func, "temperature"):
            agent_kwargs["temperature"] = step.config.temperature
        raw_output = await current_agent.run(data, **agent_kwargs)
        result.latency_s += time.monotonic() - start
        last_raw_output = raw_output

        if isinstance(raw_output, Mock):
            raise TypeError(
                f"Step '{step.name}' returned a Mock object. This is usually due to "
                "an unconfigured mock in a test. Please configure your mock agent "
                "to return a concrete value."
            )

        unpacked_output = getattr(raw_output, "output", raw_output)
        # Apply output processors
        if step.processors.output_processors:
            logfire.info(
                f"Running {len(step.processors.output_processors)} output processors for step '{step.name}'..."
            )
            processed = unpacked_output
            for proc in step.processors.output_processors:
                try:
                    processed = await proc.process(processed, context)
                except Exception as e:  # pragma: no cover - defensive
                    logfire.error(f"Processor {proc.name} failed: {e}")
                unpacked_output = processed
        last_unpacked_output = unpacked_output

        success = True
        redirect_to = None
        final_plugin_outcome: PluginOutcome | None = None
        is_validation_step, is_strict = _get_validation_flags(step)

        sorted_plugins = sorted(step.plugins, key=lambda p: p[1], reverse=True)
        for plugin, _ in sorted_plugins:
            try:
                from ..signature_tools import analyze_signature

                plugin_kwargs: Dict[str, Any] = {}
                func = getattr(plugin, "_plugin_callable", plugin.validate)
                spec = analyze_signature(func)

                if spec.needs_context:
                    if context is None:
                        raise TypeError(
                            f"Plugin in step '{step.name}' requires a context, but no context model was provided to the Flujo runner."
                        )
                    plugin_kwargs["context"] = context

                if resources is not None:
                    if spec.needs_resources:
                        plugin_kwargs["resources"] = resources
                    elif _accepts_param(func, "resources"):
                        plugin_kwargs["resources"] = resources
                validated = await asyncio.wait_for(
                    plugin.validate(
                        {"output": last_unpacked_output, "feedback": last_feedback},
                        **plugin_kwargs,
                    ),
                    timeout=step.config.timeout_s,
                )
            except asyncio.TimeoutError as e:
                raise TimeoutError(f"Plugin timeout in step {step.name}") from e

            if not validated.success:
                validation_failed = True
                plugin_failed_this_attempt = True
                if validated.feedback:
                    feedbacks.append(validated.feedback)
                redirect_to = validated.redirect_to
                final_plugin_outcome = validated
            if validated.new_solution is not None:
                final_plugin_outcome = validated

        if final_plugin_outcome and final_plugin_outcome.new_solution is not None:
            unpacked_output = final_plugin_outcome.new_solution
            last_unpacked_output = unpacked_output

        # Run programmatic validators regardless of plugin outcome
        if step.validators:
            logfire.info(
                f"Running {len(step.validators)} programmatic validators for step '{step.name}'..."
            )
            validation_tasks = [
                validator.validate(unpacked_output, context=context)
                for validator in step.validators
            ]
            try:
                validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
            except Exception as e:  # pragma: no cover - defensive
                validation_results = [e]

            failed_checks_feedback: list[str] = []
            collected_results: list[ValidationResult] = []
            for validator, res in zip(step.validators, validation_results):
                if isinstance(res, Exception):
                    vname = getattr(
                        validator,
                        "name",
                        getattr(validator, "__class__", type(validator)).__name__,
                    )
                    failed_checks_feedback.append(f"Validator '{vname}' crashed: {res}")
                    continue
                vres = cast(ValidationResult, res)
                collected_results.append(vres)
                if not vres.is_valid:
                    fb = vres.feedback or "No details provided."
                    failed_checks_feedback.append(f"Check '{vres.validator_name}' failed: {fb}")

            if step.persist_validation_results_to and context is not None:
                if hasattr(context, step.persist_validation_results_to):
                    history_list = getattr(context, step.persist_validation_results_to)
                    if isinstance(history_list, list):
                        history_list.extend(collected_results)

            if failed_checks_feedback:
                validation_failed = True
                feedbacks.extend(failed_checks_feedback)
                if is_strict or not is_validation_step:
                    success = False

        # --- RETRY LOGIC FIX ---
        if plugin_failed_this_attempt:
            success = False
        # --- END FIX ---
        # --- JOIN ALL FEEDBACKS ---
        feedback = "\n".join(feedbacks).strip() if feedbacks else None
        # --- END JOIN ---
        if not success and attempt == step.config.max_retries:
            last_attempt_feedbacks = feedbacks.copy()
            last_attempt_output = last_unpacked_output
        if success:
            result.output = unpacked_output
            result.success = True
            result.feedback = feedback
            result.token_counts += getattr(raw_output, "token_counts", 0)
            result.cost_usd += getattr(raw_output, "cost_usd", 0.0)
            _apply_validation_metadata(
                result,
                validation_failed=validation_failed,
                is_validation_step=is_validation_step,
                is_strict=is_strict,
            )
            return result

        for handler in step.failure_handlers:
            handler()

        if redirect_to:
            if hasattr(redirect_to, "__hash__") and redirect_to.__hash__ is not None:
                if redirect_to in visited:
                    raise InfiniteRedirectError(f"Redirect loop detected in step {step.name}")
                visited.add(redirect_to)
            current_agent = redirect_to
        else:
            current_agent = original_agent

        if feedback:
            if isinstance(data, dict):
                data["feedback"] = data.get("feedback", "") + "\n" + feedback
            else:
                data = f"{str(data)}\n{feedback}"
        last_feedback = feedback

    # After all retries, set feedback to last attempt's feedbacks
    result.success = False
    result.feedback = (
        "\n".join(last_attempt_feedbacks).strip() if last_attempt_feedbacks else last_feedback
    )
    is_validation_step, is_strict = _get_validation_flags(step)
    if validation_failed and is_strict:
        result.output = None
    else:
        result.output = last_attempt_output
    result.token_counts += (
        getattr(last_raw_output, "token_counts", 1) if last_raw_output is not None else 0
    )
    result.cost_usd += (
        getattr(last_raw_output, "cost_usd", 0.0) if last_raw_output is not None else 0.0
    )
    _apply_validation_metadata(
        result,
        validation_failed=validation_failed,
        is_validation_step=is_validation_step,
        is_strict=is_strict,
    )
    if not result.success and step.persist_feedback_to_context:
        if context is not None and hasattr(context, step.persist_feedback_to_context):
            history_list = getattr(context, step.persist_feedback_to_context)
            if isinstance(history_list, list) and result.feedback:
                history_list.append(result.feedback)
    return result


class Flujo(Generic[RunnerInT, RunnerOutT, ContextT]):
    """Execute a pipeline sequentially."""

    def __init__(
        self,
        pipeline: Pipeline[RunnerInT, RunnerOutT] | Step[RunnerInT, RunnerOutT],
        context_model: Optional[Type[ContextT]] = None,
        initial_context_data: Optional[Dict[str, Any]] = None,
        resources: Optional[AppResources] = None,
        usage_limits: Optional[UsageLimits] = None,
        hooks: Optional[list[HookCallable]] = None,
        backend: Optional[ExecutionBackend] = None,
        local_tracer: Union[str, "ConsoleTracer", None] = None,
    ) -> None:
        if isinstance(pipeline, Step):
            pipeline = Pipeline.from_step(pipeline)
        self.pipeline: Pipeline[RunnerInT, RunnerOutT] = pipeline
        self.context_model = context_model
        self.initial_context_data: Dict[str, Any] = initial_context_data or {}
        self.resources = resources
        self.usage_limits = usage_limits
        self.hooks = hooks or []
        tracer_instance = None
        if isinstance(local_tracer, ConsoleTracer):
            tracer_instance = local_tracer
        elif local_tracer == "default":
            tracer_instance = ConsoleTracer()
        if tracer_instance:
            self.hooks.append(tracer_instance.hook)
        if backend is None:
            from ..infra.backends import LocalBackend

            backend = LocalBackend()
        self.backend = backend

    async def _dispatch_hook(self, event_name: str, **kwargs: Any) -> None:
        payload_map: dict[str, type[HookPayload]] = {
            "pre_run": PreRunPayload,
            "post_run": PostRunPayload,
            "pre_step": PreStepPayload,
            "post_step": PostStepPayload,
            "on_step_failure": OnStepFailurePayload,
        }
        PayloadCls = payload_map.get(event_name)
        if PayloadCls is None:
            return

        payload = PayloadCls(event_name=cast(Any, event_name), **kwargs)

        for hook in self.hooks:
            try:
                should_call = True
                try:
                    sig = inspect.signature(hook)
                    params = list(sig.parameters.values())
                    if params:
                        hints = get_type_hints(hook)
                        ann = hints.get(params[0].name, params[0].annotation)
                        if ann is not inspect.Signature.empty:
                            origin = get_origin(ann)
                            if origin is Union:
                                if not any(isinstance(payload, t) for t in get_args(ann)):
                                    should_call = False
                            elif isinstance(ann, type):
                                if not isinstance(payload, ann):
                                    should_call = False
                except Exception as e:
                    name = getattr(hook, "__name__", str(hook))
                    logfire.error(f"Error in hook '{name}': {e}")

                if should_call:
                    await hook(payload)
            except PipelineAbortSignal:
                raise
            except Exception as e:
                name = getattr(hook, "__name__", str(hook))
                logfire.error(f"Error in hook '{name}': {e}")

    async def _run_step(
        self,
        step: Step[Any, Any],
        data: Any,
        context: Optional[ContextT],
        resources: Optional[AppResources],
    ) -> StepResult:
        request = StepExecutionRequest(
            step=step,
            input_data=data,
            context=context,
            resources=resources,
            context_model_defined=self.context_model is not None,
            usage_limits=self.usage_limits,
        )
        result = await self.backend.execute_step(request)
        if getattr(step, "updates_context", False):
            if self.context_model is not None and context is not None:
                if isinstance(result.output, (BaseModel, PydanticBaseModel)):
                    update_data = result.output.model_dump(exclude_unset=True)
                elif isinstance(result.output, dict):
                    update_data = result.output
                else:
                    logfire.warn(
                        f"Step '{step.name}' has updates_context=True but did not return a dict or Pydantic model. "
                        "Skipping context update."
                    )
                    return result

                try:
                    original_data = context.model_dump()
                    for key, value in update_data.items():
                        setattr(context, key, value)

                    validated = self.context_model.model_validate(context.model_dump())
                    context.__dict__.update(validated.__dict__)
                except ValidationError as e:
                    for key, value in original_data.items():
                        setattr(context, key, value)
                    error_msg = (
                        f"Context update by step '{step.name}' failed Pydantic validation: {e}"
                    )
                    logfire.error(error_msg)
                    result.success = False
                    result.feedback = error_msg
                    return result

                logfire.info(
                    f"Context successfully updated and re-validated by step '{step.name}'."
                )
        return result

    def _check_usage_limits(
        self,
        pipeline_result: PipelineResult[ContextT],
        span: Any | None,
    ) -> None:
        if self.usage_limits is None:
            return

        total_tokens = sum(sr.token_counts for sr in pipeline_result.step_history)

        if (
            self.usage_limits.total_cost_usd_limit is not None
            and pipeline_result.total_cost_usd > self.usage_limits.total_cost_usd_limit
        ):
            if span is not None:
                try:
                    span.set_attribute("governor_breached", True)
                except Exception as e:
                    # Defensive: log and ignore errors setting span attributes
                    logfire.error(f"Error setting span attribute: {e}")
                logfire.warn(f"Cost limit of ${self.usage_limits.total_cost_usd_limit} exceeded")
                raise UsageLimitExceededError(
                    f"Cost limit of ${self.usage_limits.total_cost_usd_limit} exceeded",
                    pipeline_result,
                )

        if (
            self.usage_limits.total_tokens_limit is not None
            and total_tokens > self.usage_limits.total_tokens_limit
        ):
            if span is not None:
                try:
                    span.set_attribute("governor_breached", True)
                except Exception as e:
                    # Defensive: log and ignore errors setting span attributes
                    logfire.error(f"Error setting span attribute: {e}")
                logfire.warn(f"Token limit of {self.usage_limits.total_tokens_limit} exceeded")
                raise UsageLimitExceededError(
                    f"Token limit of {self.usage_limits.total_tokens_limit} exceeded",
                    pipeline_result,
                )

    @staticmethod
    def _set_final_context(result: PipelineResult[ContextT], ctx: Optional[ContextT]) -> None:
        if ctx is not None:
            result.final_pipeline_context = ctx

    async def run_async(
        self,
        initial_input: RunnerInT,
        *,
        initial_context_data: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[PipelineResult[ContextT]]:
        """Run the pipeline asynchronously.

        This method should be used when an asyncio event loop is already
        running, such as within Jupyter notebooks or async web frameworks.

        It yields any streaming output from the final step and then the final
        ``PipelineResult`` object.
        """
        current_context_instance: Optional[ContextT] = None
        if self.context_model is not None:
            try:
                context_data = {**self.initial_context_data}
                if initial_context_data:
                    context_data.update(initial_context_data)
                current_context_instance = self.context_model(**context_data)
            except ValidationError as e:
                logfire.error(
                    f"Context initialization failed for model {self.context_model.__name__}: {e}"
                )
                msg = f"Failed to initialize context with model {self.context_model.__name__} and initial data."
                if any(err.get("loc") == ("initial_prompt",) for err in e.errors()):
                    msg += " `initial_prompt` field required. Your custom context model must inherit from flujo.domain.models.PipelineContext."
                msg += f" Validation errors:\n{e}"
                raise PipelineContextInitializationError(msg) from e

        else:
            current_context_instance = cast(
                ContextT,
                PipelineContext(initial_prompt=str(initial_input)),
            )

        if isinstance(current_context_instance, PipelineContext):
            current_context_instance.scratchpad["status"] = "running"

        data: Optional[RunnerInT] = initial_input
        pipeline_result_obj: PipelineResult[ContextT] = PipelineResult()
        try:
            await self._dispatch_hook(
                "pre_run",
                initial_input=initial_input,
                context=current_context_instance,
                resources=self.resources,
            )
            for idx, step in enumerate(self.pipeline.steps):
                await self._dispatch_hook(
                    "pre_step",
                    step=step,
                    step_input=data,
                    context=current_context_instance,
                    resources=self.resources,
                )
                with logfire.span(step.name) as span:
                    try:
                        is_last = idx == len(self.pipeline.steps) - 1
                        if is_last and step.agent is not None and hasattr(step.agent, "stream"):
                            agent_kwargs: Dict[str, Any] = {}
                            target = getattr(step.agent, "_agent", step.agent)
                            if current_context_instance is not None and _accepts_param(
                                target.stream, "context"
                            ):
                                agent_kwargs["context"] = current_context_instance
                            if self.resources is not None and _accepts_param(
                                target.stream, "resources"
                            ):
                                agent_kwargs["resources"] = self.resources
                            if step.config.temperature is not None and _accepts_param(
                                target.stream, "temperature"
                            ):
                                agent_kwargs["temperature"] = step.config.temperature
                            chunks: list[Any] = []
                            start = time.monotonic()
                            try:
                                async for chunk in step.agent.stream(data, **agent_kwargs):
                                    chunks.append(chunk)
                                    yield chunk
                                latency = time.monotonic() - start
                                final_output_success: Any
                                if chunks and all(isinstance(c, str) for c in chunks):
                                    final_output_success = "".join(chunks)
                                else:
                                    final_output_success = chunks
                                step_result = StepResult(
                                    name=step.name,
                                    output=final_output_success,
                                    success=True,
                                    attempts=1,
                                    latency_s=latency,
                                )
                            except Exception as e:
                                latency = time.monotonic() - start
                                final_output_error: Any
                                if chunks and all(isinstance(c, str) for c in chunks):
                                    final_output_error = "".join(chunks)
                                else:
                                    final_output_error = chunks
                                step_result = StepResult(
                                    name=step.name,
                                    output=final_output_error,
                                    success=False,
                                    feedback=str(e),
                                    attempts=1,
                                    latency_s=latency,
                                )
                        else:
                            step_result = await self._run_step(
                                step,
                                data,
                                context=cast(
                                    Optional[ContextT],
                                    current_context_instance,
                                ),
                                resources=self.resources,
                            )
                    except PausedException as e:
                        if isinstance(current_context_instance, PipelineContext):
                            current_context_instance.scratchpad["status"] = "paused"
                            current_context_instance.scratchpad["pause_message"] = str(e)
                            scratch = current_context_instance.scratchpad
                            if "paused_step_input" not in scratch:
                                scratch["paused_step_input"] = data
                        self._set_final_context(
                            pipeline_result_obj,
                            cast(Optional[ContextT], current_context_instance),
                        )
                        break
                    if step_result.metadata_:
                        for key, value in step_result.metadata_.items():
                            try:
                                span.set_attribute(key, value)
                            except Exception as e:
                                # Defensive: log and ignore errors setting span attributes
                                logfire.error(f"Error setting span attribute: {e}")
                    pipeline_result_obj.step_history.append(step_result)
                    pipeline_result_obj.total_cost_usd += step_result.cost_usd
                    self._check_usage_limits(pipeline_result_obj, span)
                if step_result.success:
                    await self._dispatch_hook(
                        "post_step",
                        step_result=step_result,
                        context=current_context_instance,
                        resources=self.resources,
                    )
                else:
                    await self._dispatch_hook(
                        "on_step_failure",
                        step_result=step_result,
                        context=current_context_instance,
                        resources=self.resources,
                    )
                    logfire.warn(f"Step '{step.name}' failed. Halting pipeline execution.")
                    break
                step_output: Optional[RunnerInT] = step_result.output
                if idx < len(self.pipeline.steps) - 1:
                    next_step = self.pipeline.steps[idx + 1]
                    expected = getattr(next_step, "__step_input_type__", Any)
                    actual_type = type(step_output)
                    if step_output is None:
                        actual_type = type(None)
                    if not _types_compatible(actual_type, expected):
                        raise TypeMismatchError(
                            f"Type mismatch: Output of '{step.name}' (returns `{actual_type}`) "
                            f"is not compatible with '{next_step.name}' (expects `{expected}`). "
                            "For best results, use a static type checker like mypy to catch these issues before runtime."
                        )
                data = step_output
        except asyncio.CancelledError:
            logfire.info("Pipeline cancelled")
            yield pipeline_result_obj
            return
        except PipelineAbortSignal as e:
            logfire.info(str(e))
        except UsageLimitExceededError as e:
            if current_context_instance is not None:
                self._set_final_context(
                    pipeline_result_obj,
                    cast(Optional[ContextT], current_context_instance),
                )
            raise e
        finally:
            if current_context_instance is not None:
                self._set_final_context(
                    pipeline_result_obj,
                    cast(Optional[ContextT], current_context_instance),
                )
                if isinstance(current_context_instance, PipelineContext):
                    if current_context_instance.scratchpad.get("status") != "paused":
                        status = (
                            "completed"
                            if all(s.success for s in pipeline_result_obj.step_history)
                            else "failed"
                        )
                        current_context_instance.scratchpad["status"] = status
            try:
                await self._dispatch_hook(
                    "post_run",
                    pipeline_result=pipeline_result_obj,
                    context=current_context_instance,
                    resources=self.resources,
                )
            except PipelineAbortSignal as e:
                logfire.info(str(e))

        yield pipeline_result_obj
        return

    async def stream_async(
        self,
        initial_input: RunnerInT,
        *,
        initial_context_data: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Any]:
        async for item in self.run_async(initial_input, initial_context_data=initial_context_data):
            yield item

    def run(
        self,
        initial_input: RunnerInT,
        *,
        initial_context_data: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult[ContextT]:
        """Run the pipeline synchronously.

        This helper should only be called from code that is not already running
        inside an asyncio event loop.  If a running loop is detected a
        ``TypeError`` is raised instructing the user to use ``run_async``
        instead.
        """
        try:
            asyncio.get_running_loop()
            raise TypeError(
                "Flujo.run() cannot be called from a running event loop. "
                "If you are in an async environment (like Jupyter, FastAPI, or an "
                "`async def` function), you must use the `run_async()` method."
            )
        except RuntimeError:
            # No loop running, safe to proceed
            pass

        async def _consume() -> PipelineResult[ContextT]:
            result: PipelineResult[ContextT] | None = None
            async for item in self.run_async(
                initial_input, initial_context_data=initial_context_data
            ):
                result = item  # last yield is the PipelineResult
            assert result is not None
            return result

        return asyncio.run(_consume())

    async def resume_async(
        self, paused_result: PipelineResult[ContextT], human_input: Any
    ) -> PipelineResult[ContextT]:
        """Resume a paused pipeline with human input."""
        ctx: ContextT | None = paused_result.final_pipeline_context
        if ctx is None:
            raise OrchestratorError("Cannot resume pipeline without context")
        scratch = getattr(ctx, "scratchpad", {})
        if scratch.get("status") != "paused":
            raise OrchestratorError("Pipeline is not paused")
        start_idx = len(paused_result.step_history)
        if start_idx >= len(self.pipeline.steps):
            raise OrchestratorError("No steps remaining to resume")
        paused_step = self.pipeline.steps[start_idx]

        if isinstance(paused_step, HumanInTheLoopStep) and paused_step.input_schema is not None:
            human_input = paused_step.input_schema.model_validate(human_input)

        if isinstance(ctx, PipelineContext):
            ctx.hitl_history.append(
                HumanInteraction(
                    message_to_human=scratch.get("pause_message", ""),
                    human_response=human_input,
                )
            )
            ctx.scratchpad["status"] = "running"

        paused_step_result = StepResult(
            name=paused_step.name,
            output=human_input,
            success=True,
            attempts=1,
        )
        if isinstance(ctx, PipelineContext):
            pending = ctx.scratchpad.pop("paused_step_input", None)
            if pending is not None:
                try:
                    pending_cmd = _agent_command_adapter.validate_python(pending)
                except ValidationError:
                    pending_cmd = None
                if pending_cmd is not None:
                    log_entry = ExecutedCommandLog(
                        turn=len(ctx.command_log) + 1,
                        generated_command=pending_cmd,
                        execution_result=human_input,
                    )
                    ctx.command_log.append(log_entry)
        paused_result.step_history.append(paused_step_result)

        data = human_input
        for step in self.pipeline.steps[start_idx + 1 :]:
            await self._dispatch_hook(
                "pre_step",
                step=step,
                step_input=data,
                context=ctx,
                resources=self.resources,
            )
            logfire.info(f"Executing Step '{step.name}' with agent {repr(step.agent)}")
            with logfire.span(step.name) as span:
                try:
                    step_result = await self._run_step(
                        step,
                        data,
                        context=cast(Optional[ContextT], ctx),
                        resources=self.resources,
                    )
                except PausedException as e:
                    if isinstance(ctx, PipelineContext):
                        ctx.scratchpad["status"] = "paused"
                        ctx.scratchpad["pause_message"] = str(e)
                    self._set_final_context(paused_result, cast(Optional[ContextT], ctx))
                    break
                if step_result.metadata_:
                    for key, value in step_result.metadata_.items():
                        try:
                            span.set_attribute(key, value)
                        except Exception as e:
                            # Defensive: log and ignore errors setting span attributes
                            logfire.error(f"Error setting span attribute: {e}")
                paused_result.step_history.append(step_result)
                paused_result.total_cost_usd += step_result.cost_usd
                self._check_usage_limits(paused_result, span)
            if step_result.success:
                await self._dispatch_hook(
                    "post_step",
                    step_result=step_result,
                    context=ctx,
                    resources=self.resources,
                )
            else:
                await self._dispatch_hook(
                    "on_step_failure",
                    step_result=step_result,
                    context=ctx,
                    resources=self.resources,
                )
                logfire.warn(f"Step '{step.name}' failed. Halting pipeline execution.")
                break
            data = step_result.output

        if isinstance(ctx, PipelineContext):
            if ctx.scratchpad.get("status") != "paused":
                status = (
                    "completed" if all(s.success for s in paused_result.step_history) else "failed"
                )
                ctx.scratchpad["status"] = status

        self._set_final_context(paused_result, cast(Optional[ContextT], ctx))
        return paused_result

    def as_step(
        self, name: str, *, inherit_context: bool = True, **kwargs: Any
    ) -> Step[RunnerInT, PipelineResult[ContextT]]:
        """Return this ``Flujo`` runner as a composable :class:`Step`.

        Parameters
        ----------
        name:
            Name of the resulting step.
        **kwargs:
            Additional ``Step`` configuration passed to :class:`Step`.

        Returns
        -------
        Step
            Step that executes this runner when invoked inside another pipeline.
        """

        async def _runner(
            initial_input: Any,
            *,
            context: BaseModel | None = None,
            resources: AppResources | None = None,
        ) -> PipelineResult[ContextT]:
            initial_sub_context_data: Dict[str, Any] = {}
            if inherit_context and context is not None:
                initial_sub_context_data = context.model_dump()
            else:
                initial_sub_context_data = copy.deepcopy(self.initial_context_data)

            if "initial_prompt" not in initial_sub_context_data:
                initial_sub_context_data["initial_prompt"] = str(initial_input)

            try:
                sub_runner = Flujo(
                    self.pipeline,
                    context_model=self.context_model,
                    initial_context_data=initial_sub_context_data,
                    resources=resources or self.resources,
                    usage_limits=self.usage_limits,
                    hooks=self.hooks,
                    backend=self.backend,
                )
            except PipelineContextInitializationError as e:
                cause = getattr(e, "__cause__", None)
                missing_fields = _extract_missing_fields(cause)
                raise ContextInheritanceError(
                    missing_fields=missing_fields,
                    parent_context_keys=list(context.model_dump().keys()) if context else [],
                    child_model_name=self.context_model.__name__
                    if self.context_model
                    else "Unknown",
                ) from e
            final_result: PipelineResult[ContextT] | None = None
            try:
                async for item in sub_runner.run_async(initial_input):
                    final_result = item
            except PipelineContextInitializationError as e:
                cause = getattr(e, "__cause__", None)
                missing_fields = _extract_missing_fields(cause)
                raise ContextInheritanceError(
                    missing_fields=missing_fields,
                    parent_context_keys=list(context.model_dump().keys()) if context else [],
                    child_model_name=self.context_model.__name__
                    if self.context_model
                    else "Unknown",
                ) from e
            if final_result is None:
                raise OrchestratorError(
                    "Final result is None. The pipeline did not produce a valid result."
                )
            if inherit_context and context is not None and final_result.final_pipeline_context:
                context.__dict__.update(final_result.final_pipeline_context.__dict__)
            return final_result

        return Step.from_callable(_runner, name=name, **kwargs)
