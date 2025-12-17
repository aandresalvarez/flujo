from __future__ import annotations

from typing import runtime_checkable

from .parallel_policy import DefaultParallelStepExecutor
from ._shared import (  # noqa: F401
    Awaitable,
    Callable,
    ContextManager,
    Dict,
    Failure,
    List,
    Optional,
    ParallelStep,
    Paused,
    Pipeline,
    PipelineResult,
    Protocol,
    Quota,
    StepOutcome,
    StepResult,
    Step,
    Success,
    UsageLimits,
    telemetry,
    to_outcome,
)
from ..policy_registry import StepPolicy
from ..types import ExecutionFrame
from flujo.domain.dsl.dynamic_router import DynamicParallelRouterStep
from flujo.domain.models import BaseModel as DomainBaseModel

# --- Dynamic Router Step Executor policy ---


@runtime_checkable
class _ExecutorCoreLike(Protocol):
    async def execute(
        self, frame: ExecutionFrame[DomainBaseModel]
    ) -> StepOutcome[StepResult] | StepResult: ...

    def _safe_step_name(self, step: object) -> str: ...

    def _get_current_quota(self) -> Quota | None: ...


class DynamicRouterStepExecutor(Protocol):
    async def execute(
        self, core: object, frame: ExecutionFrame[DomainBaseModel]
    ) -> StepOutcome[StepResult]: ...


class DefaultDynamicRouterStepExecutor(StepPolicy[DynamicParallelRouterStep[DomainBaseModel]]):
    @property
    def handles_type(self) -> type[DynamicParallelRouterStep[DomainBaseModel]]:
        return DynamicParallelRouterStep

    async def execute(
        self, core: object, frame: ExecutionFrame[DomainBaseModel]
    ) -> StepOutcome[StepResult]:
        """Handle DynamicParallelRouterStep execution with proper branch selection and parallel delegation."""
        if not isinstance(core, _ExecutorCoreLike):
            raise TypeError(
                "DefaultDynamicRouterStepExecutor expected core with ExecutorCore interface"
            )
        router_step = frame.step
        if not isinstance(router_step, DynamicParallelRouterStep):
            raise TypeError(
                f"DefaultDynamicRouterStepExecutor received non-DynamicParallelRouterStep: {type(router_step).__name__}"
            )
        data = frame.data
        context = frame.context
        resources = frame.resources
        limits = frame.limits
        context_setter = getattr(frame, "context_setter", None)

        telemetry.logfire.debug("=== HANDLE DYNAMIC ROUTER STEP ===")
        telemetry.logfire.debug(f"Dynamic router step name: {router_step.name}")

        # Phase 1: Execute the router agent to decide which branches to run
        router_agent_step: Step[object, object] = Step(
            name=f"{router_step.name}_router", agent=router_step.router_agent
        )
        quota = core._get_current_quota()

        router_frame = ExecutionFrame(
            step=router_agent_step,
            data=data,
            context=context,
            resources=resources,
            limits=limits,
            quota=quota,
            stream=False,
            on_chunk=None,
            context_setter=(
                context_setter if context_setter is not None else (lambda _pr, _ctx: None)
            ),
        )
        router_result = await core.execute(router_frame)
        # Normalize StepOutcome to StepResult for router evaluation
        if isinstance(router_result, StepOutcome):
            if isinstance(router_result, Success):
                router_result = router_result.step_result
            elif isinstance(router_result, Failure):
                router_result = router_result.step_result or StepResult(
                    name=core._safe_step_name(router_step),
                    success=False,
                    feedback=router_result.feedback,
                )
            elif isinstance(router_result, Paused):
                return router_result
            else:
                router_result = StepResult(
                    name=core._safe_step_name(router_step),
                    success=False,
                    feedback="Unsupported outcome",
                )

        # Merge router context updates back to parent when available
        try:
            if getattr(router_result, "branch_context", None) is not None and context is not None:
                merged_ctx = ContextManager.merge(context, router_result.branch_context)
                router_result.branch_context = merged_ctx
        except Exception:
            pass

        # Handle router failure
        if not router_result.success:
            result = StepResult(
                name=core._safe_step_name(router_step),
                success=False,
                feedback=f"Router agent failed: {router_result.feedback}",
            )
            result.cost_usd = router_result.cost_usd
            result.token_counts = router_result.token_counts
            return to_outcome(result)

        # Process router output to get branch names
        selected_branch_names = router_result.output
        if isinstance(selected_branch_names, str):
            selected_branch_names = [selected_branch_names]
        if not isinstance(selected_branch_names, list):
            return to_outcome(
                StepResult(
                    name=core._safe_step_name(router_step),
                    success=False,
                    feedback=f"Router agent must return a list of branch names, got {type(selected_branch_names).__name__}",
                )
            )

        # Filter branches based on router's decision
        selected_branches = {
            name: router_step.branches[name]
            for name in selected_branch_names
            if name in router_step.branches
        }
        # Handle no selected branches
        if not selected_branches:
            return to_outcome(
                StepResult(
                    name=core._safe_step_name(router_step),
                    success=True,
                    output={},
                    cost_usd=router_result.cost_usd,
                    token_counts=router_result.token_counts,
                )
            )

        # Phase 2: Execute selected branches in parallel via policy
        temp_parallel_step: ParallelStep[DomainBaseModel] = ParallelStep(
            name=router_step.name,
            branches=selected_branches,
            merge_strategy=router_step.merge_strategy,
            on_branch_failure=router_step.on_branch_failure,
            context_include_keys=router_step.context_include_keys,
            field_mapping=router_step.field_mapping,
        )
        # Use the DefaultParallelStepExecutor policy directly instead of legacy core method
        parallel_executor = DefaultParallelStepExecutor()
        quota = core._get_current_quota()
        frame = ExecutionFrame(
            step=temp_parallel_step,
            data=data,
            context=context,
            resources=resources,
            limits=limits,
            quota=quota,
            stream=False,
            on_chunk=None,
            context_setter=context_setter or (lambda _pr, _ctx: None),
            _fallback_depth=0,
        )
        pr_any = await parallel_executor.execute(core=core, frame=frame)

        # Normalize StepOutcome from parallel policy to StepResult for router aggregation
        if isinstance(pr_any, StepOutcome):
            if isinstance(pr_any, Success):
                parallel_result = pr_any.step_result
            elif isinstance(pr_any, Failure):
                parallel_result = pr_any.step_result or StepResult(
                    name=core._safe_step_name(router_step), success=False, feedback=pr_any.feedback
                )
            elif isinstance(pr_any, Paused):
                return pr_any
            else:
                parallel_result = StepResult(
                    name=core._safe_step_name(router_step),
                    success=False,
                    feedback="Unsupported outcome",
                )
        else:
            parallel_result = pr_any

        # Add router usage metrics
        parallel_result.cost_usd += router_result.cost_usd
        parallel_result.token_counts += router_result.token_counts

        # Merge branch context into original context
        if parallel_result.branch_context is not None and context is not None:
            merged_ctx = ContextManager.merge(context, parallel_result.branch_context)
            parallel_result.branch_context = merged_ctx
            if context_setter is not None:
                try:
                    pipeline_result: PipelineResult[DomainBaseModel] = PipelineResult(
                        step_history=[parallel_result],
                        total_cost_usd=parallel_result.cost_usd,
                        total_tokens=parallel_result.token_counts,
                        final_pipeline_context=parallel_result.branch_context,
                    )
                    context_setter(pipeline_result, context)
                except Exception as e:
                    telemetry.logfire.warning(
                        f"Context setter failed for DynamicParallelRouterStep: {e}"
                    )

        # Record executed branches
        parallel_result.metadata_["executed_branches"] = selected_branch_names
        return to_outcome(parallel_result)


# --- End Dynamic Router Step Executor policy ---
