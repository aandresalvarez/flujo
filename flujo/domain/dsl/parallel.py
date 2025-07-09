from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
    Union,
    Self,
    TYPE_CHECKING,
    Set,
)

from pydantic import Field

from ..models import BaseModel
from .step import Step, MergeStrategy, BranchFailureStrategy, StepConfig
from .pipeline import Pipeline  # Import for runtime use in normalization
from ..processors import AgentProcessors
from ..ir import StepType, ParallelStepIR

if TYPE_CHECKING:
    from ..ir import StepIR, ParallelStepIR
    from flujo.registry import CallableRegistry

TContext = TypeVar("TContext", bound=BaseModel)

__all__ = ["ParallelStep"]


class ParallelStep(Step[Any, Any], Generic[TContext]):
    """Execute multiple branch pipelines concurrently.

    Each entry in ``branches`` is run in parallel and the outputs are returned
    as a dictionary keyed by branch name. Context fields can be selectively
    copied to branches via ``context_include_keys`` and merged back using
    ``merge_strategy``.
    """

    branches: Dict[str, Any] = Field(
        description="Mapping of branch names to pipelines to run in parallel."
    )
    context_include_keys: Optional[List[str]] = Field(
        default=None,
        description="If provided, only these top-level context fields will be copied to each branch. "
        "If None, the entire context is deep-copied (default behavior).",
    )
    merge_strategy: Union[MergeStrategy, Callable[[TContext, TContext], None], dict[str, Any]] = (
        Field(
            default=MergeStrategy.NO_MERGE,
            description="Strategy for merging successful branch contexts back into the main context.",
        )
    )
    on_branch_failure: BranchFailureStrategy = Field(
        default=BranchFailureStrategy.PROPAGATE,
        description="How the ParallelStep should behave when a branch fails.",
    )

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def model_validate(cls: type[Self], *args: Any, **kwargs: Any) -> Self:
        """Validate and normalize branches before creating the instance."""
        if args and isinstance(args[0], dict):
            branches = args[0].get("branches", {})
        else:
            branches = kwargs.get("branches", {})
        if not branches:
            raise ValueError("'branches' dictionary cannot be empty.")

        normalized: Dict[str, "Pipeline[Any, Any]"] = {}
        for key, branch in branches.items():
            if isinstance(branch, Step):
                normalized[key] = Pipeline.from_step(branch)
            else:
                normalized[key] = branch

        if args and isinstance(args[0], dict):
            args = (dict(args[0], branches=normalized),) + args[1:]
        else:
            kwargs["branches"] = normalized
        return super().model_validate(*args, **kwargs)

    def __repr__(self) -> str:
        return f"ParallelStep(name={self.name!r}, branches={list(self.branches.keys())})"

    def to_model(
        self,
        callable_registry: Optional["CallableRegistry"] = None,
        visited: Optional[Set[str]] = None,
    ) -> "StepIR":
        """Convert to IR model."""
        # Convert branches to PipelineIR
        ir_branches: Dict[str, Any] = {}
        for branch_name, branch_step in self.branches.items():
            # Convert single step to pipeline
            from .pipeline import Pipeline

            branch_pipeline: Pipeline[Any, Any] = Pipeline.model_construct(steps=[branch_step])
            ir_branches[branch_name] = branch_pipeline.to_model(callable_registry)

        # Create proper default values for required fields
        from ..ir import StepConfigIR, ProcessorIR

        default_config = StepConfigIR(max_retries=3, timeout_seconds=30, temperature=0.0)
        default_processors = ProcessorIR()

        # Handle merge strategy - convert callable to string or use default
        if callable(self.merge_strategy):
            merge_strategy_ir = MergeStrategy.NO_MERGE
        else:
            merge_strategy_ir = self.merge_strategy  # type: ignore[assignment]

        base_ir: ParallelStepIR[Any, Any] = ParallelStepIR[Any, Any](
            step_type=StepType.PARALLEL,
            name=self.name,
            agent=None,  # ParallelStep doesn't have an agent
            branches=ir_branches,
            merge_strategy=merge_strategy_ir,
            on_branch_failure=self.on_branch_failure,
            context_include_keys=self.context_include_keys,
            config=default_config,
            plugins=[],
            validators=[],
            processors=default_processors,
            persist_feedback_to_context=self.persist_feedback_to_context,
            persist_validation_results_to=self.persist_validation_results_to,
            updates_context=self.updates_context,
            meta=self.meta,
            step_uid=self.step_uid,
        )
        return base_ir

    @classmethod
    def _from_parallel_ir(
        cls,
        ir_model: "ParallelStepIR[Any, Any]",
        agent_registry: Optional[Dict[str, Any]] = None,
        callable_registry: Optional["CallableRegistry"] = None,
    ) -> "ParallelStep[Any]":
        """Create a ParallelStep from its IR representation."""
        # Rehydrate branch pipelines
        branches: Dict[str, Any] = {}
        for key, pipeline_ir in ir_model.branches.items():
            branches[key] = Pipeline.from_model(pipeline_ir, agent_registry, callable_registry)

        # Handle merge strategy
        merge_strategy: Union[
            MergeStrategy, Callable[[TContext, TContext], None], dict[str, Any]
        ] = ir_model.merge_strategy
        if isinstance(merge_strategy, dict) and "ref_id" in merge_strategy:
            if callable_registry is None:
                raise ValueError("CallableRegistry required for callable merge strategies")
            merge_strategy = callable_registry.get(merge_strategy["ref_id"])

        # Create step config
        config = StepConfig(
            max_retries=ir_model.config.max_retries,
            timeout_s=ir_model.config.timeout_seconds,
            temperature=ir_model.config.temperature,
        )

        return cls(
            name=ir_model.name,
            branches=branches,
            context_include_keys=ir_model.context_include_keys,
            merge_strategy=merge_strategy,
            on_branch_failure=ir_model.on_branch_failure,
            config=config,
            plugins=[],
            validators=[],
            processors=AgentProcessors(),
            persist_feedback_to_context=ir_model.persist_feedback_to_context,
            persist_validation_results_to=ir_model.persist_validation_results_to,
            updates_context=ir_model.updates_context,
            meta=ir_model.meta,
            step_uid=ir_model.step_uid,
        )
