"""
Pipeline Intermediate Representation (IR) Models.

This module defines Pydantic models for serializing and deserializing
pipeline structures. The IR provides:

1. Type-safe serialization of pipeline components
2. Platform-independent representation
3. Version control and migration support
4. Capable of representing all pipeline constructs
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    TypeVar,
    Generic,
)
from enum import Enum
from pydantic import BaseModel, Field

from .models import BaseModel as DomainBaseModel
from .dsl.step import MergeStrategy, BranchFailureStrategy

# Type variables for generic IR models
IRInT = TypeVar("IRInT")
IROutT = TypeVar("IROutT")
ContextT = TypeVar("ContextT", bound=DomainBaseModel)

# Branch key type alias
BranchKey = Any


class StepType(str, Enum):
    """Enumeration of all possible step types in the IR."""

    STANDARD = "standard"
    LOOP = "loop"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"
    MAP = "map"
    CACHE = "cache"
    HUMAN_IN_THE_LOOP = "human_in_the_loop"


class AgentReference(BaseModel):
    """Reference to an agent in the IR.

    This represents how to locate and instantiate the agent,
    but doesn't contain the actual agent object.
    """

    agent_type: str = Field(description="Type identifier for the agent")
    agent_config: Dict[str, Any] = Field(
        default_factory=dict, description="Configuration for the agent"
    )
    agent_name: Optional[str] = Field(default=None, description="Optional name for the agent")


class CallableReference(BaseModel):
    """Reference to a callable function in the IR.

    This represents how to locate and retrieve a callable function,
    but doesn't contain the actual function object.
    """

    ref_id: str = Field(description="A unique identifier for the callable")


class StepConfigIR(BaseModel):
    """IR representation of step configuration."""

    max_retries: int = Field(default=1, ge=0, description="Maximum number of retry attempts")
    timeout_seconds: Optional[float] = Field(default=None, gt=0, description="Timeout in seconds")
    temperature: Optional[float] = Field(
        default=None, ge=0, le=2, description="Temperature for LLM agents"
    )
    max_tokens: Optional[int] = Field(
        default=None, gt=0, description="Maximum tokens for LLM agents"
    )
    custom_config: Dict[str, Any] = Field(
        default_factory=dict, description="Additional custom configuration"
    )


class ValidationPluginIR(BaseModel):
    """IR representation of a validation plugin."""

    plugin_type: str = Field(description="Type identifier for the plugin")
    plugin_config: Dict[str, Any] = Field(default_factory=dict, description="Plugin configuration")
    priority: int = Field(default=0, description="Plugin execution priority")


class ValidatorIR(BaseModel):
    """IR representation of a validator."""

    validator_type: str = Field(description="Type identifier for the validator")
    validator_config: Dict[str, Any] = Field(
        default_factory=dict, description="Validator configuration"
    )


class ProcessorIR(BaseModel):
    """IR representation of processors."""

    prompt_processors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Prompt processors"
    )
    output_processors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Output processors"
    )


class BaseStepIR(BaseModel, Generic[IRInT, IROutT]):
    """Base class for all step IR models."""

    step_type: StepType = Field(description="Type of this step")
    name: str = Field(description="Unique name for this step")
    agent: Optional[AgentReference] = Field(
        default=None, description="Agent reference for this step"
    )
    config: StepConfigIR = Field(default_factory=StepConfigIR, description="Step configuration")
    plugins: List[ValidationPluginIR] = Field(
        default_factory=list, description="Validation plugins"
    )
    validators: List[ValidatorIR] = Field(default_factory=list, description="Validators")
    processors: ProcessorIR = Field(default_factory=ProcessorIR, description="Processors")
    persist_feedback_to_context: Optional[str] = Field(
        default=None, description="Context attribute to persist feedback to"
    )
    persist_validation_results_to: Optional[str] = Field(
        default=None, description="Context attribute to persist validation results to"
    )
    updates_context: bool = Field(
        default=False, description="Whether step updates pipeline context"
    )
    meta: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")
    fallback_step: Optional["BaseStepIR[Any, Any]"] = Field(
        default=None, description="Fallback step to execute if this step fails"
    )
    step_uid: str = Field(description="Globally unique step identifier")

    def __init__(self, **data: Any) -> None:
        if self.__class__ is BaseStepIR:
            raise RuntimeError(
                "BaseStepIR is abstract and cannot be instantiated directly; use a concrete IR subclass."
            )
        super().__init__(**data)

    class Config:
        arbitrary_types_allowed = True


class StandardStepIR(BaseStepIR[IRInT, IROutT]):
    """IR representation of a standard step."""

    step_uid: str = Field(description="Globally unique step identifier")


class LoopStepIR(BaseStepIR[IRInT, IROutT]):
    """IR representation of a loop step."""

    loop_body_pipeline: "PipelineIR[Any, Any]" = Field(
        description="Pipeline to execute in each iteration"
    )
    exit_condition_callable: CallableReference = Field(
        description="Exit condition callable reference"
    )
    max_loops: int = Field(default=5, gt=0, description="Maximum number of iterations")
    initial_input_mapper: Optional[CallableReference] = Field(
        default=None, description="Initial input mapping callable reference"
    )
    iteration_input_mapper: Optional[CallableReference] = Field(
        default=None, description="Iteration input mapping callable reference"
    )
    loop_output_mapper: Optional[CallableReference] = Field(
        default=None, description="Loop output mapping callable reference"
    )


class MapStepIR(BaseStepIR[IRInT, IROutT]):
    """IR representation of a map step."""

    loop_body_pipeline: "PipelineIR[Any, Any]" = Field(
        description="Pipeline to execute in each iteration"
    )
    exit_condition_callable: CallableReference = Field(
        description="Exit condition callable reference"
    )
    max_loops: int = Field(default=5, gt=0, description="Maximum number of iterations")
    initial_input_mapper: Optional[CallableReference] = Field(
        default=None, description="Initial input mapping callable reference"
    )
    iteration_input_mapper: Optional[CallableReference] = Field(
        default=None, description="Iteration input mapping callable reference"
    )
    loop_output_mapper: Optional[CallableReference] = Field(
        default=None, description="Loop output mapping callable reference"
    )
    iterable_input: str = Field(description="Context attribute containing the iterable")


class ConditionalStepIR(BaseStepIR[IRInT, IROutT]):
    """IR representation of a conditional step."""

    condition_callable: CallableReference = Field(
        description="Condition evaluation callable reference"
    )
    branches: Dict[str, "PipelineIR[Any, Any]"] = Field(
        description="Branch pipelines by condition key"
    )
    default_branch: Optional["PipelineIR[Any, Any]"] = Field(
        default=None, description="Default branch pipeline"
    )
    branch_input_mapper: Optional[CallableReference] = Field(
        default=None, description="Branch input mapping callable reference"
    )
    branch_output_mapper: Optional[CallableReference] = Field(
        default=None, description="Branch output mapping callable reference"
    )


class ParallelStepIR(BaseStepIR[IRInT, IROutT]):
    """IR representation of a parallel step."""

    branches: Dict[str, "PipelineIR[Any, Any]"] = Field(
        description="Branch pipelines to execute in parallel"
    )
    context_include_keys: Optional[List[str]] = Field(
        default=None, description="Context keys to include in branch contexts"
    )
    merge_strategy: Union[MergeStrategy, Dict[str, Any]] = Field(
        default=MergeStrategy.NO_MERGE, description="Strategy for merging branch contexts"
    )
    on_branch_failure: BranchFailureStrategy = Field(
        default=BranchFailureStrategy.PROPAGATE, description="Behavior when a branch fails"
    )


class CacheStepIR(BaseStepIR[IRInT, IROutT]):
    """IR representation of a cache step."""

    wrapped_step: "BaseStepIR[Any, Any]" = Field(description="The step to cache")
    cache_backend: Optional[Dict[str, Any]] = Field(
        default=None, description="Cache backend configuration"
    )


class HumanInTheLoopStepIR(BaseStepIR[IRInT, IROutT]):
    """IR representation of a human-in-the-loop step."""

    message_for_user: Optional[str] = Field(default=None, description="Message to display to user")
    input_schema: Optional[Dict[str, Any]] = Field(
        default=None, description="Input schema for user"
    )


# Union type for all step IR models
StepIR = Union[
    StandardStepIR[Any, Any],
    LoopStepIR[Any, Any],
    MapStepIR[Any, Any],
    ConditionalStepIR[Any, Any],
    ParallelStepIR[Any, Any],
    CacheStepIR[Any, Any],
    HumanInTheLoopStepIR[Any, Any],
]


class PipelineIR(BaseModel, Generic[IRInT, IROutT]):
    """IR representation of a complete pipeline."""

    steps: List[StepIR] = Field(description="Ordered list of steps in the pipeline")
    name: Optional[str] = Field(default=None, description="Optional name for the pipeline")
    version: Optional[str] = Field(default=None, description="Pipeline version")
    description: Optional[str] = Field(default=None, description="Pipeline description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Pipeline metadata")

    class Config:
        arbitrary_types_allowed = True


# Update forward references
PipelineIR.model_rebuild()
LoopStepIR.model_rebuild()
ConditionalStepIR.model_rebuild()
ParallelStepIR.model_rebuild()
CacheStepIR.model_rebuild()


def get_step_type(ir_step: "BaseStepIR[Any, Any]") -> StepType:
    """Get the step type for a given IR step instance."""
    if isinstance(ir_step, StandardStepIR):
        return StepType.STANDARD
    elif isinstance(ir_step, LoopStepIR):
        return StepType.LOOP
    elif isinstance(ir_step, MapStepIR):
        return StepType.MAP
    elif isinstance(ir_step, ConditionalStepIR):
        return StepType.CONDITIONAL
    elif isinstance(ir_step, ParallelStepIR):
        return StepType.PARALLEL
    elif isinstance(ir_step, CacheStepIR):
        return StepType.CACHE
    elif isinstance(ir_step, HumanInTheLoopStepIR):
        return StepType.HUMAN_IN_THE_LOOP
    else:
        raise ValueError(f"Unknown IR step type: {type(ir_step)}")


__all__ = [
    "StepType",
    "AgentReference",
    "StepConfigIR",
    "ValidationPluginIR",
    "ValidatorIR",
    "ProcessorIR",
    "BaseStepIR",
    "StandardStepIR",
    "LoopStepIR",
    "MapStepIR",
    "ConditionalStepIR",
    "ParallelStepIR",
    "CacheStepIR",
    "HumanInTheLoopStepIR",
    "StepIR",
    "PipelineIR",
    "BranchKey",
    "get_step_type",
]
