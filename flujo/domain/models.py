"""Domain models for flujo."""

from typing import Any, List, Optional, Literal, Dict, TYPE_CHECKING, Generic
import orjson
from pydantic import BaseModel as PydanticBaseModel, Field, ConfigDict
from typing import ClassVar
from datetime import datetime, timezone
import uuid
from enum import Enum

from .types import ContextT

if TYPE_CHECKING:
    from .commands import ExecutedCommandLog


class BaseModel(PydanticBaseModel):
    """BaseModel for all flujo domain models, configured to use orjson."""

    model_config: ClassVar[ConfigDict] = {
        # Removed deprecated json_dumps and json_loads config keys
    }

    def model_dump_json(self, **kwargs: Any) -> str:
        """Override to use orjson for serialization."""
        data: bytes = orjson.dumps(self.model_dump(), **kwargs)
        return data.decode()

    @classmethod
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> "BaseModel":
        """Override to use orjson for deserialization."""
        data = orjson.loads(json_data)
        return cls.model_validate(
            data,
            strict=strict,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )


class Task(BaseModel):
    """Represents a task to be solved by the orchestrator."""

    prompt: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChecklistItem(BaseModel):
    """A single item in a checklist for evaluating a solution."""

    description: str = Field(..., description="The criterion to evaluate.")
    passed: Optional[bool] = Field(None, description="Whether the solution passes this criterion.")
    feedback: Optional[str] = Field(None, description="Feedback if the criterion is not met.")


class Checklist(BaseModel):
    """A checklist for evaluating a solution."""

    items: List[ChecklistItem]


class Candidate(BaseModel):
    """Represents a potential solution and its evaluation metadata."""

    solution: str
    score: float
    checklist: Optional[Checklist] = Field(
        None, description="Checklist evaluation for this candidate."
    )

    def __repr__(self) -> str:
        return (
            f"<Candidate score={self.score:.2f} solution={self.solution!r} "
            f"checklist_items={len(self.checklist.items) if self.checklist else 0}>"
        )

    def __str__(self) -> str:
        return (
            f"Candidate(score={self.score:.2f}, solution={self.solution!r}, "
            f"checklist_items={len(self.checklist.items) if self.checklist else 0})"
        )


class StepResult(BaseModel):
    """Result of executing a single pipeline step."""

    name: str
    output: Any | None = None
    success: bool = True
    attempts: int = 0
    latency_s: float = 0.0
    token_counts: int = 0
    cost_usd: float = 0.0
    feedback: str | None = None
    metadata_: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata about the step execution.",
    )


class PipelineResult(BaseModel, Generic[ContextT]):
    """Aggregated result of running a pipeline."""

    step_history: List[StepResult] = Field(default_factory=list)
    total_cost_usd: float = 0.0
    final_pipeline_context: Optional[ContextT] = Field(
        default=None,
        description=("The final state of the typed pipeline context, if configured and used."),
    )

    model_config: ClassVar[ConfigDict] = {"arbitrary_types_allowed": True}


class UsageLimits(BaseModel):
    """Defines resource consumption limits for a pipeline run."""

    total_cost_usd_limit: Optional[float] = Field(None, ge=0)
    total_tokens_limit: Optional[int] = Field(None, ge=0)


class SuggestionType(str, Enum):
    PROMPT_MODIFICATION = "prompt_modification"
    CONFIG_ADJUSTMENT = "config_adjustment"
    PIPELINE_STRUCTURE_CHANGE = "pipeline_structure_change"
    TOOL_USAGE_FIX = "tool_usage_fix"
    EVAL_CASE_REFINEMENT = "eval_case_refinement"
    NEW_EVAL_CASE = "new_eval_case"
    PLUGIN_ADJUSTMENT = "plugin_adjustment"
    OTHER = "other"


class ConfigChangeDetail(BaseModel):
    parameter_name: str
    suggested_value: str
    reasoning: Optional[str] = None


class PromptModificationDetail(BaseModel):
    modification_instruction: str


class ImprovementSuggestion(BaseModel):
    """A single suggestion from the SelfImprovementAgent."""

    target_step_name: Optional[str] = Field(
        None,
        description="The name of the pipeline step the suggestion primarily targets. Optional if suggestion is global or for an eval case.",
    )
    suggestion_type: SuggestionType = Field(
        ..., description="The general category of the suggested improvement."
    )
    failure_pattern_summary: str = Field(
        ..., description="A concise summary of the observed failure pattern."
    )
    detailed_explanation: str = Field(
        ...,
        description="A more detailed explanation of the issue and the rationale behind the suggestion.",
    )

    prompt_modification_details: Optional[PromptModificationDetail] = Field(
        None, description="Details for a prompt modification suggestion."
    )
    config_change_details: Optional[List[ConfigChangeDetail]] = Field(
        None, description="Details for one or more configuration adjustments."
    )

    example_failing_input_snippets: List[str] = Field(
        default_factory=list,
        description="Snippets of inputs from failing evaluation cases that exemplify the issue.",
    )
    suggested_new_eval_case_description: Optional[str] = Field(
        None, description="A description of a new evaluation case to consider adding."
    )

    estimated_impact: Optional[Literal["HIGH", "MEDIUM", "LOW"]] = Field(
        None, description="Estimated potential impact of implementing this suggestion."
    )
    estimated_effort_to_implement: Optional[Literal["HIGH", "MEDIUM", "LOW"]] = Field(
        None, description="Estimated effort required to implement this suggestion."
    )


class ImprovementReport(BaseModel):
    """Aggregated improvement suggestions returned by the agent."""

    suggestions: list[ImprovementSuggestion] = Field(default_factory=list)


class HumanInteraction(BaseModel):
    """Records a single human interaction in a HITL conversation."""

    message_to_human: str
    human_response: Any
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PipelineContext(BaseModel):
    """A built-in context object shared across the pipeline run."""

    run_id: str = Field(default_factory=lambda: f"run_{uuid.uuid4().hex}")
    initial_prompt: str
    scratchpad: Dict[str, Any] = Field(default_factory=dict)
    hitl_history: List[HumanInteraction] = Field(default_factory=list)
    command_log: List["ExecutedCommandLog"] = Field(
        default_factory=list,
        description="A log of commands executed by an AgenticLoop.",
    )

    model_config: ClassVar[ConfigDict] = {"arbitrary_types_allowed": True}
