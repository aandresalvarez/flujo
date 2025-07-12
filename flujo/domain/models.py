"""Domain models for flujo."""

from typing import Any, List, Optional, Literal, Dict, TYPE_CHECKING, Generic
from pydantic import BaseModel as PydanticBaseModel, Field, ConfigDict
from typing import ClassVar
from datetime import datetime, timezone
import uuid
from enum import Enum
from types import FunctionType, MethodType, BuiltinFunctionType, BuiltinMethodType

from .types import ContextT

if TYPE_CHECKING:
    from .commands import ExecutedCommandLog


class BaseModel(PydanticBaseModel):
    """BaseModel for all flujo domain models with intelligent fallback serialization."""

    model_config: ClassVar[ConfigDict] = {
        # Removed deprecated json_dumps and json_loads config keys
        "arbitrary_types_allowed": True,
    }

    def _is_unknown_type(self, value: Any) -> bool:
        """Check if a value is an unknown type that needs special serialization."""
        if value is None:
            return False

        # Check for types that Pydantic handles natively
        if isinstance(value, (str, int, float, bool, list, dict, datetime, Enum)):
            return False

        # Check for types that need special handling
        return (
            callable(value)
            or isinstance(value, (complex, set, frozenset, bytes, memoryview))
            or (hasattr(value, "__dict__") and not hasattr(value, "model_dump"))
        )

    def model_dump(self, **kwargs: Any) -> Any:
        """Override model_dump to use robust serialization with custom type handling."""
        # Get the standard serialized data
        data = super().model_dump(**kwargs)

        # Process the data to handle unknown types using the existing utilities
        return self._process_serialized_data(data)

    def model_dump_json(self, **kwargs: Any) -> str:
        """Override model_dump_json to use robust serialization with custom type handling."""
        # Get the standard serialized data
        data = super().model_dump(**kwargs)

        # Process the data to handle unknown types using the existing utilities
        processed_data = self._process_serialized_data(data)

        # Import json here to avoid circular imports
        import json

        return json.dumps(processed_data, **kwargs)

    def _process_serialized_data(self, data: Any) -> Any:
        """Recursively process serialized data to handle unknown types."""
        if data is None:
            return None

        if isinstance(data, (str, int, float, bool)):
            return data

        if isinstance(data, list):
            return [self._process_serialized_data(item) for item in data]

        if isinstance(data, dict):
            return {
                self._process_serialized_data(k): self._process_serialized_data(v)
                for k, v in data.items()
            }

        # If we encounter an unknown type, use our own serialization logic
        if self._is_unknown_type(data):
            return self._serialize_single_unknown_type(data)

        return data

    def _serialize_unknown_types(self, value: Any) -> Any:
        """Intelligent fallback serialization for unknown types.

        This method provides backward compatibility for common types
        that users might include in their models.
        """
        # Only process unknown types, let Pydantic handle standard types
        if not self._is_unknown_type(value):
            return value

        if value is None:
            return None

        # Handle callable objects (functions, methods, etc.)
        if callable(value):
            if isinstance(
                value, (FunctionType, MethodType, BuiltinFunctionType, BuiltinMethodType)
            ):
                # For functions and methods, serialize as module.qualname
                module = getattr(value, "__module__", "<unknown>")
                qualname = getattr(value, "__qualname__", repr(value))
                return f"{module}.{qualname}"
            else:
                # For other callables, use repr
                return repr(value)

        # Handle complex numbers
        if isinstance(value, complex):
            return {"real": value.real, "imag": value.imag}

        # Handle sets and frozensets
        if isinstance(value, (set, frozenset)):
            return list(value)

        # Handle bytes
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")

        # Handle memoryview
        if isinstance(value, memoryview):
            return bytes(value).decode("utf-8", errors="replace")

        # For other types, try to get a meaningful representation
        try:
            # Try to get a dict representation if available
            if hasattr(value, "__dict__"):
                # Only serialize public attributes to avoid circular references
                dict_repr = {k: v for k, v in value.__dict__.items() if not k.startswith("_")}
                # Recursively serialize the dict to handle nested unknown types
                return self._recursively_serialize_dict(dict_repr)

            # Try to get a string representation
            return str(value)
        except Exception:
            # Last resort: use repr
            return repr(value)

    def _recursively_serialize_dict(self, obj: Any) -> Any:
        """Recursively serialize unknown types in dictionary values.

        This ensures that even nested objects with unknown types are properly
        serialized, preventing downstream serialization failures.
        """
        if obj is None:
            return None

        # Handle basic JSON-serializable types (let them pass through)
        if isinstance(obj, (str, int, float, bool)):
            return obj

        # Handle lists and tuples
        if isinstance(obj, (list, tuple)):
            return [self._recursively_serialize_dict(item) for item in obj]

        # Handle dictionaries
        if isinstance(obj, dict):
            return {
                self._recursively_serialize_dict(k): self._recursively_serialize_dict(v)
                for k, v in obj.items()
            }

        # For other types, use the same unknown type serialization logic
        if self._is_unknown_type(obj):
            return self._serialize_single_unknown_type(obj)

        # Let Pydantic handle standard types
        return obj

    def _serialize_single_unknown_type(self, value: Any) -> Any:
        """Serialize a single unknown type value."""
        if value is None:
            return None

        # Handle callable objects (functions, methods, etc.)
        if callable(value):
            if isinstance(
                value, (FunctionType, MethodType, BuiltinFunctionType, BuiltinMethodType)
            ):
                # For functions and methods, serialize as module.qualname
                module = getattr(value, "__module__", "<unknown>")
                qualname = getattr(value, "__qualname__", repr(value))
                return f"{module}.{qualname}"
            else:
                # For other callables, use repr
                return repr(value)

        # Handle datetime objects
        if isinstance(value, datetime):
            return value.isoformat()

        # Handle Enum values
        if isinstance(value, Enum):
            return value.value

        # Handle complex numbers
        if isinstance(value, complex):
            return {"real": value.real, "imag": value.imag}

        # Handle sets and frozensets
        if isinstance(value, (set, frozenset)):
            return list(value)

        # Handle bytes
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")

        # Handle memoryview
        if isinstance(value, memoryview):
            return bytes(value).decode("utf-8", errors="replace")

        # For other types, try to get a meaningful representation
        try:
            # Try to get a dict representation if available
            if hasattr(value, "__dict__"):
                # Only serialize public attributes to avoid circular references
                dict_repr = {k: v for k, v in value.__dict__.items() if not k.startswith("_")}
                # Recursively serialize the dict to handle nested unknown types
                return self._recursively_serialize_dict(dict_repr)

            # Try to get a string representation
            return str(value)
        except Exception:
            # Last resort: use repr
            return repr(value)


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
    branch_context: Any | None = Field(
        default=None,
        description="Final context object for a branch in ParallelStep.",
    )
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
        description="The final state of the context object after pipeline execution.",
    )

    model_config: ClassVar[ConfigDict] = {"arbitrary_types_allowed": True}


class RefinementCheck(BaseModel):
    """Standardized output from a critic pipeline in a refinement loop."""

    is_complete: bool
    feedback: Optional[Any] = None


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
    """Runtime context shared by all steps in a pipeline run.

    The base ``PipelineContext`` tracks essential execution metadata and is
    automatically created for every call to :meth:`Flujo.run`. Custom context
    models should inherit from this class to add application specific fields
    while retaining the built in ones.

    Attributes
    ----------
    run_id:
        Unique identifier for the pipeline run.
    initial_prompt:
        First input provided to the run. Useful for logging and telemetry.
    scratchpad:
        Free form dictionary for transient state between steps.
    hitl_history:
        Records each human interaction when using HITL steps.
    command_log:
        Stores commands executed by an :class:`~flujo.recipes.AgenticLoop`.
    """

    run_id: str = Field(default_factory=lambda: f"run_{uuid.uuid4().hex}")
    initial_prompt: str
    scratchpad: Dict[str, Any] = Field(default_factory=dict)
    hitl_history: List[HumanInteraction] = Field(default_factory=list)
    command_log: List["ExecutedCommandLog"] = Field(
        default_factory=list,
        description="A log of commands executed by an AgenticLoop.",
    )

    model_config: ClassVar[ConfigDict] = {"arbitrary_types_allowed": True}
