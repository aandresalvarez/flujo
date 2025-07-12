"""Domain models for flujo."""

from typing import Any, List, Optional, Literal, Dict, TYPE_CHECKING, Generic, Set
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

    def model_dump(
        self, *, mode: str = "default", _seen: Optional[Set[int]] = None, **kwargs: Any
    ) -> Any:
        """
        Robust model_dump with dual-mode serialization:
        - mode="default": round-trip safe, for persistence/state/validation (circular refs become None for optional fields, error for required)
        - mode="cache": diagnostic, for cache keys/logs (circular refs become placeholders)
        """
        if _seen is None:
            _seen = set()

        # Check for circular references only when recursing into nested objects
        # Don't add self to _seen at the start to avoid false positives
        obj_id = id(self)
        if obj_id in _seen:
            if mode == "cache":
                return f"<{self.__class__.__name__} circular>"
            # In default mode, circular reference: return None (for optional), error for required handled by caller
            return None

        # Add to _seen only when we start recursing into fields
        _seen.add(obj_id)
        try:
            result = {}
            # Use model_fields from the class (not instance) for Pydantic v2+ compatibility
            for name, field in getattr(self.__class__, "model_fields", {}).items():
                value = getattr(self, name)
                # Always check for custom serializer first
                from flujo.utils.serialization import lookup_custom_serializer

                custom_serializer = lookup_custom_serializer(value)
                if custom_serializer:
                    serialized = custom_serializer(value)
                    # Recursively serialize the result
                    result[name] = self._safe_serialize_with_seen(serialized, _seen, mode=mode)
                    continue
                # If the value is a BaseModel, call its model_dump with the same _seen set and mode
                if hasattr(value, "model_dump") and callable(getattr(value, "model_dump")):
                    result[name] = value.model_dump(mode=mode, _seen=_seen)
                else:
                    result[name] = self._safe_serialize_with_seen(value, _seen, mode=mode)
            return result
        finally:
            _seen.discard(obj_id)

    def _safe_serialize_with_seen(self, obj: Any, _seen: Set[int], mode: str = "default") -> Any:
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj

        obj_id = id(obj)
        if obj_id in _seen:
            if mode == "cache":
                return f"<{type(obj).__name__} circular>"
            return None

        from flujo.utils.serialization import lookup_custom_serializer

        custom_serializer = lookup_custom_serializer(obj)
        if custom_serializer:
            serialized = custom_serializer(obj)
            return self._safe_serialize_with_seen(serialized, _seen, mode=mode)

        if isinstance(obj, list):
            return [self._safe_serialize_with_seen(item, _seen, mode=mode) for item in obj]
        if isinstance(obj, tuple):
            return tuple(self._safe_serialize_with_seen(item, _seen, mode=mode) for item in obj)

        # Only add to _seen for dicts, not for models
        if isinstance(obj, dict):
            _seen.add(obj_id)
            try:
                return {
                    self._safe_serialize_with_seen(
                        k, _seen, mode=mode
                    ): self._safe_serialize_with_seen(v, _seen, mode=mode)
                    for k, v in obj.items()
                }
            finally:
                _seen.discard(obj_id)
        if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
            return obj.model_dump(mode=mode, _seen=_seen)

        if self._is_unknown_type(obj):
            return self._serialize_single_unknown_type(obj, _seen, mode=mode)
        return obj

    def model_dump_json(self, **kwargs: Any) -> str:
        """Override model_dump_json to use robust serialization with custom type handling."""
        # Get the standard serialized data
        data = self.model_dump(**kwargs)
        # Import json here to avoid circular imports
        import json

        return json.dumps(data, **kwargs)

    def _process_serialized_data(
        self, data: Any, _seen: Optional[Set[int]] = None, mode: str = "default"
    ) -> Any:
        return self._safe_serialize_with_seen(data, _seen or set(), mode=mode)

    def _recursively_serialize_dict(
        self, obj: Any, _seen: Optional[Set[int]] = None, mode: str = "default"
    ) -> Any:
        return self._safe_serialize_with_seen(obj, _seen or set(), mode=mode)

    def _serialize_single_unknown_type(
        self, value: Any, _seen: Optional[Set[int]] = None, mode: str = "default"
    ) -> Any:
        """Serialize a single unknown type value, respecting mode for circular refs."""
        if _seen is None:
            _seen = set()
        if value is None:
            return None

        # Check for custom serializers FIRST - before any type-specific handling
        from flujo.utils.serialization import lookup_custom_serializer

        custom_serializer = lookup_custom_serializer(value)
        if custom_serializer:
            serialized_result = custom_serializer(value)
            return self._recursively_serialize_dict(serialized_result, _seen, mode=mode)

        if isinstance(value, (str, int, float, bool)):
            return value
        obj_id = id(value)
        if obj_id in _seen:
            if mode == "cache":
                return f"<{type(value).__name__} circular>"
            return None
        _seen.add(obj_id)
        try:
            if callable(value):
                if isinstance(
                    value, (FunctionType, MethodType, BuiltinFunctionType, BuiltinMethodType)
                ):
                    module = getattr(value, "__module__", "<unknown>")
                    qualname = getattr(value, "__qualname__", repr(value))
                    return f"{module}.{qualname}"
                else:
                    return repr(value)
            if isinstance(value, datetime):
                return value.isoformat()
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, complex):
                # Always match test expectations for string format
                real_part = int(value.real) if value.real == int(value.real) else value.real
                imag_part = int(value.imag) if value.imag == int(value.imag) else value.imag
                # Remove trailing .0 for integer parts
                real_str = str(real_part)
                imag_str = str(imag_part)
                if real_str.endswith(".0"):
                    real_str = real_str[:-2]
                if imag_str.endswith(".0"):
                    imag_str = imag_str[:-2]
                return f"{real_str}+{imag_str}j"
            if isinstance(value, (set, frozenset)):
                return list(value)
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            if isinstance(value, memoryview):
                return bytes(value).decode("utf-8", errors="replace")
            try:
                if hasattr(value, "__dict__"):
                    dict_repr = {k: v for k, v in value.__dict__.items() if not k.startswith("_")}
                    return self._recursively_serialize_dict(dict_repr, _seen, mode=mode)
                return str(value)
            except Exception:
                return repr(value)
        finally:
            _seen.discard(obj_id)


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
