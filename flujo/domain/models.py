"""Domain models for flujo."""

from __future__ import annotations
from typing import Any, List, Optional, Literal, Dict, Generic, TypeVar, Tuple
from threading import RLock
from pydantic import Field, ConfigDict, field_validator, PrivateAttr
from typing import ClassVar
from datetime import datetime, timezone
import uuid
from enum import Enum

from .types import ContextT
from .base_model import BaseModel

# ---------------------------------------------------------------------------
# StepOutcome algebraic data type (FSD-008)
# ---------------------------------------------------------------------------

T = TypeVar("T")


class ContextReference(BaseModel, Generic[T]):
    """
    A serializable pointer to external state.
    """

    provider_id: str
    key: str

    # Private attribute to hold runtime data.
    # Private attributes are NOT serialized by Pydantic default.
    _value: Optional[T] = PrivateAttr(default=None)

    def get(self) -> T:
        if self._value is None:
            raise ValueError("State not hydrated")
        return self._value

    def set(self, value: T) -> None:
        self._value = value


class StepOutcome(BaseModel, Generic[T]):
    """Typed, serializable outcome for a single step execution.

    Subclasses represent explicit terminal conditions a step can reach.
    This replaces exception-driven control flow for non-error states.
    """


class Success(StepOutcome[T]):
    """Successful completion with a concrete StepResult payload."""

    step_result: "StepResult"

    @field_validator("step_result", mode="before")
    @classmethod
    def _ensure_step_result(cls, v: Any) -> Any:
        """Defensively prevent construction of Success with a None payload.

        In some CI-only edge paths, adapters returned a None payload; constructing
        Success(step_result=None) raises a ValidationError. Normalize this by
        synthesizing a minimal StepResult that clearly indicates the issue,
        allowing callers to surface a meaningful failure instead of crashing.
        """
        if v is None:
            try:
                return StepResult(
                    name="<unknown>",
                    output=None,
                    success=False,
                    feedback="Missing step_result",
                )
            except Exception:
                return {"name": "<unknown>", "success": False, "feedback": "Missing step_result"}
        return v


class Failure(StepOutcome[T]):
    """Recoverable failure with partial result and feedback for callers/tests."""

    error: Any
    feedback: str | None = None
    step_result: Optional["StepResult"] = None


class Paused(StepOutcome[T]):
    """Human-in-the-loop pause. Contains message and optional token for resumption."""

    message: str
    state_token: Any | None = None


class Aborted(StepOutcome[T]):
    """Execution was intentionally aborted (e.g., circuit breaker, governance)."""

    reason: str


class Chunk(StepOutcome[T]):
    """Streaming data chunk emitted during step execution."""

    data: Any
    # Optionally link to the step name for traceability during streaming
    step_name: str | None = None


class BackgroundLaunched(StepOutcome[T]):
    """Step launched in background; execution continues immediately."""

    task_id: str
    step_name: str


__all__ = [
    "Task",
    "Candidate",
    "Checklist",
    "ChecklistItem",
    "PipelineResult",
    "StepResult",
    "UsageLimits",
    "UsageEstimate",
    "Quota",
    "ExecutedCommandLog",
    "PipelineContext",
    "HumanInteraction",
    "ConversationTurn",
    "ConversationRole",
    "BaseModel",
    "ContextReference",
]


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
    metadata_: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata about the step execution.",
    )

    @property
    def metadata(self) -> dict[str, Any]:
        """Alias for metadata_ for backward compatibility and test expectations."""
        return self.metadata_

    step_history: List["StepResult"] = Field(
        default_factory=list,
        description="History of sub-steps executed within this step.",
    )

    @field_validator("step_history", mode="before")
    @classmethod
    def _normalize_step_history(cls, v: Any) -> List["StepResult"]:
        # Accept None and coerce to empty list for backward compatibility in tests
        return [] if v is None else v


class PipelineResult(BaseModel, Generic[ContextT]):
    """Aggregated result of running a pipeline.

    For backward compatibility, this object exposes a top-level ``success`` flag
    that reflects overall pipeline status (computed by callers/runners). Some
    older tests and integrations expect ``result.success`` to exist.
    """

    step_history: List[StepResult] = Field(default_factory=list)
    total_cost_usd: float = 0.0
    total_tokens: int = 0  # Legacy field for backward compatibility
    final_pipeline_context: Optional[ContextT] = Field(
        default=None,
        description="The final state of the context object after pipeline execution.",
    )
    trace_tree: Optional[Any] = Field(
        default=None,
        description="Hierarchical trace tree (root span) for this run, if tracing is enabled.",
    )

    # Legacy top-level success indicator expected by some tests and integrations
    success: bool = True

    model_config: ClassVar[ConfigDict] = {"arbitrary_types_allowed": True}

    @property
    def status(self) -> str:
        """Best-effort status indicator for backward compatibility."""
        try:
            ctx = self.final_pipeline_context
            if ctx is not None and hasattr(ctx, "scratchpad"):
                scratch = getattr(ctx, "scratchpad")
                if isinstance(scratch, dict) and "status" in scratch:
                    return str(scratch.get("status") or "unknown")
        except Exception:
            pass
        return "completed" if self.success else "failed"

    @property
    def output(self) -> Any | None:
        """Return the output of the last step in the pipeline.

        This is a convenience property for backward compatibility with tests
        and code that expects result.output.
        """
        if not self.step_history:
            return None
        return self.step_history[-1].output


class RefinementCheck(BaseModel):
    """Standardized output from a critic pipeline in a refinement loop."""

    is_complete: bool
    feedback: Optional[Any] = None


class UsageLimits(BaseModel):
    """Defines resource consumption limits for a pipeline run."""

    total_cost_usd_limit: Optional[float] = Field(None, ge=0)
    total_tokens_limit: Optional[int] = Field(None, ge=0)


# ---------------------------------------------------------------------------
# Quota system (FSD-009)
# ---------------------------------------------------------------------------


class UsageEstimate(BaseModel):
    """Estimated resources a step intends to consume before execution."""

    cost_usd: float = 0.0
    tokens: int = 0


class Quota:
    """Thread-safe, mutable quota that enforces pre-execution reservations.

    This object is intentionally not a pydantic model to stay lightweight and
    avoid accidental serialization. It is passed by reference through frames.
    """

    __slots__ = ("_remaining_cost_usd", "_remaining_tokens", "_lock")

    def __init__(self, remaining_cost_usd: float, remaining_tokens: int) -> None:
        # Use non-negative values; infinity allowed for cost
        self._remaining_cost_usd = float(remaining_cost_usd)
        self._remaining_tokens = int(remaining_tokens)
        self._lock: RLock = RLock()

    def get_remaining(self) -> Tuple[float, int]:
        with self._lock:
            return self._remaining_cost_usd, self._remaining_tokens

    def has_sufficient_quota(self, estimate: UsageEstimate) -> bool:
        with self._lock:
            cost_ok = self._remaining_cost_usd == float("inf") or self._remaining_cost_usd >= max(
                0.0, float(estimate.cost_usd)
            )
            tokens_ok = self._remaining_tokens >= max(0, int(estimate.tokens))
            return cost_ok and tokens_ok

    def reserve(self, estimate: UsageEstimate) -> bool:
        """Atomically attempt to reserve the estimate.

        Returns True on success, False if insufficient.
        """
        cost_req = max(0.0, float(estimate.cost_usd))
        tokens_req = max(0, int(estimate.tokens))
        with self._lock:
            cost_ok = (
                self._remaining_cost_usd == float("inf") or self._remaining_cost_usd >= cost_req
            )
            tokens_ok = self._remaining_tokens >= tokens_req
            if not (cost_ok and tokens_ok):
                return False
            if self._remaining_cost_usd != float("inf"):
                self._remaining_cost_usd -= cost_req
            self._remaining_tokens -= tokens_req
            return True

    def reclaim(self, estimate: UsageEstimate, actual: UsageEstimate) -> None:
        """Atomically adjust after execution to reconcile estimate vs actual.

        - If actual < estimate, refund the difference.
        - If actual > estimate, attempt to deduct the overage if available. If
          not available, no exception is raised here; the safety guarantee comes
          from reserving conservatively up-front. Future improvements may surface
          this discrepancy for telemetry.
        """
        est_cost = max(0.0, float(estimate.cost_usd))
        act_cost = max(0.0, float(actual.cost_usd))
        est_tok = max(0, int(estimate.tokens))
        act_tok = max(0, int(actual.tokens))
        with self._lock:
            # Refund cost difference
            if self._remaining_cost_usd != float("inf"):
                delta_cost = est_cost - act_cost
                if delta_cost > 0:
                    self._remaining_cost_usd += delta_cost
                elif delta_cost < 0:
                    extra_needed = -delta_cost
                    if self._remaining_cost_usd >= extra_needed:
                        self._remaining_cost_usd -= extra_needed
                    else:
                        # Exhaust remaining; overage not fully covered
                        self._remaining_cost_usd = 0.0
            # Adjust tokens
            delta_tok = est_tok - act_tok
            if delta_tok > 0:
                self._remaining_tokens += delta_tok
            elif delta_tok < 0:
                extra_tok = -delta_tok
                if self._remaining_tokens >= extra_tok:
                    self._remaining_tokens -= extra_tok
                else:
                    self._remaining_tokens = 0

    def split(self, n: int) -> List["Quota"]:
        """Deterministically split this quota into n sub-quotas and zero this one.

        The split uses even division with remainder distributed to lower-indexed
        quotas for tokens and proportionally for cost to ensure the sum matches
        the original within floating point tolerance.
        """
        if n <= 0:
            raise ValueError("split requires n > 0")
        with self._lock:
            total_cost = self._remaining_cost_usd
            total_tokens = self._remaining_tokens
            # Prepare even splits
            base_tokens = total_tokens // n
            token_remainder = total_tokens % n
            # For cost, allow infinity to propagate: if inf, each child gets inf
            if total_cost == float("inf"):
                cost_shares = [float("inf")] * n
            else:
                base_cost = total_cost / float(n)
                cost_shares = [base_cost for _ in range(n)]
            # Build sub-quotas
            sub_quotas: List[Quota] = []
            for i in range(n):
                share_tokens = base_tokens + (1 if i < token_remainder else 0)
                share_cost = cost_shares[i]
                sub_quotas.append(Quota(share_cost, share_tokens))
            # Zero out parent
            self._remaining_cost_usd = 0.0 if total_cost != float("inf") else 0.0
            self._remaining_tokens = 0
            return sub_quotas


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


class ExecutedCommandLog(BaseModel):
    """Structured log entry for a command executed in the loop."""

    turn: int
    generated_command: Any
    execution_result: Any
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_config: ClassVar[ConfigDict] = {"arbitrary_types_allowed": True}


class ConversationRole(str, Enum):
    """Canonical roles for conversational turns.

    Using lower-case values aligns with common chat semantics and future
    compatibility with multi-provider chat message schemas.
    """

    user = "user"
    assistant = "assistant"


class ConversationTurn(BaseModel):
    """A single conversational turn captured during a run."""

    role: ConversationRole
    content: str


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
    conversation_history:
        Ordered list of conversational turns (user/assistant) maintained when
        a loop is configured with ``conversation: true``. Persisted with the
        context to support pause/resume and restarts.
    """

    run_id: str = Field(default_factory=lambda: f"run_{uuid.uuid4().hex}")
    initial_prompt: Optional[str] = None
    scratchpad: Dict[str, Any] = Field(default_factory=dict)
    hitl_history: List[HumanInteraction] = Field(default_factory=list)
    command_log: List[ExecutedCommandLog] = Field(
        default_factory=list,
        description="A log of commands executed by an agentic loop pipeline.",
    )
    conversation_history: List[ConversationTurn] = Field(
        default_factory=list,
        description=(
            "Conversation history (user/assistant turns) for conversational loops. "
            "This field is optional and empty unless conversational mode is enabled."
        ),
    )
    # Utility counter used by test hooks; kept in base context for simplicity
    call_count: int = 0

    model_config: ClassVar[ConfigDict] = {"arbitrary_types_allowed": True}

    @property
    def steps(self) -> Dict[str, Any]:
        """Expose recorded step outputs stored in scratchpad['steps']."""
        try:
            sp = getattr(self, "scratchpad", {})
            if isinstance(sp, dict):
                steps_map = sp.get("steps")
                if isinstance(steps_map, dict):
                    return steps_map
        except Exception:
            pass
        return {}
