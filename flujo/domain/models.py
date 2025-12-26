"""Domain models for flujo."""

from __future__ import annotations
from typing import (
    Any,
    ClassVar,
    Generic,
    Iterator,
    ItemsView,
    KeysView,
    List,
    Literal,
    MutableMapping,
    Optional,
    Tuple,
    TypeVar,
    ValuesView,
)
from flujo.type_definitions.common import JSONObject
from threading import RLock
from pydantic import Field, ConfigDict, field_validator, model_validator, PrivateAttr
from datetime import datetime, timezone
import uuid
from enum import Enum

from .types import ContextT
from .memory import ScoredMemory, VectorStoreProtocol
from .sandbox import SandboxProtocol
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

    error: Any = None
    feedback: str | None = None
    step_result: Optional["StepResult"] = None


class Paused(StepOutcome[T]):
    """Human-in-the-loop pause. Contains message and optional token for resumption."""

    message: str
    state_token: JSONObject | None = None


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
    "SearchNode",
    "SearchState",
    "PipelineResult",
    "StepResult",
    "UsageLimits",
    "UsageEstimate",
    "Quota",
    "ExecutedCommandLog",
    "ImportArtifacts",
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


class SearchNode(BaseModel):
    """A single node in a tree search frontier."""

    node_id: str
    parent_id: str | None = None
    depth: int = 0
    candidate: Any | None = None
    output: Any | None = None
    g_cost: float = 0.0
    h_cost: float = 1.0
    f_cost: float = 1.0
    state_hash: str = ""
    evaluation: JSONObject | None = None
    context_snapshot: JSONObject | None = None
    metadata: JSONObject = Field(default_factory=dict)

    _context: Optional["PipelineContext"] = PrivateAttr(default=None)

    def attach_context(self, context: Optional["PipelineContext"]) -> None:
        """Attach runtime context and capture a snapshot for persistence."""
        self._context = context
        if context is None:
            self.context_snapshot = None
            return

        # Circular reference prevention: always exclude search state and history-heavy fields
        exclude_keys = {"tree_search_state", "step_history", "command_log", "granular_state"}
        try:
            # Prefer model_dump with explicit exclusions to break recursion
            self.context_snapshot = context.model_dump(exclude=exclude_keys, mode="json")
        except Exception:
            # Robust fallback: build dict manually to avoid hitting RecursionError
            # in model_dump() if circularities somehow exist.
            try:
                data: JSONObject = {}
                # Use model_fields for Pydantic v2 if available
                fields = getattr(context, "model_fields", None)
                if fields:
                    for field in fields:
                        if field not in exclude_keys:
                            try:
                                val: object = getattr(context, field)
                                # Basic serialization for common types to ensure JSON safety
                                if hasattr(val, "model_dump"):
                                    data[field] = val.model_dump(mode="json")
                                else:
                                    data[field] = val
                            except Exception:
                                continue
                else:
                    # Fallback to __dict__ for other objects
                    for key, val in getattr(context, "__dict__", {}).items():
                        if key not in exclude_keys and not key.startswith("_"):
                            data[key] = val
                self.context_snapshot = data
            except Exception:
                # Absolute last resort: empty snapshot rather than crashing the runtime
                self.context_snapshot = {}

    def rehydrate_context(
        self, context_type: type["PipelineContext"]
    ) -> Optional["PipelineContext"]:
        """Rebuild runtime context from the stored snapshot if needed."""
        if self._context is not None:
            return self._context
        if self.context_snapshot is None:
            return None
        try:
            ctx = context_type.model_validate(self.context_snapshot)
        except Exception:
            return None
        try:
            if hasattr(ctx, "tree_search_state"):
                setattr(ctx, "tree_search_state", None)
        except Exception:
            pass
        self._context = ctx
        return ctx


class SearchState(BaseModel):
    """Persistent tree search state stored in the pipeline context."""

    version: int = 1
    open_set: list[str] = Field(default_factory=list)
    closed_set: list[str] = Field(default_factory=list)
    nodes: dict[str, SearchNode] = Field(default_factory=dict)
    deduced_invariants: list[str] = Field(default_factory=list)
    iterations: int = 0
    expansions: int = 0
    next_node_id: int = 0
    best_node_id: str | None = None
    status: Literal["running", "paused", "completed", "failed"] = "running"
    trace: list[JSONObject] = Field(default_factory=list)
    metadata: JSONObject = Field(default_factory=dict)

    def sorted_open_nodes(self) -> list[SearchNode]:
        """Return open nodes sorted by A* priority."""
        nodes = [self.nodes[nid] for nid in self.open_set if nid in self.nodes]
        return sorted(nodes, key=lambda n: (n.f_cost, n.g_cost, n.depth, n.node_id))

    def pop_best_open(self) -> SearchNode | None:
        """Pop the best node from the open set."""
        nodes = self.sorted_open_nodes()
        if not nodes:
            return None
        best = nodes[0]
        try:
            self.open_set.remove(best.node_id)
        except ValueError:
            pass
        return best


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

    model_config: ClassVar[ConfigDict] = {"extra": "allow"}

    @property
    def status(self) -> str:
        """Best-effort status indicator for backward compatibility."""
        try:
            ctx = self.final_pipeline_context
            # Prefer typed status field.
            if ctx is not None:
                st = getattr(ctx, "status", None)
                if isinstance(st, str) and st:
                    return st
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
    feedback: str | JSONObject | None = None


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


class QuotaExceededError(Exception):
    """Raised when quota reconciliation cannot cover actual usage.

    This is an internal signal used by the quota system. Application code should translate
    it to a user-facing `UsageLimitExceededError` with a stable legacy message.
    """

    def __init__(
        self,
        *,
        remaining_cost_usd: float,
        remaining_tokens: int,
        extra_cost_usd: float,
        extra_tokens: int,
    ) -> None:
        self.remaining_cost_usd = float(remaining_cost_usd)
        self.remaining_tokens = int(remaining_tokens)
        self.extra_cost_usd = float(extra_cost_usd)
        self.extra_tokens = int(extra_tokens)
        super().__init__("Insufficient quota")


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
        # Quota methods are synchronous/non-blocking; RLock is adequate in the single-threaded
        # asyncio model we run in. If await points are added in the future, switch to asyncio.Lock.
        self._lock: RLock = RLock()

    def get_remaining(self) -> Tuple[float, int]:
        with self._lock:
            return self._remaining_cost_usd, self._remaining_tokens

    def refund(self, amount: UsageEstimate) -> None:
        """Refund capacity back into this quota."""
        add_cost = max(0.0, float(getattr(amount, "cost_usd", 0.0) or 0.0))
        add_tokens = max(0, int(getattr(amount, "tokens", 0) or 0))
        with self._lock:
            if self._remaining_cost_usd != float("inf"):
                self._remaining_cost_usd += add_cost
            self._remaining_tokens += add_tokens

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
          not available, raise UsageLimitExceededError. This keeps enforcement
          within the quota system for callers that only know actual usage after
          execution (e.g., custom step executors in tests).
        """
        est_cost = max(0.0, float(estimate.cost_usd))
        act_cost = max(0.0, float(actual.cost_usd))
        est_tok = max(0, int(estimate.tokens))
        act_tok = max(0, int(actual.tokens))
        remaining_before: tuple[float, int]
        extra_cost_usd = 0.0
        extra_tokens = 0
        with self._lock:
            remaining_before = (self._remaining_cost_usd, self._remaining_tokens)
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
                        extra_cost_usd = float(extra_needed)
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
                    extra_tokens = int(extra_tok)

        if extra_cost_usd > 0.0 or extra_tokens > 0:
            raise QuotaExceededError(
                remaining_cost_usd=float(remaining_before[0]),
                remaining_tokens=int(remaining_before[1]),
                extra_cost_usd=float(extra_cost_usd),
                extra_tokens=int(extra_tokens),
            )

    def split(self, n: int) -> List["Quota"]:
        """Split this quota into n deterministic child quotas.

        The parent quota is zeroed out and its remaining capacity is partitioned into
        child quotas. This is used by parallel execution to avoid race conditions
        where one branch can consume budget intended for others.
        """
        if n <= 0:
            raise ValueError("split requires n > 0")
        with self._lock:
            parent_cost = float(self._remaining_cost_usd)
            parent_tokens = int(self._remaining_tokens)
            # Zero parent tokens (always finite)
            self._remaining_tokens = 0

            # Split tokens evenly, distributing remainder to lower indices.
            base_tokens = parent_tokens // n
            remainder_tokens = parent_tokens % n
            token_parts = [base_tokens + (1 if i < remainder_tokens else 0) for i in range(n)]

            # Split cost evenly when finite; preserve infinity when unlimited.
            if parent_cost == float("inf"):
                cost_parts = [float("inf") for _ in range(n)]
            else:
                self._remaining_cost_usd = 0.0
                per_cost = parent_cost / float(n)
                cost_parts = [per_cost for _ in range(n)]

        return [Quota(cost_parts[i], token_parts[i]) for i in range(n)]


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
    # model_config inherited from BaseModel


class ImportArtifacts(BaseModel, MutableMapping[str, Any]):
    """Typed container for import projections to avoid scratchpad for user data."""

    result: Any | None = None
    marker: Any | None = None
    counter: int | None = None
    value: Any | None = None
    initial_input: Any | None = None
    child_echo: Any | None = None
    final_sql: Any | None = None
    concept_sets: list[Any] = Field(default_factory=list)
    cohort_definition: dict[str, Any] | None = None
    captured: Any | None = None
    captured_sp: dict[str, Any] = Field(default_factory=dict)
    captured_prompt: Any | None = None
    extras: dict[str, Any] = Field(default_factory=dict)

    model_config: ClassVar[ConfigDict] = {
        "extra": "allow",
    }

    def _data(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        fields_set: set[str] = getattr(self, "__pydantic_fields_set__", set())
        fields_def = type(self).model_fields
        for name in fields_def:
            val = getattr(self, name)
            if val is not None or name in fields_set:
                data[name] = val
        extra = getattr(self, "__pydantic_extra__", None)
        if isinstance(extra, dict):
            data.update(extra)
        data.update(self.extras)
        return data

    # MutableMapping interface
    def __getitem__(self, key: str) -> Any:
        return self._data()[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key in type(self).model_fields:
            setattr(self, key, value)
            return
        extra = getattr(self, "__pydantic_extra__", None)
        if not isinstance(extra, dict):
            extra = {}
            object.__setattr__(self, "__pydantic_extra__", extra)
        extra[key] = value
        self.extras[key] = value

    def __delitem__(self, key: str) -> None:
        if key in type(self).model_fields:
            object.__setattr__(self, key, None)
        self.extras.pop(key, None)
        extra = getattr(self, "__pydantic_extra__", None)
        if isinstance(extra, dict):
            extra.pop(key, None)

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        return iter(self._data())

    def __len__(self) -> int:
        return len(self._data())

    def get(self, key: str, default: Any = None) -> Any:
        return self._data().get(key, default)

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key in self._data():
            return self._data()[key]
        self[key] = default
        return default

    def update(self, *args: Any, **kwargs: Any) -> None:
        merged: dict[str, Any] = {}
        if args:
            merged.update(dict(*args))
        if kwargs:
            merged.update(kwargs)
        for k, v in merged.items():
            self[k] = v

    def items(self) -> ItemsView[str, Any]:
        return self._data().items()

    def keys(self) -> KeysView[str]:
        return self._data().keys()

    def values(self) -> ValuesView[Any]:
        return self._data().values()


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
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="allow",
        validate_assignment=True,
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_root_keys(cls, data: object) -> object:
        if isinstance(data, dict) and "scratchpad" in data:
            raise ValueError("'scratchpad' has been removed; use typed context fields instead.")
        return data

    run_id: str = Field(default_factory=lambda: f"run_{uuid.uuid4().hex}")
    initial_prompt: Optional[str] = None
    # Structured slot for import-related artifacts to avoid scratchpad usage.
    import_artifacts: ImportArtifacts = Field(default_factory=ImportArtifacts)
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
    # Runtime-only handle; excluded from serialization/validation.
    memory_store: VectorStoreProtocol | None = Field(default=None, exclude=True)
    _sandbox: SandboxProtocol | None = PrivateAttr(default=None)

    # -------------------------------------------------------------------------
    # Typed fields (promoted from scratchpad)
    # -------------------------------------------------------------------------
    status: Literal["running", "paused", "completed", "failed"] = Field(
        default="running",
        description="Current execution status of the pipeline.",
    )
    pause_message: str | None = Field(
        default=None,
        description="Human-readable message when pipeline is paused (HITL).",
    )
    step_outputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Step outputs keyed by step name. Replaces scratchpad['steps'].",
    )
    current_state: str | None = Field(
        default=None,
        description="Current state for state-machine pipelines.",
    )
    next_state: str | None = Field(
        default=None,
        description="Next state for state-machine pipelines.",
    )
    paused_step_input: Any | None = Field(
        default=None,
        description="Input data preserved when a step pauses for HITL.",
    )
    user_input: Any | None = Field(
        default=None,
        description="User-provided input during HITL resume.",
    )
    hitl_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional HITL metadata.",
    )
    loop_iteration_index: int | None = Field(
        default=None,
        description="Current loop iteration index.",
    )
    loop_step_index: int | None = Field(
        default=None,
        description="Current step index within the loop.",
    )
    loop_last_output: Any | None = Field(
        default=None,
        description="Output from last loop iteration.",
    )
    loop_resume_requires_hitl_output: bool = Field(
        default=False,
        description="Flag indicating if the loop resume requires HITL output.",
    )
    loop_paused_step_name: str | None = Field(
        default=None,
        description="Name of the step where the loop paused.",
    )
    granular_state: dict[str, Any] | None = Field(
        default=None,
        description="Granular execution state for GranularStep.",
    )
    tree_search_state: SearchState | None = Field(
        default=None,
        description="Persistent TreeSearchStep frontier state for crash recovery.",
    )
    # Background task bookkeeping (migrated from scratchpad)
    is_background_task: bool = Field(
        default=False,
        description="True when this context is executing inside a background task.",
    )
    task_id: str | None = Field(
        default=None,
        description="Background task id when is_background_task=True.",
    )
    parent_run_id: str | None = Field(
        default=None,
        description="Parent run id for background tasks when applicable.",
    )
    background_error: str | None = Field(
        default=None,
        description="Error message for background task failures (if any).",
    )
    background_error_category: str | None = Field(
        default=None,
        description="Optional structured category for background task failures.",
    )
    # Recipe-specific fields (Candidate/Checklist pattern)
    solution: str | None = Field(
        default=None,
        description="Current solution content (Chain of Thought pattern).",
    )
    checklist: Checklist | None = Field(
        default=None,
        description="Evaluation checklist (Self-Correction pattern).",
    )

    def get(self, key: str, default: Any = None) -> Any:  # pragma: no cover - small helper
        try:
            return getattr(self, key)
        except Exception:
            return default

    @property
    def steps(self) -> JSONObject:
        """Expose recorded step outputs."""
        return self.step_outputs

    async def retrieve(
        self,
        query_text: str | None = None,
        *,
        query_vector: list[float] | None = None,
        limit: int = 5,
    ) -> list[ScoredMemory]:
        """Retrieve memories using the configured memory store.

        If query_vector is not provided, an embedding model will be used when available.
        Returns an empty list when no memory store is configured.
        """
        store = self.memory_store
        if store is None:
            return []
        try:
            from .memory import VectorQuery
            from flujo.embeddings import get_embedding_client
            from flujo.infra.settings import get_settings
        except Exception:
            return []

        vector = query_vector
        if vector is None:
            if query_text is None:
                return []
            settings = get_settings()
            model_id = getattr(settings, "memory_embedding_model", None)
            if not model_id:
                return []
            try:
                client = get_embedding_client(model_id)
                res = await client.embed([query_text])
                if not res.embeddings:
                    return []
                vector = res.embeddings[0]
            except Exception:
                return []
        try:
            results = await store.query(VectorQuery(vector=vector or [], limit=limit))
            return results
        except Exception:
            return []

    @property
    def sandbox(self) -> SandboxProtocol | None:
        """Return the sandbox handle attached to this context when available."""
        return self._sandbox
