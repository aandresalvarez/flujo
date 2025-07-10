from __future__ import annotations

from typing import Any, List, Dict, Optional, ClassVar
import uuid

from pydantic import Field, ConfigDict
import orjson

from .models import BaseModel
from .dsl.step import StepConfig, StepType

from .processors import AgentProcessors


# IR representations for plugins and validators
class ValidationPluginIR(BaseModel):
    plugin_type: str
    priority: int = 0


class ValidatorIR(BaseModel):
    validator_type: str


def _default_encoder(obj: Any) -> Any:
    """Best-effort serialization for arbitrary objects."""
    if obj is None:
        return None
    if callable(obj):
        return (
            f"{getattr(obj, '__module__', '<unknown>')}:{getattr(obj, '__qualname__', repr(obj))}"
        )
    try:
        orjson.dumps(obj)
        return obj
    except Exception:
        return repr(obj)


class StepIR(BaseModel):
    """Serializable representation of a single pipeline step."""

    step_uid: Optional[str] = Field(default=None, description="Unique step identifier")
    step_type: str = Field(default=StepType.STANDARD.value)
    name: str
    agent_ref: str | None = None
    agent: Any | None = Field(default=None, exclude=True)
    config: StepConfig = Field(default_factory=StepConfig)
    plugins: List[ValidationPluginIR] = Field(default_factory=list)
    validators: List[ValidatorIR] = Field(default_factory=list)
    processors: AgentProcessors = Field(default_factory=AgentProcessors)
    persist_feedback_to_context: str | None = None
    persist_validation_results_to: str | None = None
    updates_context: bool = False
    meta: Dict[str, Any] = Field(default_factory=dict)
    fallback_step_uid: str | None = None
    # Loop/Map specific
    loop_body: Optional["PipelineIR"] = None
    exit_condition_callable: Any | None = None
    max_loops: int | None = None
    initial_input_to_loop_body_mapper: Any | None = None
    iteration_input_mapper: Any | None = None
    loop_output_mapper: Any | None = None
    iterable_input: str | None = None
    # Conditional specific
    condition_callable: Any | None = None
    branches: Optional[Dict[Any, "PipelineIR"]] = None
    default_branch_pipeline: Optional["PipelineIR"] = None
    branch_input_mapper: Any | None = None
    branch_output_mapper: Any | None = None
    # Parallel specific
    parallel_branches: Optional[Dict[str, "PipelineIR"]] = None
    context_include_keys: List[str] | None = None
    merge_strategy: Any | None = None
    on_branch_failure: Any | None = None
    # Cache specific
    wrapped_step: Optional["StepIR"] = None
    cache_backend: Any | None = None
    # Human step
    message_for_user: str | None = None
    input_schema: Any | None = None

    model_config: ClassVar[ConfigDict] = {
        "arbitrary_types_allowed": True,
        "json_encoders": {object: lambda v: _default_encoder(v)},
    }

    def __init__(self, **data: Any) -> None:
        if "step_uid" not in data or data["step_uid"] is None:
            data["step_uid"] = uuid.uuid4().hex
        super().__init__(**data)


class PipelineIR(BaseModel):
    """Serializable representation of a pipeline."""

    steps: List[StepIR] = Field(default_factory=list)
    version: int = 1

    model_config: ClassVar[ConfigDict] = {"arbitrary_types_allowed": True}


StepIR.model_rebuild()
PipelineIR.model_rebuild()
