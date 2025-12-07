from __future__ import annotations

from typing import Any, Optional, Literal, List

from ..base_model import BaseModel
from ..models import PipelineContext

from .step import Step
from .pipeline import Pipeline


class OutputMapping(BaseModel):
    """Declarative mapping from child → parent context paths.

    Example: { child: "scratchpad.final_sql", parent: "scratchpad.final_sql" }
    """

    child: str
    parent: str


def _get_nested(source: Any, path: str) -> Any:
    """Safely read a dotted path from nested dict/attr structures."""
    if not path:
        return None
    current = source
    for part in path.split("."):
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
            continue
        try:
            current = getattr(current, part)
        except Exception:
            return None
    return current


def _set_nested(target: dict[str, Any], path: str, value: Any) -> None:
    """Set a dotted path inside a dictionary, creating intermediates as needed."""
    if not path:
        return
    current: dict[str, Any] = target
    parts = path.split(".")
    for part in parts[:-1]:
        next_val = current.get(part)
        if not isinstance(next_val, dict):
            next_val = {}
            current[part] = next_val
        current = next_val
    current[parts[-1]] = value


class ImportStep(Step[Any, Any]):
    """Compose an imported Pipeline as a first-class Step with policy-driven semantics.

    Fields
    ------
    pipeline:
        The child Pipeline to execute.
    inherit_context:
        Whether to inherit and deep-copy the parent context into the child run.
    input_to:
        Where to project the parent step input for the child run. One of:
        - "initial_prompt": JSON/text into child initial_prompt
        - "scratchpad": merge dict input into scratchpad (or store under key)
        - "both": apply both behaviors
    input_scratchpad_key:
        Optional key when projecting scalar inputs into the scratchpad.
    outputs:
        Optional list of mappings from child context paths → parent context paths.
        Semantics with ``updates_context=True``:
        - outputs is None → merge all child fields (legacy behavior)
        - outputs is []   → merge nothing
        - outputs has items → merge only the listed fields
    inherit_conversation:
        If True, conversation-related fields are preserved end-to-end. This is
        a hint for future enhancements; current implementation relies on context
        inheritance behavior.
    propagate_hitl:
        When True, a HITL pause raised within the child pipeline will be
        propagated to the parent as a Paused outcome, allowing the runner to
        surface the question and resume correctly. When False, the import step
        will not proxy pauses (legacy behavior).
    on_failure:
        Control behavior when the child import fails. One of:
        - "abort": propagate failure to parent (default)
        - "skip": treat as success and merge nothing
        - "continue_with_default": treat as success with empty/default output
    """

    pipeline: Pipeline[Any, Any]
    inherit_context: bool = False
    input_to: Literal["initial_prompt", "scratchpad", "both"] = "initial_prompt"
    input_scratchpad_key: Optional[str] = "initial_input"
    outputs: Optional[List[OutputMapping]] = None
    inherit_conversation: bool = True
    propagate_hitl: bool = True
    on_failure: Literal["abort", "skip", "continue_with_default"] = "abort"

    @property
    def is_complex(self) -> bool:  # pragma: no cover - metadata only
        return True

    def _project_output_to_parent(
        self,
        child_output: dict[str, Any],
        child_context: Optional["PipelineContext"],
        parent_context: Optional["PipelineContext"],
        updates_context: bool,
    ) -> dict[str, Any]:
        """Merge outputs from child context/output into parent context."""
        result: dict[str, Any] = {}

        if not updates_context:
            # If updates_context=False, only map outputs (default: none)
            if self.outputs:
                for mapping in self.outputs:
                    child_value = _get_nested(child_output, mapping.child)
                    _set_nested(result, mapping.parent, child_value)
            return result

        # Legacy behavior: outputs=None -> merge entire child context
        if self.outputs is None:
            if child_context is not None:
                result = child_context.model_dump()
            return result

        # outputs=[] -> merge nothing
        if not self.outputs:
            return result

        # outputs list provided -> merge specified fields
        for mapping in self.outputs:
            child_value = _get_nested(child_output, mapping.child)
            if child_value is None and child_context is not None:
                child_value = _get_nested(child_context.model_dump(), mapping.child)
            # Prefer writing into import_artifacts when available to avoid scratchpad
            if parent_context is not None and hasattr(parent_context, "import_artifacts"):
                target = parent_context.import_artifacts
                _set_nested(target, mapping.parent, child_value)
            else:
                _set_nested(result, mapping.parent, child_value)

        return result
