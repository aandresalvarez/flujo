from __future__ import annotations

from typing import Literal
from collections.abc import MutableMapping

from ..base_model import BaseModel
from ..models import PipelineContext

from .step import Step
from .pipeline import Pipeline


class OutputMapping(BaseModel):
    """Declarative mapping from child → parent context paths.

    Example: { child: "import_artifacts.final_sql", parent: "import_artifacts.final_sql" }
    """

    child: str
    parent: str


def _get_nested_present(source: object, path: str) -> tuple[bool, object | None]:
    """Safely read a dotted path and track presence.

    This distinguishes between "missing path" and "present with value None"
    for mapping semantics (notably ImportStep output projection).
    """
    if not path:
        return False, None
    current = source
    for part in path.split("."):
        if isinstance(current, MutableMapping):
            if part not in current:
                return False, None
            current = current[part]
            continue
        # Pydantic models: if a field is None and not explicitly set, treat as missing.
        fields_def = getattr(type(current), "model_fields", None)
        if isinstance(fields_def, dict) and part in fields_def:
            val = getattr(current, part, None)
            fields_set: set[str] = getattr(current, "__pydantic_fields_set__", set())
            if val is None and part not in fields_set:
                return False, None
            current = val
            continue
        extra = getattr(current, "__pydantic_extra__", None)
        if isinstance(extra, dict) and part in extra:
            current = extra[part]
            continue
        try:
            current = getattr(current, part)
        except Exception:
            return False, None
    return True, current


def _get_nested(source: object, path: str) -> object | None:
    """Safely read a dotted path from nested dict/attr structures."""
    present, value = _get_nested_present(source, path)
    return value if present else None


def _set_nested(target: object, path: str, value: object) -> None:
    """Set a dotted path inside a mapping/object, creating intermediates as needed."""
    if not path:
        return
    current: object = target
    parts = path.split(".")
    for part in parts[:-1]:
        next_val: object | None = None
        if isinstance(current, MutableMapping):
            next_val = current.get(part)
            if not isinstance(next_val, MutableMapping):
                next_val = {}
                current[part] = next_val
        else:
            try:
                next_val = getattr(current, part)
            except Exception:
                next_val = None
            if not isinstance(next_val, MutableMapping):
                next_val = {}
                try:
                    setattr(current, part, next_val)
                except Exception:
                    if isinstance(current, MutableMapping):
                        current[part] = next_val
        current = next_val
    if isinstance(current, MutableMapping):
        current[parts[-1]] = value
    else:
        try:
            setattr(current, parts[-1], value)
        except Exception:
            if isinstance(current, MutableMapping):
                try:
                    current[parts[-1]] = value
                except Exception:
                    pass


class ImportStep(Step[object, object]):
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
        - "import_artifacts": merge dict input into import_artifacts (or store under key)
        - "both": apply both behaviors
    input_scratchpad_key:
        Optional key when projecting scalar inputs into import_artifacts.
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

    pipeline: Pipeline[object, object]
    inherit_context: bool = False
    input_to: Literal["initial_prompt", "import_artifacts", "both"] = "initial_prompt"
    input_scratchpad_key: str | None = "initial_input"
    outputs: list[OutputMapping] | None = None
    inherit_conversation: bool = True
    propagate_hitl: bool = True
    on_failure: Literal["abort", "skip", "continue_with_default"] = "abort"

    @property
    def is_complex(self) -> bool:  # pragma: no cover - metadata only
        return True

    def _project_output_to_parent(
        self,
        child_output: dict[str, object],
        child_context: "PipelineContext | None",
        parent_context: "PipelineContext | None",
        updates_context: bool,
    ) -> dict[str, object]:
        """Merge outputs from child context/output into parent context."""
        result: dict[str, object] = {}

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
            present_ctx = False
            child_value = None
            if child_context is not None:
                present_ctx, child_value = _get_nested_present(child_context, mapping.child)
            if not present_ctx:
                present_out, child_value = _get_nested_present(child_output, mapping.child)
                if not present_out:
                    child_value = None
            _set_nested(result, mapping.parent, child_value)

        return result
