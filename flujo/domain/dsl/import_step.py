from __future__ import annotations

from typing import Any, Optional, Literal, List

from ..base_model import BaseModel

from .step import Step
from .pipeline import Pipeline


class OutputMapping(BaseModel):
    """Declarative mapping from child → parent context paths.

    Example: { child: "scratchpad.final_sql", parent: "scratchpad.final_sql" }
    """

    child: str
    parent: str


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
