from __future__ import annotations

from typing import Any, Optional, Literal, List

from pydantic import Field
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
        When provided and ``updates_context=True``, only these mapped fields are
        merged back.
    inherit_conversation:
        If True, conversation-related fields are preserved end-to-end. This is
        a hint for future enhancements; current implementation relies on context
        inheritance behavior.
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
    outputs: List[OutputMapping] = Field(default_factory=list)
    inherit_conversation: bool = True
    on_failure: Literal["abort", "skip", "continue_with_default"] = "abort"

    @property
    def is_complex(self) -> bool:  # pragma: no cover - metadata only
        return True
