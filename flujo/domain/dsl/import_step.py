from __future__ import annotations

from typing import Any, Dict, Optional, Literal

from pydantic import Field

from .step import Step
from .pipeline import Pipeline


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
        Optional mapping of child context paths → parent context paths.
        When provided and ``updates_context=True``, only these mapped fields
        are merged back; otherwise the full child context is merged back.
    inherit_conversation:
        If True, conversation-related fields are preserved end-to-end. This is
        a hint for future enhancements; current implementation relies on context
        inheritance behavior.
    """

    pipeline: Pipeline[Any, Any]
    inherit_context: bool = True
    input_to: Literal["initial_prompt", "scratchpad", "both"] = "initial_prompt"
    input_scratchpad_key: Optional[str] = None
    outputs: Dict[str, str] = Field(default_factory=dict)
    inherit_conversation: bool = False

    @property
    def is_complex(self) -> bool:  # pragma: no cover - metadata only
        return True
