from __future__ import annotations

from typing import Any

from flujo.architect.states.common import goto
from flujo.domain.base_model import BaseModel as _BaseModel
from flujo.domain.dsl import Pipeline, Step
from flujo.type_definitions.common import JSONObject


async def _capture_refinement(*_a: Any, context: _BaseModel | None = None) -> JSONObject:
    fb = None
    try:
        fb = getattr(context, "refinement_feedback", None)
    except Exception:
        fb = None
    if not isinstance(fb, str) or not fb.strip():
        fb = "Please improve the plan based on user feedback."
    return {"refinement_feedback": fb}


def build_refinement_state() -> Pipeline[Any, Any]:
    """Capture refinement feedback and re-enter Planning."""
    return Pipeline.from_step(
        Step.from_callable(_capture_refinement, name="CaptureRefinement", updates_context=True)
    ) >> Step.from_callable(
        lambda *_a, **_k: goto("Planning"), name="GotoReplan", updates_context=True
    )
