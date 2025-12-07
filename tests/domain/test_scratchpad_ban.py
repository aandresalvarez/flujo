from flujo.domain.models import PipelineContext


class UserContext(PipelineContext):
    user_id: str
    note: str | None = None


def test_pipeline_context_has_framework_reserved_scratchpad():
    ctx = PipelineContext()
    assert isinstance(ctx.scratchpad, dict)
    assert ctx.scratchpad == {}


def test_user_context_prefers_typed_fields_over_scratchpad():
    ctx = UserContext(user_id="u1", note="n1")
    assert ctx.user_id == "u1"
    assert ctx.note == "n1"
    # scratchpad reserved for framework metadata; should be empty by default
    assert ctx.scratchpad == {}


def test_scratchpad_remains_available_for_framework_metadata():
    ctx = PipelineContext()
    ctx.scratchpad["steps"] = {"s1": "out"}
    assert ctx.steps == {"s1": "out"}
