from flujo.application.core.context_adapter import _inject_context_with_deep_merge
from flujo.domain.models import PipelineContext
from flujo.utils.template_vars import get_steps_map_from_context


def test_step_outputs_dual_write():
    """Verify that updating legacy 'steps' updates 'step_outputs'."""
    ctx = PipelineContext()

    # 1. Simulate a step update coming from the runner (via context adapter)
    update_data = {"steps": {"step_a": {"output": "foo"}}}

    # Inject
    error = _inject_context_with_deep_merge(ctx, update_data, PipelineContext)
    assert error is None

    # Check typed field (new) is populated
    assert ctx.step_outputs["step_a"]["output"] == "foo"

    # 2. Verify get_steps_map_from_context prefers typed field
    # We'll artificially desync them to prove it reads from typed field
    ctx.step_outputs["step_a"]["output"] = "bar"

    steps_map = get_steps_map_from_context(ctx)
    assert steps_map["step_a"]["output"] == "bar"


def test_checklist_and_solution_fields():
    """Verify new typed fields on PipelineContext."""
    ctx = PipelineContext()
    assert ctx.checklist is None
    assert ctx.solution is None

    ctx.solution = "my solution"
    assert ctx.solution == "my solution"
