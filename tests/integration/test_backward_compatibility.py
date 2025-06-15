from flujo.recipes import Default
from flujo.testing.utils import StubAgent
from flujo.domain.models import Task, Checklist


async def test_orchestrator_init_backward_compatible():
    review = StubAgent([Checklist(items=[])])
    solution = StubAgent(["sol"])
    validator = StubAgent([Checklist(items=[])])
    reflection = StubAgent([None])
    orch = Default(review, solution, validator, reflection)
    result = await orch.run_async(Task(prompt="hi"))
    assert result is None or hasattr(result, "score")
