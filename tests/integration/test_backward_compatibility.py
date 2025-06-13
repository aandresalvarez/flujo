from pydantic_ai_orchestrator.application.orchestrator import Orchestrator
from pydantic_ai_orchestrator.testing.utils import StubAgent
from pydantic_ai_orchestrator.domain.models import Task, Checklist


async def test_orchestrator_init_backward_compatible():
    review = StubAgent([Checklist(items=[])])
    solution = StubAgent(["sol"])
    validator = StubAgent([Checklist(items=[])])
    reflection = StubAgent([None])
    orch = Orchestrator(review, solution, validator, reflection)
    result = await orch.run_async(Task(prompt="hi"))
    assert result is None or hasattr(result, "score")
