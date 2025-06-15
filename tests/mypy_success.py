from flujo.domain import Step
from flujo.testing.utils import StubAgent
from pydantic import BaseModel


class UserInfo(BaseModel):
    name: str


class Report(BaseModel):
    summary: str


def test_pipeline_type_continuity() -> None:
    agent1 = StubAgent([UserInfo(name="test")])
    agent2 = StubAgent([Report(summary="report")])
    step1: Step[str, UserInfo] = Step.solution(agent1)
    step2: Step[UserInfo, Report] = Step.solution(agent2)
    _pipeline = step1 >> step2

    # pipeline should type check
    # The following should fail mypy if uncommented:
    # agent3 = StubAgent(["raw_string"])
    # step3: Step[int, str] = Step.solution(agent3)
    # bad_pipeline = step1 >> step3
