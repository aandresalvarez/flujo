from flujo.domain.agent_protocol import AgentProtocol
from flujo.infra.agents import (
    make_review_agent,
    make_solution_agent,
    make_validator_agent,
    get_reflection_agent,
    NoOpReflectionAgent,
)


def test_agents_conform_to_protocol() -> None:
    assert isinstance(make_review_agent(), AgentProtocol)
    assert isinstance(make_solution_agent(), AgentProtocol)
    assert isinstance(make_validator_agent(), AgentProtocol)
    assert isinstance(get_reflection_agent(), AgentProtocol)
    assert isinstance(NoOpReflectionAgent(), AgentProtocol)
