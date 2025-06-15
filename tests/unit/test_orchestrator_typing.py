from flujo.domain.agent_protocol import AgentProtocol
from flujo.infra.agents import (
    review_agent,
    solution_agent,
    validator_agent,
    get_reflection_agent,
    NoOpReflectionAgent,
)


def test_agents_conform_to_protocol() -> None:
    assert isinstance(review_agent, AgentProtocol)
    assert isinstance(solution_agent, AgentProtocol)
    assert isinstance(validator_agent, AgentProtocol)
    assert isinstance(get_reflection_agent(), AgentProtocol)
    assert isinstance(NoOpReflectionAgent(), AgentProtocol)
