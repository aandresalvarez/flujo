from pydantic_ai_orchestrator.domain.agent_protocol import AgentProtocol
from pydantic_ai_orchestrator.infra.agents import (
    review_agent,
    solution_agent,
    validator_agent,
    get_reflection_agent,
    NoOpReflectionAgent,
)

def test_agents_conform_to_protocol():
    assert isinstance(review_agent, AgentProtocol)
    assert isinstance(solution_agent, AgentProtocol)
    assert isinstance(validator_agent, AgentProtocol)
    assert isinstance(get_reflection_agent(), AgentProtocol)
    assert isinstance(NoOpReflectionAgent(), AgentProtocol)
