"""
Agent factory utilities and wrapper classes.

This module provides factory functions for creating agents and wrapper classes
for async execution. It focuses on agent creation and resilience wrapping,
while system prompts are now in the flujo.prompts module.

This is the public API for the flujo agents package.
"""

from .monitoring import monitored_agent
from .factory import make_agent, _unwrap_type_adapter
from .wrapper import (
    AsyncAgentWrapper,
    make_agent_async,
    TemplatedAsyncAgentWrapper,
    make_templated_agent_async,
)
from .repair import DeterministicRepairProcessor, make_repair_agent, get_repair_agent
from .utils import get_raw_output_from_exception
from .recipes import (
    make_review_agent,
    make_solution_agent,
    make_validator_agent,
    get_reflection_agent,
    make_self_improvement_agent,
    LoggingReviewAgent,
    NoOpReflectionAgent,
    NoOpChecklistAgent,
    _is_image_generation_model,
    _attach_image_cost_post_processor,
    _deprecated_agent,
    __getattr__,
)

# Re-export important types from domain for convenience
from ..domain.agent_protocol import AsyncAgentProtocol, AgentInT, AgentOutT
from pydantic_ai import Agent

__all__ = [
    "monitored_agent",
    "make_agent",
    "_unwrap_type_adapter",
    "AsyncAgentWrapper",
    "make_agent_async",
    "TemplatedAsyncAgentWrapper",
    "make_templated_agent_async",
    # Repair functions
    "DeterministicRepairProcessor",
    "make_repair_agent",
    "get_repair_agent",
    "get_raw_output_from_exception",
    # Recipe functions
    "make_review_agent",
    "make_solution_agent",
    "make_validator_agent",
    "get_reflection_agent",
    "make_self_improvement_agent",
    "LoggingReviewAgent",
    "NoOpReflectionAgent",
    "NoOpChecklistAgent",
    "_is_image_generation_model",
    "_attach_image_cost_post_processor",
    "_deprecated_agent",
    "__getattr__",
    # Type exports
    "Agent",
    "AsyncAgentProtocol",
    "AgentInT",
    "AgentOutT",
]
