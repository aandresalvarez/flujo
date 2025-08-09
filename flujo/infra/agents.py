"""
Agent factory utilities and wrapper classes.

This module provides factory functions for creating agents and wrapper classes
for async execution. It focuses on agent creation and resilience wrapping,
while system prompts are now in the flujo.prompts module.
"""

from __future__ import annotations

import asyncio
import json
import warnings
from typing import Any, Optional, Type, Generic, get_origin

from pydantic import ValidationError
from pydantic_ai import Agent, ModelRetry
from pydantic import BaseModel as PydanticBaseModel, TypeAdapter
import os
from tenacity import AsyncRetrying, RetryError, stop_after_attempt, wait_exponential

from ..domain.agent_protocol import AsyncAgentProtocol, AgentInT, AgentOutT
from ..domain.models import Checklist, ImprovementReport
from ..domain.processors import AgentProcessors
from ..exceptions import OrchestratorError, OrchestratorRetryError, ConfigurationError
from ..utils.serialization import safe_serialize

from .settings import settings
from .telemetry import logfire

# Import prompts from the new prompts module
from ..prompts import (
    REVIEW_SYS,
    SOLUTION_SYS,
    VALIDATE_SYS,
    REFLECT_SYS,
    SELF_IMPROVE_SYS,
    REPAIR_SYS,
    _format_repair_prompt,
)





# Import the agent factory and wrapper from the new dedicated modules
from ..agents.factory import make_agent, _unwrap_type_adapter
from ..agents.wrapper import AsyncAgentWrapper, make_agent_async

# Alias for backward compatibility with tests that monkeypatch this module
make_agent_async = make_agent_async


# The AsyncAgentWrapper class has been moved to flujo.agents.wrapper
# This import maintains backward compatibility






# Import all the recipe functions from the new recipes module
from ..agents.recipes import (
    _is_image_generation_model,
    _attach_image_cost_post_processor,
    NoOpReflectionAgent,
    NoOpChecklistAgent,
    get_reflection_agent,
    make_self_improvement_agent,
    make_review_agent,
    make_solution_agent,
    make_validator_agent,
    LoggingReviewAgent,
)

# Import repair functions from the new repair module
from ..agents.repair import make_repair_agent, get_repair_agent
# Import from utils (where it's actually implemented)
from ..agents.utils import get_raw_output_from_exception


# Explicit exports
__all__ = [
    "make_agent",
    "make_agent_async",
    "AsyncAgentWrapper",
    "NoOpReflectionAgent",
    "NoOpChecklistAgent",
    "get_reflection_agent",
    "make_self_improvement_agent",
    "make_repair_agent",
    "get_repair_agent",
    "get_raw_output_from_exception",
    "make_review_agent",
    "make_solution_agent",
    "make_validator_agent",
    "LoggingReviewAgent",
    "_is_image_generation_model",
    "_attach_image_cost_post_processor",
    "Agent",
    "AsyncAgentProtocol",
    "AgentInT",
    "AgentOutT",
]


# Import deprecation handling from the recipes module
from ..agents.recipes import _deprecated_agent, __getattr__
