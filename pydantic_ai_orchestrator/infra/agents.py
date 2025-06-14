"""
Agent prompt templates and agent factory utilities.
"""

from pydantic_ai import Agent
from typing import Type, Any
import os
from pydantic_ai_orchestrator.infra.settings import settings
from pydantic_ai_orchestrator.domain.models import Checklist
from pydantic_ai_orchestrator.domain.agent_protocol import AgentProtocol
from pydantic_ai_orchestrator.exceptions import (
    OrchestratorRetryError,
    ConfigurationError,
)
import asyncio
from pydantic_ai_orchestrator.infra.telemetry import logfire
import traceback
from pydantic import BaseModel, Field
from tenacity import AsyncRetrying, RetryError, stop_after_attempt, wait_exponential

# 1. Prompt Constants
REVIEW_SYS = """You are an expert software engineer.\nYour task is to generate an objective, comprehensive, and actionable checklist of criteria to evaluate a solution for the user's request.\nThe checklist should be detailed and cover all key aspects of a good solution.\nFocus on correctness, completeness, and best practices.\n\nReturn **JSON only** that conforms to this schema:\nChecklist(items=[ChecklistItem(description:str, passed:bool|None, feedback:str|None)])\n\nExample:\n{\n  \"items\": [\n    {\"description\": \"The code is correct and runs without errors.\", \"passed\": null, \"feedback\": null},\n    {\"description\": \"The code follows best practices.\", \"passed\": null, \"feedback\": null}\n  ]\n}\n"""

SOLUTION_SYS = """You are a world-class programmer.
Your task is to provide a solution to the user's request.
Follow the user's instructions carefully and provide a high-quality, production-ready solution.
If you are given feedback on a previous attempt, use it to improve your solution.
"""

VALIDATE_SYS = """You are a meticulous QA engineer.\nReturn **JSON only** that conforms to this schema:\nChecklist(items=[ChecklistItem(description:str, passed:bool, feedback:str|None)])\nInput: {{ \"solution\": <string>, \"checklist\": <Checklist JSON> }}\nFor each item, fill `passed` & optional `feedback`.\n"""

REFLECT_SYS = """You are a senior principal engineer and an expert in root cause analysis.
You will be given a list of failed checklist items from a previous attempt.
Your task is to analyze these failures and provide a concise, high-level reflection on what went wrong.
Focus on the root cause of the failures and suggest a concrete, actionable strategy for the next attempt.
Do not repeat the failed items, but instead provide a new perspective on how to approach the problem.
Your output should be a single string.
"""

SELF_IMPROVE_SYS = """You are a debugging assistant specialized in AI pipelines.\n" \
    "You will receive step-by-step logs from failed evaluation cases and one" \
    " successful example. Analyze these to find root causes and suggest" \
    " concrete improvements. Consider pipeline prompts, step configuration" \
    " parameters such as temperature, and the evaluation suite itself" \
    " (proposing new tests or evaluator tweaks). Return JSON ONLY matching" \
    " ImprovementReport(suggestions=[ImprovementSuggestion(...)])."""


# 2. Agent Factory
def make_agent(
    model: str,
    system_prompt: str,
    output_type: Type[Any],
    tools: list[Any] | None = None,
) -> Agent[Any, Any]:
    """Creates a pydantic_ai.Agent, injecting the correct API key."""
    provider_name = model.split(":")[0].lower()
    api_key = None

    if provider_name == "openai":
        if not settings.openai_api_key:
            raise ConfigurationError(
                "To use OpenAI models, the OPENAI_API_KEY environment variable must be set."
            )
        api_key = settings.openai_api_key.get_secret_value()
        # Ensure the OpenAI client can locate the key
        os.environ.setdefault("OPENAI_API_KEY", api_key)
    elif provider_name in {"google-gla", "gemini"}:
        if not settings.google_api_key:
            raise ConfigurationError(
                "To use Gemini models, the GOOGLE_API_KEY environment variable must be set."
            )
        api_key = settings.google_api_key.get_secret_value()
    elif provider_name == "anthropic":
        if not settings.anthropic_api_key:
            raise ConfigurationError(
                "To use Anthropic models, the ANTHROPIC_API_KEY environment variable must be set."
            )
        api_key = settings.anthropic_api_key.get_secret_value()

    agent: Agent[Any, Any] = Agent(  # type: ignore[call-overload]
        model,
        system_prompt=system_prompt,
        output_type=output_type,
        api_key=api_key,
        tools=tools or [],
    )
    return agent


class AsyncAgentWrapper:
    """
    Wraps a pydantic_ai.Agent to provide an asynchronous interface
    with retry and timeout capabilities.
    """

    def __init__(
        self,
        agent: Agent[Any, Any],
        max_retries: int = 3,
        timeout: int | None = None,
        model_name: str | None = None,
    ) -> None:
        if not isinstance(max_retries, int):
            raise TypeError(f"max_retries must be an integer, got {type(max_retries).__name__}.")
        if max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer.")
        if timeout is not None:
            if not isinstance(timeout, int):
                raise TypeError(f"timeout must be an integer or None, got {type(timeout).__name__}.")
            if timeout <= 0:
                raise ValueError("timeout must be a positive integer if specified.")
        self._agent = agent
        self._max_retries = max_retries
        self._timeout_seconds: int | None = timeout if timeout is not None else settings.agent_timeout
        self._model_name: str | None = model_name or getattr(agent, "model", "unknown_model")

    async def _run_with_retry(self, *args: Any, **kwargs: Any) -> Any:
        temp = kwargs.pop("temperature", None)
        if temp is not None:
            if "generation_kwargs" not in kwargs or not isinstance(kwargs.get("generation_kwargs"), dict):
                kwargs["generation_kwargs"] = {}
            kwargs["generation_kwargs"]["temperature"] = temp

        retryer = AsyncRetrying(
            reraise=False,
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(max=60),
        )

        try:
            async for attempt in retryer:
                with attempt:
                    raw_agent_response = await asyncio.wait_for(
                        self._agent.run(*args, **kwargs),
                        timeout=self._timeout_seconds,
                    )
                    logfire.info(
                        f"Agent '{self._model_name}' raw response: {raw_agent_response}"
                    )

                    if isinstance(raw_agent_response, str) and raw_agent_response.startswith("Agent failed after"):
                        raise OrchestratorRetryError(raw_agent_response)

                    return raw_agent_response
        except RetryError as e:
            last_exc = e.last_attempt.exception()
            raise OrchestratorRetryError(
                f"Agent '{self._model_name}' failed after {self._max_retries} attempts. Last error: {type(last_exc).__name__}({last_exc})"
            ) from last_exc
        except Exception as e:
            tb = traceback.format_exc()
            logfire.error(
                f"Agent '{self._model_name}' call failed on attempt {attempt.retry_state.attempt_number} with exception: {type(e).__name__}({e})\nTraceback:\n{tb}"
            )
            raise

    async def run_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._run_with_retry(*args, **kwargs)

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        return await self.run_async(*args, **kwargs)


def make_agent_async(
    model: str,
    system_prompt: str,
    output_type: Type[Any],
    max_retries: int = 3,
    timeout: int | None = None,
) -> AsyncAgentWrapper:
    """
    Creates a pydantic_ai.Agent and returns an AsyncAgentWrapper exposing .run_async.
    """
    agent = make_agent(model, system_prompt, output_type)
    return AsyncAgentWrapper(agent, max_retries=max_retries, timeout=timeout, model_name=model)


class NoOpReflectionAgent:
    """A stub agent that does nothing, used when reflection is disabled."""

    async def run(self, *args: Any, **kwargs: Any) -> str:
        return ""


# 3. Agent Instances
try:
    review_agent = make_agent_async(settings.default_review_model, REVIEW_SYS, Checklist)
    solution_agent = make_agent_async(settings.default_solution_model, SOLUTION_SYS, str)
    validator_agent = make_agent_async(settings.default_validator_model, VALIDATE_SYS, Checklist)
except ConfigurationError:
    review_agent = solution_agent = validator_agent = NoOpReflectionAgent()  # type: ignore[assignment]


def get_reflection_agent(
    model: str | None = None,
) -> AsyncAgentWrapper | NoOpReflectionAgent:
    """Returns a new instance of the reflection agent, or a no-op if disabled."""
    if not settings.reflection_enabled:
        return NoOpReflectionAgent()
    try:
        model_name = model or settings.default_reflection_model
        agent = make_agent_async(model_name, REFLECT_SYS, str)
        logfire.info("Reflection agent created successfully.")
        return agent
    except Exception as e:
        logfire.error(f"Failed to create reflection agent: {e}")
        return NoOpReflectionAgent()


# Create a default instance for convenience and API consistency
reflection_agent = get_reflection_agent()


def make_self_improvement_agent(model: str | None = None) -> AsyncAgentWrapper:
    """Create the SelfImprovementAgent."""
    model_name = model or settings.default_solution_model
    return make_agent_async(model_name, SELF_IMPROVE_SYS, str)


# Default instance used by high level API
self_improvement_agent = make_self_improvement_agent()


# Add a wrapper for review_agent to log API errors
class LoggingReviewAgent:
    def __init__(self, agent: AgentProtocol[Any, Any]) -> None:
        self.agent = agent
        self.run_async = self._run_async
        self._run_with_retry = getattr(agent, "_run_with_retry", None)

    def __getattr__(self, name: str) -> Any:
        # Expose _run_with_retry and any other attributes from the wrapped agent
        if hasattr(self.agent, name):
            return getattr(self.agent, name)
        raise AttributeError(f"{self.__class__.__name__!r} object has no attribute {name!r}")

    async def _run_inner(self, method: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            result = await method(*args, **kwargs)
            logfire.info(f"Review agent result: {result}")
            return result
        except Exception as e:
            logfire.error(f"Review agent API error: {e}")
            raise

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        return await self._run_inner(self.agent.run, *args, **kwargs)

    async def _run_async(self, *args: Any, **kwargs: Any) -> Any:
        if hasattr(self.agent, "run_async") and callable(getattr(self.agent, "run_async")):
            return await self._run_inner(self.agent.run_async, *args, **kwargs)
        else:
            return await self.run(*args, **kwargs)


review_agent = LoggingReviewAgent(review_agent)  # type: ignore[assignment]
