"""
Agent prompt templates and agent factory utilities.
"""
from pydantic_ai import Agent
from typing import Type
from pydantic_ai_orchestrator.infra.settings import settings
from pydantic_ai_orchestrator.domain.models import Checklist
from pydantic_ai_orchestrator.exceptions import OrchestratorRetryError
from tenacity import retry_if_exception_type, wait_random_exponential, RetryError
import asyncio
import logfire
import traceback

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

# 2. Agent Factory
def make_agent(
    model: str,
    system_prompt: str,
    output_type: Type,
) -> Agent:
    """Creates a pydantic_ai.Agent."""
    return Agent(
        model,
        system_prompt=system_prompt,
        output_type=output_type,
    )

class AsyncAgentWrapper:
    """Wraps an agent to expose .run_async with retry/timeout."""
    def __init__(self, agent, max_retries=3, timeout=120, model_name=None):
        self._agent = agent
        self._max_retries = max_retries
        self._timeout = timeout
        self._model_name = model_name or getattr(agent, 'model', None)

    async def _run_with_retry(self, *args, **kwargs):
        from tenacity import AsyncRetrying, stop_after_attempt
        temp = kwargs.pop("temperature", None)
        if temp is not None:
            kwargs.setdefault("generation_kwargs", {})["temperature"] = temp
        attempt = 0
        async for attempt_ctx in AsyncRetrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_random_exponential(multiplier=1, max=60),
            retry_error_callback=lambda retry_state: OrchestratorRetryError(
                f"Agent failed after {retry_state.attempt_number} attempts."
            ),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        ):
            with attempt_ctx:
                try:
                    result = await asyncio.wait_for(self._agent.run(*args, **kwargs), timeout=self._timeout)
                    logfire.info(f"Raw LLM response: {result}")
                    if isinstance(result, str) and result.startswith("Agent failed after"):
                        raise OrchestratorRetryError(result)
                    return result
                except Exception as e:
                    tb = traceback.format_exc()
                    logfire.error(f"Agent call failed with exception: {e}\nTraceback:\n{tb}")
                    raise

    async def run_async(self, *args, **kwargs):
        return await self._run_with_retry(*args, **kwargs)

    async def run(self, *args, **kwargs):
        return await self._run_with_retry(*args, **kwargs)

def make_agent_async(
    model: str,
    system_prompt: str,
    output_type: Type,
    max_retries: int = 3,
    timeout: int = 120,
) -> AsyncAgentWrapper:
    """
    Creates a pydantic_ai.Agent and returns an AsyncAgentWrapper exposing .run_async.
    """
    agent = make_agent(model, system_prompt, output_type)
    return AsyncAgentWrapper(agent, max_retries=max_retries, timeout=timeout, model_name=model)

class NoOpReflectionAgent:
    """A stub agent that does nothing, used when reflection is disabled."""
    async def run(self, *args, **kwargs):
        return ""

# 3. Agent Instances
review_agent = make_agent_async("openai:gpt-4o", REVIEW_SYS, Checklist)
solution_agent = make_agent_async("openai:gpt-4o", SOLUTION_SYS, str)
validator_agent = make_agent_async("openai:gpt-4o", VALIDATE_SYS, Checklist)

def get_reflection_agent() -> AsyncAgentWrapper | NoOpReflectionAgent:
    """Returns a new instance of the reflection agent, or a no-op if disabled."""
    if not settings.reflection_enabled:
        return NoOpReflectionAgent()
    try:
        agent = make_agent_async("openai:gpt-4o", REFLECT_SYS, str)
        logfire.info("Reflection agent created successfully.")
        return agent
    except Exception as e:
        logfire.error(f"Failed to create reflection agent: {e}")
        return NoOpReflectionAgent()

# Add a wrapper for review_agent to log API errors
class LoggingReviewAgent:
    def __init__(self, agent):
        self.agent = agent
        self.run_async = self._run_async
        self._run_with_retry = getattr(agent, "_run_with_retry", None)
    async def run(self, *args, **kwargs):
        try:
            result = await self.agent.run(*args, **kwargs)
            logfire.info(f"Review agent result: {result}")
            return result
        except Exception as e:
            logfire.error(f"Review agent API error: {e}")
            raise
    async def _run_async(self, *args, **kwargs):
        try:
            if hasattr(self.agent, "run_async") and callable(getattr(self.agent, "run_async")):
                result = await self.agent.run_async(*args, **kwargs)
            else:
                result = await self.run(*args, **kwargs)
            logfire.info(f"Review agent result (async): {result}")
            return result
        except Exception as e:
            logfire.error(f"Review agent API error (async): {e}")
            raise

review_agent = LoggingReviewAgent(review_agent) 