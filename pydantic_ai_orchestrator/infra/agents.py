"""
Agent prompt templates and agent factory utilities.
"""
from pydantic_ai import Agent
from tenacity import retry, wait_exponential, stop_after_attempt
from typing import Type
from .settings import settings
from ..domain.models import Checklist

# 1. Prompt Constants
REVIEW_SYS = """You are an expert software engineer.
Your task is to generate an objective, comprehensive, and actionable checklist of criteria to evaluate a solution for the user's request.
The checklist should be detailed and cover all key aspects of a good solution.
Focus on correctness, completeness, and best practices.
Output a Checklist object.
"""

SOLUTION_SYS = """You are a world-class programmer.
Your task is to provide a solution to the user's request.
Follow the user's instructions carefully and provide a high-quality, production-ready solution.
If you are given feedback on a previous attempt, use it to improve your solution.
"""

VALIDATE_SYS = """You are a meticulous quality assurance engineer.
Your task is to evaluate a given solution against a checklist of criteria.
For each item in the checklist, you must determine if the solution passes or fails.
If an item fails, you must provide clear and concise feedback on why it failed.
Your response must be a Checklist object with the `passed` and `feedback` fields filled for every item.
"""

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
    max_retries: int = 2,
    timeout: int = 90,
) -> Agent:
    """Creates a pydantic_ai.Agent with retry logic."""
    return Agent(
        model,
        system_prompt=system_prompt,
        output_type=output_type,
        max_retries=max_retries,
        timeout=timeout,
        retry=retry(
            wait=wait_exponential(multiplier=1, min=2, max=10),
            stop=stop_after_attempt(max_retries),
        ),
    )

class NoOpReflectionAgent:
    """A stub agent that does nothing, used when reflection is disabled."""
    def run_sync(self, *args, **kwargs):
        return "" # Must return a string to be appended to memory

# 3. Agent Instances
review_agent = make_agent("openai:gpt-4o", REVIEW_SYS, Checklist)
solution_agent = make_agent("openai:gpt-4o", SOLUTION_SYS, str)
validator_agent = make_agent("openai:gpt-4o", VALIDATE_SYS, Checklist)

def get_reflection_agent() -> Agent | NoOpReflectionAgent:
    """Returns a real reflection agent or a no-op stub based on settings."""
    if settings.reflection_enabled:
        return make_agent("openai:gpt-4o", REFLECT_SYS, str)
    else:
        return NoOpReflectionAgent()

reflection_agent = get_reflection_agent() 