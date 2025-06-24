"""
Agent prompt templates and agent factory utilities.
"""

from __future__ import annotations

from typing import (
    Type,
    Any,
    Generic,
)
from pydantic_ai import Agent
from pydantic import BaseModel as PydanticBaseModel
import os
from flujo.infra.settings import settings
from flujo.domain.models import Checklist
from flujo.domain.agent_protocol import (
    AsyncAgentProtocol,
    AgentInT,
    AgentOutT,
)
from flujo.exceptions import (
    OrchestratorRetryError,
    ConfigurationError,
)
import asyncio
from flujo.infra.telemetry import logfire
import traceback
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
    " parameters such as temperature, retries, and timeout. Each step may" \
    " include a SystemPromptSummary line showing a redacted snippet of its" \
    " system prompt. Also consider the evaluation suite itself" \
    " (proposing new tests or evaluator tweaks). Return JSON ONLY matching" \
    " ImprovementReport(suggestions=[ImprovementSuggestion(...)])."\n\n" \
    "Here are some examples of desired input/output:\n\n" \
    "EXAMPLE 1:\n" \
    "Input Context:\n" \
    "Case: test_short_summary_too_long\n" \
    "- PlanGeneration: Output(content=\"Create a 5-sentence summary.\") (success=True)\n" \
    "- SummarizationStep: Output(content=\"This is a very long summary that unfortunately exceeds the five sentence limit by quite a bit, going into extensive detail about many different aspects of the topic, providing background, and also some future outlook which was not requested.\") (success=True)\n" \
    "- ValidationStep: Output(passed=False, feedback=\"Summary exceeds 5 sentences.\") (success=True)\n" \
    "Successful example:\n" \
    "Case: test_short_summary_correct_length\n" \
    "- PlanGeneration: Output(content=\"Create a 3-sentence summary.\") (success=True)\n" \
    "- SummarizationStep: Output(content=\"Topic is complex. It has three main points. This is the third sentence.\") (success=True)\n" \
    "- ValidationStep: Output(passed=True, feedback=\"Summary within length.\") (success=True)\n\n" \
    "Expected JSON Output:\n" \
    "{\n" \
    "  \"suggestions\": [\n" \
    "    {\n" \
    "      \"target_step_name\": \"SummarizationStep\",\n" \
    "      \"suggestion_type\": \"PROMPT_MODIFICATION\",\n" \
    "      \"failure_pattern_summary\": \"Generated summary consistently exceeds specified sentence limits.\",\n" \
    "      \"detailed_explanation\": \"The agent in 'SummarizationStep' seems to be overly verbose. Its system prompt should be strengthened to strictly adhere to length constraints. Consider adding phrases like 'Be concise and strictly follow the sentence limit provided in the plan.' or 'Do not add extra information beyond the core summary points.'\",\n" \
    "      \"prompt_modification_details\": {\n" \
    "        \"modification_instruction\": \"Update system prompt for 'SummarizationStep' to emphasize strict adherence to sentence limits, e.g., add 'Be concise and strictly follow the sentence limit. Do not add extra information.'\"\n" \
    "      },\n" \
    "      \"example_failing_input_snippets\": [\"Input for test_short_summary_too_long...\"],\n" \
    "      \"estimated_impact\": \"HIGH\",\n" \
    "      \"estimated_effort_to_implement\": \"LOW\"\n" \
    "    },\n" \
    "    {\n" \
    "      \"suggestion_type\": \"EVAL_CASE_REFINEMENT\",\n" \
    "      \"failure_pattern_summary\": \"Evaluation relies on a simple length check by ValidationStep.\",\n" \
    "      \"detailed_explanation\": \"The 'ValidationStep' correctly identifies length issues. However, to make the evaluation more robust, consider if the 'SummarizationStep' itself could be made to output a Pydantic model like `SummaryOutput(text: str, sentence_count: int)` which would make length validation trivial and less prone to LLM misinterpretation of 'sentence'.\",\n" \
    "      \"example_failing_input_snippets\": [\"Input for test_short_summary_too_long...\"],\n" \
    "      \"estimated_impact\": \"MEDIUM\",\n" \
    "      \"estimated_effort_to_implement\": \"MEDIUM\"\n" \
    "    }\n" \
    "  ]\n" \
    "}\n\n" \
    "EXAMPLE 2:\n" \
    "Input Context:\n" \
    "Case: test_sql_syntax_error\n" \
    "- GenerateSQL: Output(content=\"SELEC * FRM Users WHERE id = 1\") (success=True)\n" \
    "- ValidateSQL: Output(passed=False, feedback=\"Syntax error near 'SELEC'\") (success=True)\n" \
    "Successful example:\n" \
    "Case: test_sql_correct_syntax\n" \
    "- GenerateSQL: Output(content=\"SELECT * FROM Users WHERE id = 1\") (success=True)\n" \
    "- ValidateSQL: Output(passed=True, feedback=\"Valid SQL\") (success=True)\n\n" \
    "Expected JSON Output:\n" \
    "{\n" \
    "  \"suggestions\": [\n" \
    "    {\n" \
    "      \"target_step_name\": \"GenerateSQL\",\n" \
    "      \"suggestion_type\": \"PROMPT_MODIFICATION\",\n" \
    "      \"failure_pattern_summary\": \"Agent frequently makes basic SQL syntax errors (e.g., typos like 'SELEC').\",\n" \
    "      \"detailed_explanation\": \"The agent in 'GenerateSQL' needs stronger guidance on SQL syntax. Its system prompt could include a reminder to double-check keywords or even a small example of correct syntax. Alternatively, if this is a common agent, consider fine-tuning it on valid SQL examples.\",\n" \
    "      \"prompt_modification_details\": {\n" \
    "        \"modification_instruction\": \"Add to 'GenerateSQL' system prompt: 'Ensure all SQL keywords like SELECT, FROM, WHERE are spelled correctly.'\"\n" \
    "      },\n" \
    "      \"example_failing_input_snippets\": [\"Input for test_sql_syntax_error...\"],\n" \
    "      \"estimated_impact\": \"HIGH\",\n" \
    "      \"estimated_effort_to_implement\": \"LOW\"\n" \
    "    },\n" \
    "    {\n" \
    "      \"suggestion_type\": \"NEW_EVAL_CASE\",\n" \
    "      \"failure_pattern_summary\": \"Current tests only cover basic SELECT typos.\",\n" \
    "      \"detailed_explanation\": \"To improve robustness, add new evaluation cases that test for other common SQL syntax errors, such as incorrect JOIN syntax, missing commas, or issues with aggregate functions.\",\n" \
    "      \"suggested_new_eval_case_description\": \"Create an eval case with an input prompt that requires a JOIN statement, and expect the agent to generate it correctly. Another case could test for correct use of GROUP BY.\",\n" \
    "      \"estimated_impact\": \"MEDIUM\",\n" \
    "      \"estimated_effort_to_implement\": \"MEDIUM\"\n" \
    "    }\n" \
    "  ]\n" \
    "}\n" \
    """


# 2. Agent Factory
def make_agent(
    model: str,
    system_prompt: str,
    output_type: Type[Any],
    tools: list[Any] | None = None,
    **kwargs: Any,
) -> Agent[Any, Any]:
    """Creates a pydantic_ai.Agent, injecting the correct API key."""
    provider_name = model.split(":")[0].lower()

    if provider_name == "openai":
        if not settings.openai_api_key:
            raise ConfigurationError(
                "To use OpenAI models, the OPENAI_API_KEY environment variable must be set."
            )
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key.get_secret_value())
    elif provider_name in {"google-gla", "gemini"}:
        if not settings.google_api_key:
            raise ConfigurationError(
                "To use Gemini models, the GOOGLE_API_KEY environment variable must be set."
            )
        os.environ.setdefault("GOOGLE_API_KEY", settings.google_api_key.get_secret_value())
    elif provider_name == "anthropic":
        if not settings.anthropic_api_key:
            raise ConfigurationError(
                "To use Anthropic models, the ANTHROPIC_API_KEY environment variable must be set."
            )
        os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key.get_secret_value())

    # Handle TypeAdapter and complex type patterns
    actual_type = output_type
    try:
        if hasattr(output_type, "_type"):
            # Handle TypeAdapter instances - extract the underlying type
            actual_type = output_type._type
        elif hasattr(output_type, "__origin__") and output_type.__origin__ is not None:
            # Handle generic types like TypeAdapter[str]
            if hasattr(output_type, "__args__") and output_type.__args__:
                if output_type.__origin__.__name__ == "TypeAdapter":
                    actual_type = output_type.__args__[0]

        # Validate that the actual_type is a valid Pydantic type
        # We avoid testing schema generation directly to prevent the infinite recursion issue
        if hasattr(actual_type, "__name__"):
            # Built-in types like str, int, etc. are always valid
            pass
        elif hasattr(actual_type, "__bases__") and PydanticBaseModel in actual_type.__bases__:
            # Pydantic models are valid
            pass
        else:
            # For other types, try a simple validation
            try:
                from pydantic import create_model

                create_model("TestModel", value=(actual_type, ...))
            except Exception as schema_error:
                raise ValueError(
                    f"Invalid output_type '{output_type}' (resolved to '{actual_type}'): {schema_error}. "
                    "Use a Pydantic model, built-in type, or properly configured TypeAdapter."
                ) from schema_error

    except Exception as e:
        raise ValueError(f"Error processing output_type '{output_type}': {e}") from e

    # The Agent constructor's type hints are not strict enough for mypy strict mode.
    # See: https://github.com/pydantic/pydantic-ai/issues (file an issue if not present)
    agent: Agent[Any, Any] = Agent(
        model=model,
        system_prompt=system_prompt,
        output_type=actual_type,
        tools=tools or [],
    )
    return agent


class AsyncAgentWrapper(Generic[AgentInT, AgentOutT], AsyncAgentProtocol[AgentInT, AgentOutT]):
    """
    Wraps a pydantic_ai.Agent to provide an asynchronous interface
    with retry and timeout capabilities.
    """

    def __init__(
        self,
        agent: Agent[Any, AgentOutT],
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
                raise TypeError(
                    f"timeout must be an integer or None, got {type(timeout).__name__}."
                )
            if timeout <= 0:
                raise ValueError("timeout must be a positive integer if specified.")
        self._agent = agent
        self._max_retries = max_retries
        self._timeout_seconds: int | None = (
            timeout if timeout is not None else settings.agent_timeout
        )
        self._model_name: str | None = model_name or getattr(agent, "model", "unknown_model")

    def _call_agent_with_dynamic_args(self, *args: Any, **kwargs: Any) -> Any:
        return self._agent.run(*args, **kwargs)

    async def _run_with_retry(self, *args: Any, **kwargs: Any) -> Any:
        temp = kwargs.pop("temperature", None)
        if temp is not None:
            if "generation_kwargs" not in kwargs or not isinstance(
                kwargs.get("generation_kwargs"), dict
            ):
                kwargs["generation_kwargs"] = {}
            kwargs["generation_kwargs"]["temperature"] = temp

        # Compatibility shim: pydantic-ai expects serializable dicts for its
        # internal function-calling message generation, not Pydantic model
        # instances. We automatically serialize any BaseModel inputs here to
        # ensure compatibility.
        processed_args = [
            arg.model_dump() if isinstance(arg, PydanticBaseModel) else arg for arg in args
        ]
        processed_kwargs = {
            key: value.model_dump() if isinstance(value, PydanticBaseModel) else value
            for key, value in kwargs.items()
        }

        retryer = AsyncRetrying(
            reraise=False,
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(max=60),
        )

        try:
            async for attempt in retryer:
                with attempt:
                    raw_agent_response = await asyncio.wait_for(
                        self._call_agent_with_dynamic_args(
                            *processed_args,
                            **processed_kwargs,
                        ),
                        timeout=self._timeout_seconds,
                    )
                    logfire.info(f"Agent '{self._model_name}' raw response: {raw_agent_response}")

                    if isinstance(raw_agent_response, str) and raw_agent_response.startswith(
                        "Agent failed after"
                    ):
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
) -> AsyncAgentWrapper[Any, Any]:
    """
    Creates a pydantic_ai.Agent and returns an AsyncAgentWrapper exposing .run_async.
    """
    agent = make_agent(model, system_prompt, output_type)
    return AsyncAgentWrapper(agent, max_retries=max_retries, timeout=timeout, model_name=model)


class NoOpReflectionAgent(AsyncAgentProtocol[Any, str]):
    """A stub agent that does nothing, used when reflection is disabled."""

    async def run(self, data: Any | None = None, **kwargs: Any) -> str:
        return ""

    async def run_async(self, data: Any | None = None, **kwargs: Any) -> str:
        return ""


class NoOpChecklistAgent(AsyncAgentProtocol[Any, Checklist]):
    """A stub agent that returns an empty Checklist, used as a fallback for checklist agents."""

    async def run(self, data: Any | None = None, **kwargs: Any) -> Checklist:
        return Checklist(items=[])

    async def run_async(self, data: Any | None = None, **kwargs: Any) -> Checklist:
        return Checklist(items=[])


# 3. Agent Instances
try:
    review_agent: AsyncAgentProtocol[Any, Checklist] = make_agent_async(
        settings.default_review_model, REVIEW_SYS, Checklist
    )
    solution_agent: AsyncAgentProtocol[Any, str] = make_agent_async(
        settings.default_solution_model, SOLUTION_SYS, str
    )
    validator_agent: AsyncAgentProtocol[Any, Checklist] = make_agent_async(
        settings.default_validator_model, VALIDATE_SYS, Checklist
    )
except ConfigurationError:
    # If configuration fails, use a no-op stub agent for all three
    review_agent = NoOpChecklistAgent()
    solution_agent = NoOpReflectionAgent()
    validator_agent = NoOpChecklistAgent()


def get_reflection_agent(
    model: str | None = None,
) -> AsyncAgentProtocol[Any, Any] | NoOpReflectionAgent:
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
reflection_agent: AsyncAgentProtocol[Any, Any] | NoOpReflectionAgent = get_reflection_agent()


def make_self_improvement_agent(
    model: str | None = None,
) -> AsyncAgentWrapper[Any, str]:
    """Create the SelfImprovementAgent."""
    model_name = model or settings.default_self_improvement_model
    return make_agent_async(model_name, SELF_IMPROVE_SYS, str)


# Default instance used by high level API
try:
    self_improvement_agent: AsyncAgentProtocol[Any, str] = make_self_improvement_agent()
except ConfigurationError:  # pragma: no cover - config may be missing in tests
    self_improvement_agent = NoOpReflectionAgent()


class LoggingReviewAgent(AsyncAgentProtocol[Any, Any]):
    """Wrapper for review agent that adds logging."""

    def __init__(self, agent: AsyncAgentProtocol[Any, Any]) -> None:
        self.agent = agent

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        return await self._run_inner(self.agent.run, *args, **kwargs)

    async def _run_async(self, *args: Any, **kwargs: Any) -> Any:
        if hasattr(self.agent, "run_async") and callable(getattr(self.agent, "run_async")):
            return await self._run_inner(self.agent.run_async, *args, **kwargs)
        else:
            return await self.run(*args, **kwargs)

    async def run_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._run_async(*args, **kwargs)

    async def _run_inner(self, method: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            result = await method(*args, **kwargs)
            logfire.info(f"Review agent result: {result}")
            return result
        except Exception as e:
            logfire.error(f"Review agent API error: {e}")
            raise


# Update the review_agent assignment to use proper typing
review_agent = LoggingReviewAgent(review_agent)

# Explicit exports
__all__ = [
    "REVIEW_SYS",
    "SOLUTION_SYS",
    "VALIDATE_SYS",
    "REFLECT_SYS",
    "SELF_IMPROVE_SYS",
    "make_agent",
    "make_agent_async",
    "AsyncAgentWrapper",
    "NoOpReflectionAgent",
    "get_reflection_agent",
    "make_self_improvement_agent",
    "review_agent",
    "solution_agent",
    "validator_agent",
    "reflection_agent",
    "self_improvement_agent",
    "Agent",
    "AsyncAgentProtocol",
    "AgentInT",
    "AgentOutT",
]
