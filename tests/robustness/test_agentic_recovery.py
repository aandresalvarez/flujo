"""Tests for Agentic Recovery and Orchestration components.

This module tests the core agentic orchestration components:
- FinishCommand: Signal for loop termination with final answer
- AskHumanCommand: Signal for HITL pause
- RunAgentCommand: Signal to invoke sub-agents
- ReplayAgent: Deterministic replay for testing
- TemplatedAsyncAgentWrapper: Dynamic system prompt rendering

These components enable sophisticated agentic workflows including
multi-agent coordination, human-in-the-loop patterns, and deterministic testing.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from flujo.domain.commands import (
    FinishCommand,
    AskHumanCommand,
    RunAgentCommand,
    AgentCommand,
)
from flujo.testing.replay import ReplayAgent, ReplayError

pytestmark = pytest.mark.fast


# =============================================================================
# FinishCommand Tests
# =============================================================================


class TestFinishCommand:
    """Tests for FinishCommand signal handling.

    FinishCommand terminates a LoopStep with a final answer, even if
    max_loops has not been reached. This enables agents to signal
    completion based on task logic rather than iteration limits.
    """

    def test_create_with_dict_answer(self) -> None:
        """FinishCommand accepts dict as final answer."""
        command = FinishCommand(final_answer={"result": "success", "data": [1, 2, 3]})

        assert command.type == "finish"
        assert command.final_answer == {"result": "success", "data": [1, 2, 3]}

    def test_create_with_string_answer(self) -> None:
        """FinishCommand accepts string as final answer."""
        command = FinishCommand(final_answer="Task completed successfully")

        assert command.type == "finish"
        assert command.final_answer == "Task completed successfully"

    def test_create_with_complex_nested_answer(self) -> None:
        """FinishCommand accepts complex nested structures."""
        complex_answer = {
            "status": "complete",
            "results": [
                {"id": 1, "score": 0.95},
                {"id": 2, "score": 0.87},
            ],
            "metadata": {"elapsed_ms": 1234},
        }

        command = FinishCommand(final_answer=complex_answer)

        assert command.final_answer["results"][0]["score"] == 0.95

    def test_serialization_roundtrip(self) -> None:
        """FinishCommand survives JSON serialization roundtrip."""
        command = FinishCommand(final_answer={"key": "value"})

        serialized = command.model_dump(mode="json")
        restored = FinishCommand.model_validate(serialized)

        assert restored.type == "finish"
        assert restored.final_answer == {"key": "value"}

    def test_type_discriminator_is_finish(self) -> None:
        """Type discriminator must be 'finish' for command routing."""
        command = FinishCommand(final_answer="done")

        # The type field is used for discriminated union matching
        assert command.type == "finish"
        assert command.model_dump()["type"] == "finish"


# =============================================================================
# AskHumanCommand Tests
# =============================================================================


class TestAskHumanCommand:
    """Tests for AskHumanCommand (HITL pause signal).

    AskHumanCommand pauses execution and presents a question to the user.
    The response is captured and execution resumes with the human input.
    """

    def test_create_with_question(self) -> None:
        """AskHumanCommand requires a question string."""
        command = AskHumanCommand(question="What action should I take?")

        assert command.type == "ask_human"
        assert command.question == "What action should I take?"

    def test_question_can_be_multiline(self) -> None:
        """Question can contain newlines for complex prompts."""
        multiline_question = """Please review the following options:
        
1. Option A - Fast but risky
2. Option B - Slow but safe
3. Option C - Balanced approach

Which would you prefer?"""

        command = AskHumanCommand(question=multiline_question)

        assert "Option A" in command.question
        assert "Option C" in command.question

    def test_serialization_roundtrip(self) -> None:
        """AskHumanCommand survives JSON serialization roundtrip."""
        command = AskHumanCommand(question="Approve this action?")

        serialized = command.model_dump(mode="json")
        restored = AskHumanCommand.model_validate(serialized)

        assert restored.type == "ask_human"
        assert restored.question == "Approve this action?"


# =============================================================================
# RunAgentCommand Tests
# =============================================================================


class TestRunAgentCommand:
    """Tests for RunAgentCommand (sub-agent invocation).

    RunAgentCommand delegates work to a named sub-agent with input data.
    This enables multi-agent workflows where a coordinator agent can
    invoke specialized agents for specific tasks.
    """

    def test_create_with_agent_name_and_input(self) -> None:
        """RunAgentCommand requires agent_name and input_data."""
        command = RunAgentCommand(
            agent_name="analyzer", input_data={"text": "analyze this content"}
        )

        assert command.type == "run_agent"
        assert command.agent_name == "analyzer"
        assert command.input_data["text"] == "analyze this content"

    def test_input_data_can_be_complex(self) -> None:
        """Input data can be complex nested structures."""
        command = RunAgentCommand(
            agent_name="processor",
            input_data={"records": [{"id": 1}, {"id": 2}], "config": {"batch_size": 10}},
        )

        assert len(command.input_data["records"]) == 2

    def test_serialization_roundtrip(self) -> None:
        """RunAgentCommand survives JSON serialization roundtrip."""
        command = RunAgentCommand(agent_name="helper", input_data={"key": "value"})

        serialized = command.model_dump(mode="json")
        restored = RunAgentCommand.model_validate(serialized)

        assert restored.agent_name == "helper"
        assert restored.input_data == {"key": "value"}


# =============================================================================
# AgentCommand Union Tests
# =============================================================================


class TestAgentCommandUnion:
    """Tests for AgentCommand discriminated union."""

    def test_finish_command_is_agent_command(self) -> None:
        """FinishCommand is a valid AgentCommand variant."""
        command: AgentCommand = FinishCommand(final_answer="done")

        assert isinstance(command, FinishCommand)

    def test_ask_human_command_is_agent_command(self) -> None:
        """AskHumanCommand is a valid AgentCommand variant."""
        command: AgentCommand = AskHumanCommand(question="Help?")

        assert isinstance(command, AskHumanCommand)

    def test_run_agent_command_is_agent_command(self) -> None:
        """RunAgentCommand is a valid AgentCommand variant."""
        command: AgentCommand = RunAgentCommand(agent_name="helper", input_data={})

        assert isinstance(command, RunAgentCommand)


# =============================================================================
# ReplayAgent Tests
# =============================================================================


class TestReplayAgent:
    """Tests for ReplayAgent deterministic replay.

    ReplayAgent serves pre-recorded responses keyed by step name and attempt
    number. This enables deterministic testing of pipelines without actual
    LLM calls.
    """

    @pytest.mark.asyncio
    async def test_serves_recorded_response_by_step_name(self) -> None:
        """ReplayAgent returns correct response for step name."""
        responses = {
            "analyze:attempt_1": {"analysis": "positive", "score": 0.85},
            "summarize:attempt_1": {"summary": "Good results"},
        }

        agent = ReplayAgent(responses)

        result = await agent.run(step_name="analyze")
        assert result == {"analysis": "positive", "score": 0.85}

        result = await agent.run(step_name="summarize")
        assert result == {"summary": "Good results"}

    @pytest.mark.asyncio
    async def test_handles_different_attempt_numbers(self) -> None:
        """ReplayAgent distinguishes between retry attempts."""
        responses = {
            "flaky_step:attempt_1": {"error": "rate_limited"},
            "flaky_step:attempt_2": {"error": "timeout"},
            "flaky_step:attempt_3": {"result": "success"},
        }

        agent = ReplayAgent(responses)

        result1 = await agent.run(step_name="flaky_step", attempt_number=1)
        assert result1["error"] == "rate_limited"

        result2 = await agent.run(step_name="flaky_step", attempt_number=2)
        assert result2["error"] == "timeout"

        result3 = await agent.run(step_name="flaky_step", attempt_number=3)
        assert result3["result"] == "success"

    @pytest.mark.asyncio
    async def test_defaults_to_attempt_1(self) -> None:
        """Without explicit attempt_number, defaults to attempt_1."""
        responses = {"step:attempt_1": {"data": "first_try"}}

        agent = ReplayAgent(responses)
        result = await agent.run(step_name="step")  # No attempt_number

        assert result == {"data": "first_try"}

    @pytest.mark.asyncio
    async def test_raises_replay_error_for_missing_key(self) -> None:
        """ReplayError raised when step not in recorded responses."""
        responses = {"known_step:attempt_1": {"data": "value"}}

        agent = ReplayAgent(responses)

        with pytest.raises(ReplayError) as exc_info:
            await agent.run(step_name="unknown_step")

        assert "No recorded response" in str(exc_info.value)
        assert "unknown_step:attempt_1" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_replay_error_for_missing_attempt(self) -> None:
        """ReplayError raised when specific attempt not recorded."""
        responses = {"step:attempt_1": {"data": "value"}}

        agent = ReplayAgent(responses)

        with pytest.raises(ReplayError) as exc_info:
            await agent.run(step_name="step", attempt_number=5)

        assert "No recorded response" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_supports_string_responses(self) -> None:
        """ReplayAgent can serve string responses."""
        responses = {
            "chat:attempt_1": "Hello, how can I help?",
            "farewell:attempt_1": "Goodbye!",
        }

        agent = ReplayAgent(responses)

        assert await agent.run(step_name="chat") == "Hello, how can I help?"
        assert await agent.run(step_name="farewell") == "Goodbye!"

    @pytest.mark.asyncio
    async def test_supports_list_responses(self) -> None:
        """ReplayAgent can serve list responses."""
        responses = {
            "items:attempt_1": ["item1", "item2", "item3"],
        }

        agent = ReplayAgent(responses)

        result = await agent.run(step_name="items")
        assert result == ["item1", "item2", "item3"]

    @pytest.mark.asyncio
    async def test_uses_name_kwarg_as_fallback(self) -> None:
        """ReplayAgent accepts 'name' as alternative to 'step_name'."""
        responses = {"my_step:attempt_1": {"using": "name_kwarg"}}

        agent = ReplayAgent(responses)

        result = await agent.run(name="my_step")
        assert result == {"using": "name_kwarg"}


# =============================================================================
# TemplatedAsyncAgentWrapper Tests
# =============================================================================


class TestTemplatedAsyncAgentWrapper:
    """Tests for TemplatedAsyncAgentWrapper dynamic prompt rendering.

    This wrapper enables just-in-time system prompt rendering using
    Jinja-style templates with access to runtime context and step outputs.
    """

    @pytest.mark.asyncio
    async def test_wrapper_applies_template_to_system_prompt(self) -> None:
        """Wrapper should render template and apply to agent."""
        from flujo.agents.wrapper import TemplatedAsyncAgentWrapper
        from flujo.domain.agent_result import FlujoAgentResult

        # Create mock agent
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = "response"
        mock_result.usage = MagicMock(return_value=None)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_agent.system_prompt = ""

        wrapper = TemplatedAsyncAgentWrapper(
            mock_agent,
            template_string="You are helping {{ context.user_name }} with their task.",
            variables_spec={},
        )

        # Create mock context
        mock_context = SimpleNamespace(user_name="Alice", hitl_history=[])

        # Patch the adapter to control execution
        captured_prompt: dict[str, str | None] = {}

        async def _capture_prompt(*args: object, **kwargs: object) -> FlujoAgentResult:
            captured_prompt["value"] = getattr(wrapper._agent, "system_prompt", None)
            return FlujoAgentResult(
                output="mocked response",
                usage=None,
                cost_usd=None,
                token_counts=None,
            )

        with patch.object(wrapper._adapter, "run", side_effect=_capture_prompt) as mock_run:
            result = await wrapper.run_async("test", context=mock_context)

            # Verify result returned
            assert result.output == "mocked response"
            mock_run.assert_called_once()
            assert captured_prompt["value"] == "You are helping Alice with their task."

    @pytest.mark.asyncio
    async def test_wrapper_without_template_behaves_like_base(self) -> None:
        """Wrapper without template should behave like base AsyncAgentWrapper."""
        from flujo.agents.wrapper import TemplatedAsyncAgentWrapper
        from flujo.domain.agent_result import FlujoAgentResult

        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = "base response"
        mock_result.usage = MagicMock(return_value=None)
        mock_agent.run = AsyncMock(return_value=mock_result)

        wrapper = TemplatedAsyncAgentWrapper(
            mock_agent,
            template_string="",  # Empty template
            variables_spec={},
        )

        with patch.object(wrapper._adapter, "run") as mock_run:
            mock_run.return_value = FlujoAgentResult(
                output="response",
                usage=None,
                cost_usd=None,
                token_counts=None,
            )

            result = await wrapper.run_async("test")
            assert result.output == "response"
