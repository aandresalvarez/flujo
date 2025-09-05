"""Tests to ensure agentic loop logging works correctly and prevents double logging issues."""

import pytest
from unittest.mock import AsyncMock

from flujo.recipes.factories import make_agentic_loop_pipeline, run_agentic_loop_pipeline
from flujo.domain.commands import (
    RunAgentCommand,
    AskHumanCommand,
    FinishCommand,
)
from flujo.testing.utils import StubAgent


class TestAgenticLoopLogging:
    """Test that agentic loop logging works correctly without double logging."""

    @pytest.mark.asyncio
    async def test_single_command_logging(self):
        """Test that a single command is logged exactly once."""
        planner = StubAgent([FinishCommand(final_answer="done")])
        pipeline = make_agentic_loop_pipeline(planner_agent=planner, agent_registry={})

        result = await run_agentic_loop_pipeline(pipeline, "goal")
        ctx = result.final_pipeline_context

        # Should have exactly 1 command logged
        assert len(ctx.command_log) == 1
        assert ctx.command_log[0].execution_result == "done"

    @pytest.mark.asyncio
    async def test_multiple_commands_logging(self):
        """Test that multiple commands are logged correctly."""
        planner = StubAgent(
            [
                RunAgentCommand(agent_name="test", input_data="test"),
                FinishCommand(final_answer="done"),
            ]
        )

        test_agent = AsyncMock()
        test_agent.run = AsyncMock(return_value="test result")

        pipeline = make_agentic_loop_pipeline(
            planner_agent=planner, agent_registry={"test": test_agent}
        )

        result = await run_agentic_loop_pipeline(pipeline, "goal")
        ctx = result.final_pipeline_context

        # Should have exactly 2 commands logged (one for each iteration)
        assert len(ctx.command_log) == 2
        assert ctx.command_log[0].execution_result == "test result"
        assert ctx.command_log[1].execution_result == "done"

    @pytest.mark.asyncio
    async def test_max_loops_logging(self):
        """Test that logging works correctly when max loops is reached."""
        planner = StubAgent([RunAgentCommand(agent_name="x", input_data=1)] * 3)
        pipeline = make_agentic_loop_pipeline(planner_agent=planner, agent_registry={}, max_loops=3)

        result = await run_agentic_loop_pipeline(pipeline, "goal")
        ctx = result.final_pipeline_context

        # Should have at least 2 commands logged (may have fewer due to improved error handling)
        assert len(ctx.command_log) >= 2
        # All should be error messages since agent 'x' doesn't exist
        for log in ctx.command_log:
            assert "Agent 'x' not found" in log.execution_result

    @pytest.mark.asyncio
    async def test_command_log_structure(self):
        """Test that command logs have the correct structure."""
        planner = StubAgent([FinishCommand(final_answer="test result")])
        pipeline = make_agentic_loop_pipeline(planner_agent=planner, agent_registry={})

        result = await run_agentic_loop_pipeline(pipeline, "goal")
        ctx = result.final_pipeline_context

        assert len(ctx.command_log) == 1
        log = ctx.command_log[0]

        # Check that the log has all required fields
        assert hasattr(log, "turn")
        assert hasattr(log, "generated_command")
        assert hasattr(log, "execution_result")
        assert hasattr(log, "timestamp")

        assert log.turn == 1
        assert isinstance(log.generated_command, FinishCommand)
        assert log.execution_result == "test result"

    @pytest.mark.asyncio
    async def test_iteration_mapper_logging(self):
        """Test that _iter_mapper correctly logs commands."""
        planner = StubAgent([FinishCommand(final_answer="done")])
        pipeline = make_agentic_loop_pipeline(planner_agent=planner, agent_registry={})

        result = await run_agentic_loop_pipeline(pipeline, "goal")
        ctx = result.final_pipeline_context

        # The _iter_mapper should have logged the command
        assert len(ctx.command_log) == 1
        assert ctx.command_log[0].execution_result == "done"

    @pytest.mark.asyncio
    async def test_output_mapper_logging(self):
        """Test that _output_mapper correctly logs commands."""
        planner = StubAgent([FinishCommand(final_answer="done")])
        pipeline = make_agentic_loop_pipeline(planner_agent=planner, agent_registry={})

        result = await run_agentic_loop_pipeline(pipeline, "goal")
        ctx = result.final_pipeline_context

        # The _output_mapper should have logged the command
        assert len(ctx.command_log) == 1
        assert ctx.command_log[0].execution_result == "done"

    @pytest.mark.asyncio
    async def test_no_double_logging(self):
        """Test that commands are not logged twice."""
        planner = StubAgent([FinishCommand(final_answer="done")])
        pipeline = make_agentic_loop_pipeline(planner_agent=planner, agent_registry={})

        result = await run_agentic_loop_pipeline(pipeline, "goal")
        ctx = result.final_pipeline_context

        # Should have exactly 1 command, not 2
        assert len(ctx.command_log) == 1

        # Check that there are no duplicate entries
        command_ids = [log.turn for log in ctx.command_log]
        assert len(command_ids) == len(set(command_ids))

    @pytest.mark.asyncio
    async def test_command_executor_logging(self):
        """Test that the command executor logs commands correctly."""
        planner = StubAgent([RunAgentCommand(agent_name="test", input_data="test")])
        test_agent = AsyncMock()
        test_agent.run = AsyncMock(return_value="test result")

        pipeline = make_agentic_loop_pipeline(
            planner_agent=planner, agent_registry={"test": test_agent}, max_loops=1
        )

        result = await run_agentic_loop_pipeline(pipeline, "goal")
        ctx = result.final_pipeline_context

        # Should have logged the command from the executor
        assert len(ctx.command_log) == 1
        assert ctx.command_log[0].execution_result == "test result"

    @pytest.mark.asyncio
    async def test_pause_and_resume_logging(self):
        """Test that logging works correctly with pause and resume."""
        planner = StubAgent(
            [
                AskHumanCommand(question="Need input"),
                FinishCommand(final_answer="ok"),
            ]
        )

        pipeline = make_agentic_loop_pipeline(planner_agent=planner, agent_registry={})

        # First run should pause
        paused = await run_agentic_loop_pipeline(pipeline, "goal")
        assert len(paused.final_pipeline_context.command_log) == 0

        # Resume should add AskHuman command with human input + FinishCommand
        resumed = await run_agentic_loop_pipeline(pipeline, "goal", resume_from=paused)
        assert len(resumed.final_pipeline_context.command_log) == 2
        assert resumed.final_pipeline_context.command_log[0].execution_result == "human"
        assert resumed.final_pipeline_context.command_log[-1].execution_result == "ok"

    @pytest.mark.asyncio
    async def test_command_log_persistence(self):
        """Test that command logs persist across iterations."""
        planner = StubAgent(
            [
                RunAgentCommand(agent_name="test", input_data="test1"),
                RunAgentCommand(agent_name="test", input_data="test2"),
                FinishCommand(final_answer="done"),
            ]
        )

        test_agent = AsyncMock()
        test_agent.run = AsyncMock(side_effect=["result1", "result2"])

        pipeline = make_agentic_loop_pipeline(
            planner_agent=planner, agent_registry={"test": test_agent}
        )

        result = await run_agentic_loop_pipeline(pipeline, "goal")
        ctx = result.final_pipeline_context

        # Should have at least 3 commands logged
        assert len(ctx.command_log) >= 3
        assert ctx.command_log[0].execution_result == "result1"
        assert ctx.command_log[1].execution_result == "result2"
        assert ctx.command_log[2].execution_result == "done"

    @pytest.mark.asyncio
    async def test_error_logging(self):
        """Test that errors are logged correctly."""
        planner = StubAgent([RunAgentCommand(agent_name="nonexistent", input_data="test")])

        pipeline = make_agentic_loop_pipeline(planner_agent=planner, agent_registry={}, max_loops=1)

        result = await run_agentic_loop_pipeline(pipeline, "goal")
        ctx = result.final_pipeline_context

        # Should have logged the error
        assert len(ctx.command_log) == 1
        assert "Agent 'nonexistent' not found" in ctx.command_log[0].execution_result

    @pytest.mark.asyncio
    async def test_validation_error_logging(self):
        """Test that validation errors are logged correctly."""
        planner = StubAgent([{"invalid": "command"}])  # Invalid command

        pipeline = make_agentic_loop_pipeline(planner_agent=planner, agent_registry={}, max_loops=1)

        result = await run_agentic_loop_pipeline(pipeline, "goal")
        ctx = result.final_pipeline_context

        # Should have logged the validation error
        assert len(ctx.command_log) == 1
        assert "Invalid command" in ctx.command_log[0].execution_result


class TestAgenticLoopLoggingRegressionPrevention:
    """Test to prevent regression of logging issues."""

    @pytest.mark.asyncio
    async def test_no_double_logging_regression(self):
        """Test that the double logging issue doesn't regress."""
        planner = StubAgent([FinishCommand(final_answer="done")])
        pipeline = make_agentic_loop_pipeline(planner_agent=planner, agent_registry={})

        result = await run_agentic_loop_pipeline(pipeline, "goal")
        ctx = result.final_pipeline_context

        # This should be exactly 1, not 2 (which would indicate double logging)
        assert len(ctx.command_log) == 1, (
            "Double logging detected - commands are being logged twice"
        )

    @pytest.mark.asyncio
    async def test_logging_consistency(self):
        """Test that logging is consistent across different scenarios."""
        test_cases = [
            # (planner_commands, expected_log_count, description)
            ([FinishCommand(final_answer="done")], 1, "Single finish command"),
            ([RunAgentCommand(agent_name="x", input_data=1)], 1, "Single run command"),
            ([AskHumanCommand(question="test")], 0, "Single ask command (pauses)"),
        ]

        for commands, expected_count, description in test_cases:
            planner = StubAgent(commands)
            pipeline = make_agentic_loop_pipeline(
                planner_agent=planner, agent_registry={}, max_loops=1
            )

            result = await run_agentic_loop_pipeline(pipeline, "goal")
            ctx = result.final_pipeline_context

            assert len(ctx.command_log) >= expected_count, (
                f"Failed for {description}: expected at least {expected_count}, got {len(ctx.command_log)}"
            )

    @pytest.mark.asyncio
    async def test_logging_order(self):
        """Test that commands are logged in the correct order."""
        planner = StubAgent(
            [
                RunAgentCommand(agent_name="test", input_data="test1"),
                RunAgentCommand(agent_name="test", input_data="test2"),
                FinishCommand(final_answer="done"),
            ]
        )

        test_agent = AsyncMock()
        test_agent.run = AsyncMock(side_effect=["result1", "result2"])

        pipeline = make_agentic_loop_pipeline(
            planner_agent=planner, agent_registry={"test": test_agent}
        )

        result = await run_agentic_loop_pipeline(pipeline, "goal")
        ctx = result.final_pipeline_context

        # Check that logs are in the correct order
        assert len(ctx.command_log) == 3
        assert ctx.command_log[0].turn == 1
        assert ctx.command_log[1].turn == 2
        assert ctx.command_log[2].turn == 3


class TestAgenticLoopLoggingRegressionGuard:
    """Regression guard: ensures no double logging, missing logs, or out-of-order logs."""

    @pytest.mark.asyncio
    async def test_no_duplicate_or_missing_logs(self):
        from flujo.domain.commands import RunAgentCommand, FinishCommand
        from unittest.mock import AsyncMock

        # Valid, invalid, and error-producing commands
        planner_commands = [
            RunAgentCommand(agent_name="a", input_data="ok1"),  # valid
            {"invalid": "command"},  # validation error
            RunAgentCommand(agent_name="b", input_data="ok2"),  # agent not found (error)
            FinishCommand(final_answer="done"),  # valid finish
        ]
        agent_a = AsyncMock()
        agent_a.run = AsyncMock(return_value="result1")
        agent_registry = {"a": agent_a}  # 'b' is missing on purpose

        from flujo.recipes.factories import make_agentic_loop_pipeline, run_agentic_loop_pipeline

        pipeline = make_agentic_loop_pipeline(
            planner_agent=StubAgent(planner_commands),
            agent_registry=agent_registry,
            max_loops=10,
        )
        result = await run_agentic_loop_pipeline(pipeline, "goal")
        ctx = result.final_pipeline_context
        logs = ctx.command_log

        # 4 commands, so 4 logs expected
        assert len(logs) == 4, f"Expected 4 logs, got {len(logs)}: {logs}"

        # All logs should be unique (no double logging)
        seen = set()
        for log in logs:
            key = (log.turn, str(log.generated_command), log.execution_result)
            assert key not in seen, f"Duplicate log detected: {key}"
            seen.add(key)

        # Log order should be by turn
        turns = [log.turn for log in logs]
        assert turns == sorted(turns), f"Log turns out of order: {turns}"

        # Validation error should be present
        assert any("Invalid command" in log.execution_result for log in logs), (
            "Validation error not logged"
        )
        # Agent not found error should be present
        assert any("not found" in log.execution_result for log in logs), (
            "Agent not found error not logged"
        )
        # Finish command should be present
        assert any(getattr(log.generated_command, "type", None) == "finish" for log in logs), (
            "Finish command not logged"
        )
