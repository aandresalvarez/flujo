"""
Integration tests for HITL in loops - PR #500 fix verification.

These tests ensure that HITL (Human-In-The-Loop) steps within loop bodies
work correctly with pause/resume, including:
- No nested loop creation on resume
- Agent outputs are captured
- Context/slots update correctly
- Data flow integrity across pause/resume
- Proper cleanup on completion

CRITICAL: These are regression tests for the bug where loops would create
nested instances on resume instead of continuing from the saved position.

Test Strategy:
- Mark as @pytest.mark.slow since they involve HITL state management
- Mark as @pytest.mark.serial to avoid SQLite contention
- Use real pause/resume flow to catch integration issues
- Verify trace structure shows no nesting
- Verify context state is maintained correctly
"""

import pytest
from flujo import Flujo
from flujo.domain.dsl import Pipeline, Step, LoopStep, HumanInTheLoopStep
from pydantic import BaseModel
from flujo.type_definitions.common import JSONObject


pytestmark = [pytest.mark.slow, pytest.mark.serial]


class SlotOutput(BaseModel):
    """Output schema for slot-filling agent."""

    action: str
    question: str
    slots: JSONObject


@pytest.mark.timeout(120)
async def test_hitl_in_loop_no_nesting_on_resume():
    """
    REGRESSION TEST: Verify loops don't create nested instances on HITL resume.

    This test specifically catches the bug where:
    1. Loop starts, agent asks question
    2. HITL pauses for user input
    3. On resume, loop created NESTED instance instead of continuing
    4. Result: infinite nesting, same question repeated

    Expected behavior after fix:
    1. Loop starts, agent asks question
    2. HITL pauses
    3. On resume, loop continues from saved position
    4. Agent sees updated context, asks DIFFERENT question
    """

    class AskQuestionAgent:
        _model_name = "ask-question"
        _provider = "test"

        async def run(
            self, _data: object, *, context: object | None = None, **_kw: object
        ) -> dict[str, object]:
            slots: object = {}
            try:
                import_artifacts = getattr(context, "import_artifacts", None)
                slots = (
                    getattr(import_artifacts, "slots", {}) if import_artifacts is not None else {}
                )
            except Exception:
                slots = {}

            metric = slots.get("metric") if isinstance(slots, dict) else None
            if metric:
                return {"action": "finish", "question": "Done", "slots": slots}
            return {"action": "ask", "question": "What metric?", "slots": slots}

    # Create a slot-filling pipeline similar to user's clarification pipeline
    pipeline = Pipeline(
        steps=[
            # Initialize state
            Step(
                name="init_state",
                agent={"id": "flujo.builtins.passthrough"},
                input="",
                updates_context=True,
                sink_to="import_artifacts",
                processors={
                    "output_processors": [
                        {
                            "type": "callable",
                            "callable": lambda x: {"slots": {}, "initial_goal": ""},
                        }
                    ]
                },
            ),
            # Loop with HITL inside
            LoopStep(
                name="clarification_loop",
                body=[
                    # Step 0: Agent asks clarifying question based on current slots
                    Step(
                        name="ask_question",
                        agent=AskQuestionAgent(),
                        input="Current slots: {{ context.import_artifacts.slots | tojson }}",
                        updates_context=True,
                        output_schema=SlotOutput.model_json_schema(),
                    ),
                    # Step 1: HITL asks user (this is where pause happens)
                    HumanInTheLoopStep(
                        name="ask_user", message="{{ steps.ask_question.output.question }}"
                    ),
                    # Step 2: Update slots with user response (this should execute after resume!)
                    Step(
                        name="update_slots",
                        agent={"id": "flujo.builtins.passthrough"},
                        input="{{ previous_step }}",
                        updates_context=True,
                        sink_to="import_artifacts.slots.metric",
                        processors={
                            "output_processors": [
                                {
                                    "type": "callable",
                                    "callable": lambda x: x,  # Pass through user input
                                }
                            ]
                        },
                    ),
                ],
                exit_expression="{{ context.import_artifacts.slots.metric is defined }}",
                max_loops=5,
            ),
        ]
    )

    runner = Flujo(pipeline)

    # First execution - should pause at HITL
    result1 = await runner.run_async("")

    # Verify paused
    assert result1.status == "paused", "Pipeline should pause at HITL"
    assert len(result1.step_history) >= 1, "Should have at least init step"

    # Verify context state before resume
    ctx1 = result1.final_pipeline_context
    assert ctx1.status == "paused", "Status should be paused"
    assert ctx1.loop_iteration_index is not None, "Should save loop iteration"
    assert ctx1.loop_step_index is not None, "Should save loop step index"
    assert ctx1.loop_last_output is not None, "Should save loop last output"

    # Resume with user input
    result2 = await runner.resume_async(result1, "prevalence")

    # CRITICAL VERIFICATION: Check for nested loops in trace
    # If bug exists, we'd see nested loop instances in step history
    loop_steps = [s for s in result2.step_history if s.name == "clarification_loop"]

    # Should have exactly ONE loop step result (not nested)
    assert len(loop_steps) == 1, (
        f"Expected 1 loop step result (flat), got {len(loop_steps)}. "
        f"This indicates nested loops were created! Step names: "
        f"{[s.name for s in result2.step_history]}"
    )

    # Verify slots were updated (proves step 2 executed after resume)
    ctx2 = result2.final_pipeline_context
    slots = ctx2.import_artifacts.get("slots")
    assert isinstance(slots, dict), "Should have slots"
    assert slots.get("metric") == "prevalence", (
        f"Slots should be updated with user input. Got: {slots}"
    )

    # Verify loop completed successfully
    loop_result = loop_steps[0]
    assert loop_result.success, f"Loop should succeed. Feedback: {loop_result.feedback}"

    # Verify cleanup - loop state should be cleared
    assert ctx2.status != "paused", "Status should not be paused after completion"
    assert ctx2.loop_iteration_index is None, "Loop iteration should be cleared after completion"
    assert ctx2.loop_step_index is None, "Loop step index should be cleared after completion"
    assert ctx2.loop_last_output is None, "Loop last output should be cleared after completion"


@pytest.mark.timeout(120)
async def test_hitl_in_loop_agent_output_captured():
    """
    REGRESSION TEST: Verify agent outputs are captured when HITL is in loop.

    The bug caused agent steps to restart on resume without completing,
    so no agent.output events were captured.

    Expected: Agent step completes BEFORE HITL pause, output is captured.
    """
    executed_steps: list[str] = []

    def track_execution(step_name: str):
        """Track which steps actually execute."""

        def tracker(x):
            executed_steps.append(step_name)
            return {"output": f"{step_name}_completed", "data": x}

        return tracker

    pipeline = Pipeline(
        steps=[
            LoopStep(
                name="test_loop",
                body=[
                    # Agent step - should complete before HITL
                    Step(
                        name="agent_step",
                        agent={"id": "flujo.builtins.passthrough"},
                        input="test_input",
                        updates_context=True,
                        processors={
                            "output_processors": [
                                {"type": "callable", "callable": track_execution("agent_step")}
                            ]
                        },
                    ),
                    # HITL step
                    HumanInTheLoopStep(name="hitl_step", message="Please respond"),
                    # Update step - should execute after resume
                    Step(
                        name="update_step",
                        agent={"id": "flujo.builtins.passthrough"},
                        input="{{ previous_step }}",
                        updates_context=True,
                        processors={
                            "output_processors": [
                                {"type": "callable", "callable": track_execution("update_step")}
                            ]
                        },
                    ),
                ],
                exit_expression="{{ context.steps.update_step is defined }}",
                max_loops=3,
            )
        ]
    )

    runner = Flujo(pipeline)

    # Execute until pause
    result1 = await runner.run_async("")
    assert result1.status == "paused"

    # Verify agent_step executed BEFORE pause
    assert "agent_step" in executed_steps, (
        f"Agent step should execute before HITL pause. Executed steps: {executed_steps}"
    )

    # Verify agent output is in context
    ctx1 = result1.final_pipeline_context
    assert hasattr(ctx1, "steps"), "Context should have steps"
    assert "agent_step" in ctx1.steps, "Agent step output should be in context"
    agent_output = ctx1.steps["agent_step"]
    assert agent_output is not None, "Agent output should not be None"
    assert isinstance(agent_output, dict), "Agent output should be captured as dict"
    assert agent_output.get("output") == "agent_step_completed", (
        f"Agent output should be captured correctly. Got: {agent_output}"
    )

    # Resume
    _ = await runner.resume_async(result1, "user_response")

    # Verify update_step executed after resume
    assert "update_step" in executed_steps, (
        f"Update step should execute after resume. Executed steps: {executed_steps}"
    )

    # Verify no duplicate executions (would indicate loop restarted)
    agent_count = executed_steps.count("agent_step")
    assert agent_count == 1, (
        f"Agent step should execute exactly once, got {agent_count} times. "
        f"This indicates loop restarted on resume! Steps: {executed_steps}"
    )


@pytest.mark.timeout(120)
async def test_hitl_in_loop_data_flow_integrity():
    """
    REGRESSION TEST: Verify data flows correctly across pause/resume in loops.

    The bug caused human input to be passed as loop data on resume,
    breaking the data flow and causing agent to restart.

    Expected: Loop data preserved, human input goes to HITL step only.
    """
    pipeline = Pipeline(
        steps=[
            LoopStep(
                name="data_flow_loop",
                body=[
                    # Step 0: Process data and output a value
                    Step(
                        name="process",
                        agent={"id": "flujo.builtins.passthrough"},
                        input="initial_data",
                        updates_context=True,
                        sink_to="processed_value",
                        processors={
                            "output_processors": [
                                {
                                    "type": "callable",
                                    "callable": lambda x: {"value": f"processed_{x}", "count": 1},
                                }
                            ]
                        },
                    ),
                    # Step 1: HITL shows the processed value
                    HumanInTheLoopStep(
                        name="show_value", message="Value: {{ context.processed_value.value }}"
                    ),
                    # Step 2: Verify we got human input, not loop data
                    Step(
                        name="verify_input",
                        agent={"id": "flujo.builtins.passthrough"},
                        input="{{ previous_step }}",
                        updates_context=True,
                        sink_to="verified",
                        processors={
                            "output_processors": [
                                {
                                    "type": "callable",
                                    "callable": lambda x: {
                                        "received": x,
                                        "is_human_input": x == "human_response",
                                    },
                                }
                            ]
                        },
                    ),
                ],
                exit_expression="{{ context.verified.is_human_input == true }}",
                max_loops=3,
            )
        ]
    )

    runner = Flujo(pipeline)

    # Execute until pause
    result1 = await runner.run_async("")
    assert result1.status == "paused"

    # Verify processed value is in context
    ctx1 = result1.final_pipeline_context
    assert hasattr(ctx1, "processed_value"), "Processed value should be in context"
    assert ctx1.processed_value["value"] == "processed_initial_data"

    # Verify loop saved the processed data (not human input)
    assert ctx1.loop_last_output is not None, "Loop should save last output"
    saved_output = ctx1.loop_last_output
    assert saved_output is not None, "Saved output should not be None"
    # The saved output should be the agent's output, not None or human input

    # Resume with human input
    result2 = await runner.resume_async(result1, "human_response")

    # CRITICAL: Verify human input went to verify_input step, not to process step
    ctx2 = result2.final_pipeline_context
    assert hasattr(ctx2, "verified"), "Verified should be in context"
    assert ctx2.verified["received"] == "human_response", (
        f"Step should receive human input. Got: {ctx2.verified['received']}"
    )
    assert ctx2.verified["is_human_input"] is True, (
        "Step should confirm it received human input, not loop data"
    )

    # Verify processed_value is still correct (didn't get overwritten by human input)
    assert ctx2.processed_value["value"] == "processed_initial_data", (
        f"Processed value should be preserved. Got: {ctx2.processed_value}"
    )


@pytest.mark.timeout(180)
async def test_hitl_in_loop_multiple_iterations():
    """
    REGRESSION TEST: Verify loops can handle multiple HITL pauses across iterations.

    This tests the full cycle:
    1. Iteration 1: pause, resume, complete
    2. Iteration 2: pause, resume, complete
    3. Exit condition met

    The bug would cause each resume to create nested loops, preventing
    the loop from progressing through iterations.
    """
    iteration_data: list[JSONObject] = []

    def track_iteration(iteration_num: int):
        """Track data for each iteration."""

        def tracker(x):
            data = {"iteration": iteration_num, "input": x, "timestamp": len(iteration_data)}
            iteration_data.append(data)
            return data

        return tracker

    pipeline = Pipeline(
        steps=[
            # Init counter
            Step(
                name="init",
                agent={"id": "flujo.builtins.passthrough"},
                input="0",
                updates_context=True,
                sink_to="counter",
                processors={
                    "output_processors": [{"type": "callable", "callable": lambda x: int(x)}]
                },
            ),
            LoopStep(
                name="multi_iteration_loop",
                body=[
                    # Increment counter
                    Step(
                        name="increment",
                        agent={"id": "flujo.builtins.passthrough"},
                        input="{{ context.counter | default(0) }}",
                        updates_context=True,
                        processors={
                            "output_processors": [
                                {
                                    "type": "callable",
                                    "callable": lambda x: track_iteration(x + 1)
                                    if isinstance(x, int)
                                    else track_iteration(1),
                                }
                            ]
                        },
                    ),
                    # HITL asks to continue
                    HumanInTheLoopStep(
                        name="ask_continue",
                        message="Count: {{ steps.increment.output.iteration }}. Continue?",
                    ),
                    # Update counter
                    Step(
                        name="update_counter",
                        agent={"id": "flujo.builtins.passthrough"},
                        input="{{ steps.increment.output.iteration }}",
                        updates_context=True,
                        sink_to="counter",
                    ),
                ],
                exit_expression="{{ context.counter >= 3 }}",
                max_loops=5,
            ),
        ]
    )

    runner = Flujo(pipeline)

    # Execute until first pause (iteration 1)
    result = await runner.run_async("")
    assert result.status == "paused"

    # Resume iteration 1
    result = await runner.resume_async(result, "yes")
    # Should pause again in iteration 2
    assert result.status == "paused", "Should pause in iteration 2"

    # Resume iteration 2
    result = await runner.resume_async(result, "yes")
    # Should pause again in iteration 3
    assert result.status == "paused", "Should pause in iteration 3"

    # Resume iteration 3
    result = await runner.resume_async(result, "yes")
    # Should complete (counter >= 3)
    assert result.status in [
        "completed",
        "paused",
    ], f"Should complete or pause. Got: {result.status}"

    # Verify we went through 3 distinct iterations (not nested)
    assert len(iteration_data) == 3, (
        f"Should have 3 iterations. Got {len(iteration_data)}: {iteration_data}"
    )

    # Verify iterations incremented correctly (proves no nesting)
    iterations = [d["iteration"] for d in iteration_data]
    assert iterations == [1, 2, 3], (
        f"Iterations should be [1, 2, 3]. Got {iterations}. "
        f"If nested, we'd see [1, 1, 1] or [1, 1, 2, 1, 2, 3]"
    )

    # Verify final counter value
    ctx = result.final_pipeline_context
    assert ctx.counter == 3, f"Counter should be 3. Got: {ctx.counter}"


@pytest.mark.timeout(120)
async def test_hitl_in_loop_cleanup_on_completion():
    """
    TEST: Verify loop state is properly cleaned up on completion.

    After loop completes, the resume state (loop_iteration, loop_step_index,
    loop_last_output) should be cleared so next execution doesn't think
    it's resuming.
    """
    pipeline = Pipeline(
        steps=[
            LoopStep(
                name="cleanup_test_loop",
                body=[
                    Step(
                        name="work",
                        agent={"id": "flujo.builtins.passthrough"},
                        input="data",
                        updates_context=True,
                        sink_to="done",
                    ),
                    HumanInTheLoopStep(name="confirm", message="Done?"),
                ],
                exit_expression="{{ previous_step == 'yes' }}",
                max_loops=2,
            )
        ]
    )

    runner = Flujo(pipeline)

    # Execute and complete the loop
    result1 = await runner.run_async("")
    assert result1.status == "paused"

    result2 = await runner.resume_async(result1, "yes")

    # Verify completion
    assert result2.status in ["completed", "failed"], f"Should complete. Got: {result2.status}"

    # CRITICAL: Verify cleanup happened
    ctx = result2.final_pipeline_context

    # Check that loop resume state is cleared
    assert ctx.loop_iteration_index is None, (
        f"loop_iteration should be cleared. Got: {ctx.loop_iteration_index}"
    )
    assert ctx.loop_step_index is None, (
        f"loop_step_index should be cleared. Got: {ctx.loop_step_index}"
    )
    assert ctx.loop_last_output is None, (
        f"loop_last_output should be cleared. Got: {ctx.loop_last_output}"
    )

    # Status should be completed, not paused
    if ctx.status is not None:
        assert ctx.status != "paused", (
            f"Status should not be paused after completion. Got: {ctx.status}"
        )


@pytest.mark.timeout(120)
async def test_hitl_in_loop_resume_at_correct_step():
    """
    TEST: Verify resume continues from the exact saved step position.

    If loop has 3 steps and pauses at step 1, resume should continue
    from step 2 (not restart from step 0).
    """
    executed_steps: list[tuple] = []  # (iteration, step_name)

    def track(step_name: str):
        def tracker(x):
            # Get current iteration from context if available
            base_iter = len([s for s in executed_steps if s[1] == "step0"])
            if step_name != "step0":
                base_iter = max(0, base_iter - 1)
            executed_steps.append((base_iter, step_name))
            return f"{step_name}_done"

        return tracker

    pipeline = Pipeline(
        steps=[
            LoopStep(
                name="position_test_loop",
                body=[
                    Step(
                        name="step0",
                        agent={"id": "flujo.builtins.passthrough"},
                        input="data",
                        processors={
                            "output_processors": [{"type": "callable", "callable": track("step0")}]
                        },
                    ),
                    HumanInTheLoopStep(name="step1_hitl", message="Continue?"),
                    Step(
                        name="step2",
                        agent={"id": "flujo.builtins.passthrough"},
                        input="{{ previous_step }}",
                        updates_context=True,
                        sink_to="result",
                        processors={
                            "output_processors": [{"type": "callable", "callable": track("step2")}]
                        },
                    ),
                ],
                exit_expression="{{ context.result is defined }}",
                max_loops=2,
            )
        ]
    )

    runner = Flujo(pipeline)

    # Execute until pause
    result1 = await runner.run_async("")
    assert result1.status == "paused"

    # Verify step0 executed in iteration 0
    assert (
        0,
        "step0",
    ) in executed_steps, f"step0 should execute before pause. Got: {executed_steps}"

    # Verify step2 has NOT executed yet
    assert (
        0,
        "step2",
    ) not in executed_steps, f"step2 should NOT execute before pause. Got: {executed_steps}"

    # Verify saved position
    ctx1 = result1.final_pipeline_context
    saved_index = ctx1.loop_step_index
    assert saved_index == 2, f"Should save step index 2 (next after HITL). Got: {saved_index}"

    # Resume
    _ = await runner.resume_async(result1, "yes")

    # CRITICAL: Verify step2 executed ONCE (not restarted)
    step2_executions = [s for s in executed_steps if s[1] == "step2"]
    assert len(step2_executions) == 1, (
        f"step2 should execute exactly once. Got {len(step2_executions)} times: {executed_steps}"
    )

    # Verify step0 did NOT execute again (would indicate restart)
    step0_executions = [s for s in executed_steps if s[1] == "step0"]
    assert len(step0_executions) == 1, (
        f"step0 should execute exactly once. Got {len(step0_executions)} times: {executed_steps}. "
        f"Multiple executions indicate loop restarted on resume!"
    )

    # Verify execution order is correct
    assert executed_steps == [
        (0, "step0"),
        (0, "step2"),
    ], f"Execution order should be step0, then step2 (skipping HITL). Got: {executed_steps}"
