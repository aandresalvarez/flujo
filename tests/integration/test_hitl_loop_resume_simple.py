"""
Simplified HITL in loops regression tests - PR #500 fix verification.

These tests verify the critical bug fix: loops don't create nested instances on HITL resume.

CRITICAL BUG FIXED:
- Before: Loop would create nested instances on resume (visible in trace as child loops)
- After: Loop continues from saved position (flat trace structure)

Markers:
- @pytest.mark.slow: HITL involves state management (60-180s per test)
- @pytest.mark.serial: Avoid SQLite contention
"""

import pytest
from flujo import Flujo
from flujo.domain.dsl import Pipeline, Step, LoopStep, HumanInTheLoopStep
from flujo.testing.utils import StubAgent


pytestmark = [pytest.mark.slow, pytest.mark.serial]


@pytest.mark.timeout(120)
@pytest.mark.asyncio
async def test_hitl_in_loop_no_nesting_simple():
    """
    CRITICAL REGRESSION TEST: Verify loop doesn't nest on HITL resume.

    This is the simplest possible test that catches the bug:
    1. Loop with 2 steps: agent + HITL
    2. Execute until HITL pause
    3. Resume with user input
    4. Verify NO nested loops in result

    If bug exists: step_history will have multiple loop results (nested)
    If fixed: step_history will have exactly 1 loop result (flat)
    """
    # Create body steps: agent then HITL
    agent_step = Step(name="agent", agent=StubAgent(["agent_output"]))

    hitl_step = HumanInTheLoopStep(name="hitl", message_for_user="Respond please")

    # Create pipeline with both steps
    body_pipeline = agent_step >> hitl_step

    # Create loop that exits after one iteration
    loop = LoopStep(
        name="test_loop",
        loop_body_pipeline=body_pipeline,
        exit_condition_callable=lambda output, context: output == "user_input",
        max_loops=3,
    )

    # Wrap in pipeline
    pipeline = Pipeline.from_step(loop)
    runner = Flujo(pipeline)

    # Execute until pause
    result1 = None
    async for res in runner.run_async(""):
        result1 = res
        break  # Will break on first pause

    assert result1 is not None, "Should get a result"

    # CRITICAL CHECK: Verify state shows paused
    ctx1 = result1.final_pipeline_context
    assert hasattr(ctx1, "scratchpad"), "Context should have scratchpad"
    assert ctx1.scratchpad.get("status") == "paused", "Should pause at HITL"

    # Resume
    result2 = await runner.resume_async(result1, "user_input")

    # CRITICAL VERIFICATION: Check for nested loops
    loop_steps = [s for s in result2.step_history if s.name == "test_loop"]

    assert len(loop_steps) == 1, (
        f"Expected 1 loop step (flat structure), got {len(loop_steps)}. "
        f"Multiple loop steps indicate nested loops! "
        f"Step names: {[s.name for s in result2.step_history]}"
    )

    # Verify success
    assert loop_steps[0].success, f"Loop should succeed. Feedback: {loop_steps[0].feedback}"


@pytest.mark.timeout(120)
@pytest.mark.asyncio
async def test_hitl_in_loop_agent_output_captured_simple():
    """
    REGRESSION TEST: Verify agent output is captured before HITL pause.

    Bug symptom: No agent.output events because agent restarted on resume.
    Fix verification: Agent output should be in context before pause.
    """
    call_count = [0]  # Track agent calls

    def counting_agent():
        """Agent that tracks how many times it's called."""

        def run(payload, context=None, resources=None, **kwargs):
            call_count[0] += 1
            return {"output": f"call_{call_count[0]}", "count": call_count[0]}

        class Agent:
            async def run(self, payload, context=None, resources=None, **kwargs):
                return run(payload, context, resources, **kwargs)

        return Agent()

    # Create steps
    agent_step = Step(name="agent", agent=counting_agent())
    hitl_step = HumanInTheLoopStep(name="hitl", message_for_user="Continue?")

    # Create loop
    body_pipeline = agent_step >> hitl_step
    loop = LoopStep(
        name="test_loop",
        loop_body_pipeline=body_pipeline,
        exit_condition_callable=lambda output, context: True,  # Exit after first iteration
        max_loops=2,
    )

    pipeline = Pipeline.from_step(loop)
    runner = Flujo(pipeline)

    # Execute until pause
    result1 = None
    async for res in runner.run_async(""):
        result1 = res
        break

    assert result1 is not None, "Should get a result"

    # Verify paused
    ctx_paused = result1.final_pipeline_context
    assert hasattr(ctx_paused, "scratchpad"), "Context should have scratchpad"
    assert ctx_paused.scratchpad.get("status") == "paused", "Should be paused"

    # CRITICAL: Verify agent was called ONCE before pause
    assert call_count[0] == 1, f"Agent should be called once before pause, got {call_count[0]}"

    # Verify agent output is captured in context
    ctx1 = result1.final_pipeline_context
    if hasattr(ctx1, "steps") and "agent" in ctx1.steps:
        agent_output = ctx1.steps["agent"]
        assert agent_output is not None, "Agent output should be captured"
        assert isinstance(agent_output, dict), "Agent output should be a dict"
        assert agent_output.get("count") == 1, (
            f"Agent output should show call 1, got: {agent_output}"
        )

    # Resume
    _ = await runner.resume_async(result1, "yes")

    # CRITICAL: Verify agent was NOT called again (no restart)
    assert call_count[0] == 1, (
        f"Agent should still be called only once after resume, got {call_count[0]}. "
        f"Multiple calls indicate loop restarted on resume!"
    )


@pytest.mark.timeout(180)
@pytest.mark.asyncio
async def test_hitl_in_loop_multiple_iterations_simple():
    """
    REGRESSION TEST: Verify loop progresses through multiple iterations with HITL.

    Bug symptom: Iterations would be [1,1,1...] (nested) instead of [1,2,3] (sequential).
    Fix verification: Track iteration numbers to verify sequential progression.
    """
    iteration_numbers = []

    def tracking_agent():
        """Agent that tracks iteration numbers."""

        def run(payload, context=None, resources=None, **kwargs):
            # Get current counter from context or default to 0
            if context and hasattr(context, "counter"):
                current = context.counter
            else:
                current = 0

            next_num = current + 1
            iteration_numbers.append(next_num)
            return next_num

        class Agent:
            async def run(self, payload, context=None, resources=None, **kwargs):
                return run(payload, context, resources, **kwargs)

        return Agent()

    # Create steps
    agent_step = Step(
        name="increment", agent=tracking_agent(), updates_context=True, sink_to="counter"
    )
    hitl_step = HumanInTheLoopStep(name="hitl", message_for_user="Continue?")

    # Create loop that runs 3 iterations
    body_pipeline = agent_step >> hitl_step
    loop = LoopStep(
        name="multi_iter_loop",
        loop_body_pipeline=body_pipeline,
        exit_condition_callable=lambda output, context: (
            hasattr(context, "counter") and context.counter >= 3
        ),
        max_loops=5,
    )

    pipeline = Pipeline.from_step(loop)
    runner = Flujo(pipeline)

    # Iteration 1: execute and resume
    result = None
    async for res in runner.run_async(""):
        result = res
        break

    assert result is not None, "Should get a result"

    # Verify paused in iteration 1
    assert result.final_pipeline_context.scratchpad.get("status") == "paused", (
        "Should pause in iteration 1"
    )
    result = await runner.resume_async(result, "yes")

    # Iteration 2: pause and resume
    if result.final_pipeline_context.scratchpad.get("status") == "paused":
        result = await runner.resume_async(result, "yes")

    # Iteration 3: pause and resume
    if result.final_pipeline_context.scratchpad.get("status") == "paused":
        result = await runner.resume_async(result, "yes")

    # CRITICAL VERIFICATION: Check iteration progression
    assert len(iteration_numbers) >= 3, (
        f"Should have at least 3 iterations. Got {len(iteration_numbers)}: {iteration_numbers}"
    )

    # Verify sequential progression (not nested)
    # If bug exists, we'd see [1,1,1] or [1,1,2,1,2,3]
    # If fixed, we see [1,2,3]
    first_three = iteration_numbers[:3]
    assert first_three == [1, 2, 3], (
        f"Iterations should be sequential [1,2,3]. Got {first_three}. "
        f"Non-sequential indicates nested loops! Full list: {iteration_numbers}"
    )


@pytest.mark.timeout(120)
@pytest.mark.asyncio
async def test_hitl_in_loop_state_cleanup_simple():
    """
    TEST: Verify loop state is cleaned up after completion.

    After loop completes, resume state should be cleared so next run
    doesn't think it's resuming.
    """
    # Simple loop: agent + HITL that exits after one iteration
    agent_step = Step(name="work", agent=StubAgent(["done"]))
    hitl_step = HumanInTheLoopStep(name="hitl", message_for_user="Done?")

    body_pipeline = agent_step >> hitl_step
    loop = LoopStep(
        name="cleanup_loop",
        loop_body_pipeline=body_pipeline,
        exit_condition_callable=lambda output, context: output == "yes",
        max_loops=2,
    )

    pipeline = Pipeline.from_step(loop)
    runner = Flujo(pipeline)

    # Execute and complete
    result1 = None
    async for res in runner.run_async(""):
        result1 = res
        break

    assert result1 is not None, "Should get a result"

    # Verify paused
    assert result1.final_pipeline_context.scratchpad.get("status") == "paused", "Should be paused"

    result2 = await runner.resume_async(result1, "yes")

    # CRITICAL: Verify cleanup
    ctx = result2.final_pipeline_context
    scratchpad = ctx.scratchpad if hasattr(ctx, "scratchpad") else {}

    # Check loop state is cleared
    assert scratchpad.get("loop_iteration") is None, (
        f"loop_iteration should be cleared. Got: {scratchpad.get('loop_iteration')}"
    )
    assert scratchpad.get("loop_step_index") is None, (
        f"loop_step_index should be cleared. Got: {scratchpad.get('loop_step_index')}"
    )
    assert scratchpad.get("loop_last_output") is None, (
        f"loop_last_output should be cleared. Got: {scratchpad.get('loop_last_output')}"
    )
