"""
Minimal HITL in loop test - verify basic functionality works.
"""

import pytest
from flujo import Flujo
from flujo.domain.dsl import Pipeline, Step, LoopStep, HumanInTheLoopStep
from flujo.testing.utils import StubAgent


pytestmark = [pytest.mark.slow, pytest.mark.serial]


@pytest.mark.timeout(120)
@pytest.mark.asyncio
async def test_minimal_hitl_loop():
    """
    Absolute minimal test: just verify HITL in loop can pause and resume.
    """
    # Single agent step followed by HITL
    step1 = Step(name="work", agent=StubAgent(["done"]))
    step2 = HumanInTheLoopStep(name="hitl", message_for_user="OK?")
    
    # Loop that exits after first iteration
    # NOTE: Can't use >> operator with HITL, must use Pipeline(steps=[...])
    loop = LoopStep(
        name="simple_loop",
        loop_body_pipeline=Pipeline(steps=[step1, step2]),
        exit_condition_callable=lambda output, context: True,  # Always exit after first iteration
        max_loops=2
    )
    
    pipeline = Pipeline.from_step(loop)
    runner = Flujo(pipeline)
    
    # Run until pause
    paused = None
    async for res in runner.run_async(""):
        paused = res
        break
    
    print(f"Paused result: {paused}")
    print(f"Paused step_history: {paused.step_history if paused else 'None'}")
    print(f"Paused context status: {paused.final_pipeline_context.scratchpad.get('status') if paused and paused.final_pipeline_context else 'None'}")
    
    assert paused is not None, "Should get paused result"
    assert paused.final_pipeline_context is not None, "Should have context"
    assert paused.final_pipeline_context.scratchpad.get("status") == "paused", "Should be paused"
    
    # Resume
    resumed = await runner.resume_async(paused, "yes")
    
    print(f"Resumed result: {resumed}")
    print(f"Resumed step_history length: {len(resumed.step_history)}")
    print(f"Resumed step names: {[s.name for s in resumed.step_history]}")
    
    assert resumed is not None, "Should get resumed result"
    assert len(resumed.step_history) > 0, f"Should have step history, got: {resumed.step_history}"
    
    # Verify no nesting - should have exactly 1 loop step
    loop_steps = [s for s in resumed.step_history if s.name == "simple_loop"]
    print(f"Loop steps found: {len(loop_steps)}")
    
    assert len(loop_steps) == 1, (
        f"Expected 1 loop step (flat), got {len(loop_steps)}. "
        f"This indicates nested loops! Step names: {[s.name for s in resumed.step_history]}"
    )

