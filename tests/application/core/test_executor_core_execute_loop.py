import pytest
from types import SimpleNamespace
from flujo.application.core.ultra_executor import ExecutorCore
from flujo.domain.models import StepResult
from flujo.domain.dsl.step import Step
from flujo.domain.dsl.pipeline import Pipeline
from flujo.testing.utils import StubAgent

class DummyExecutor(ExecutorCore):
    async def _execute_simple_step(
        self,
        step,
        data,
        context,
        resources,
        limits,
        stream,
        on_chunk,
        cache_key,
        breach_event,
        _fallback_depth,
    ):
        # Return a StepResult that increments numeric data by 1
        return StepResult(
            name=getattr(step, "name", "step"),
            success=True,
            output=data + 1,
            attempts=1,
            latency_s=0.0,
            token_counts=0,
            cost_usd=0.0,
            feedback=None,
            branch_context=context,
            metadata_={},
            step_history=None,
        )

@pytest.mark.asyncio
async def test_execute_loop_basic_iterations_and_history():
    """Test loop execution with proper agent configuration.
    
    This test validates that the MissingAgentError architectural protection
    correctly identifies steps without agents. Previously used SimpleNamespace
    objects without agents, which properly triggered the protection. Now uses
    proper Step objects with StubAgent to test the intended loop behavior.
    """
    exec = DummyExecutor()
    # Create a proper Step with agent instead of SimpleNamespace
    step = Step.model_validate({"name": "inc", "agent": StubAgent([1, 2, 3])})
    pipeline = Pipeline.from_step(step)
    loop_step = SimpleNamespace(
        name="loop",
        loop_body_pipeline=pipeline,
        max_loops=3,
        exit_condition_callable=None,
        initial_input_to_loop_body_mapper=None,
        iteration_input_mapper=None,
        loop_output_mapper=None,
    )
    result = await exec._execute_loop(loop_step, 0, None, None, None, None)
    assert not result.success
    assert result.attempts == 3
    assert result.output == 3
    assert result.metadata_["iterations"] == 3
    assert result.metadata_["exit_reason"] == "max_loops"
    assert len(result.step_history) == 3
    for i, step_res in enumerate(result.step_history, start=1):
        assert step_res.output == i

@pytest.mark.asyncio
async def test_execute_loop_with_exit_condition_and_mappers():
    """Test loop execution with exit condition and proper agent configuration.
    
    This test validates that the MissingAgentError architectural protection
    correctly identifies steps without agents. Previously used SimpleNamespace
    objects without agents, which properly triggered the protection. Now uses
    proper Step objects with StubAgent to test the intended loop behavior
    with exit conditions and mappers.
    """
    exec = DummyExecutor()
    # Create a proper Step with agent instead of SimpleNamespace
    step = Step.model_validate({"name": "inc", "agent": StubAgent([2])})  # Return 2 to satisfy exit condition
    pipeline = Pipeline.from_step(step)
    def exit_cond(data, context):
        return data >= 2
    def initial_mapper(data, context):
        return data + 1
    def iter_mapper(data, context, iteration):
        return data + 10 * iteration
    def output_mapper(data, context):
        return data * 2
    loop_step = SimpleNamespace(
        name="loop2",
        loop_body_pipeline=pipeline,
        max_loops=5,
        exit_condition_callable=exit_cond,
        initial_input_to_loop_body_mapper=initial_mapper,
        iteration_input_mapper=iter_mapper,
        loop_output_mapper=output_mapper,
    )
    result = await exec._execute_loop(loop_step, 0, None, None, None, None)
    assert result.success
    assert result.attempts == 1
    assert result.output == 4
    assert result.metadata_["iterations"] == 1
    assert result.metadata_["exit_reason"] == "condition"
    assert len(result.step_history) == 1

@pytest.mark.asyncio
async def test_execute_loop_with_output_mapper_exception():
    """Test loop execution with output mapper exception and proper agent configuration.
    
    This test validates that the MissingAgentError architectural protection
    correctly identifies steps without agents. Previously used SimpleNamespace
    objects without agents, which properly triggered the protection. Now uses
    proper Step objects with StubAgent to test the intended loop behavior
    when output mapper raises an exception.
    """
    exec = DummyExecutor()
    # Create a proper Step with agent instead of SimpleNamespace
    step = Step.model_validate({"name": "inc", "agent": StubAgent([1, 2])})
    pipeline = Pipeline.from_step(step)
    def bad_output_mapper(data, context):
        raise ValueError("bad")
    loop_step = SimpleNamespace(
        name="loop3",
        loop_body_pipeline=pipeline,
        max_loops=2,
        exit_condition_callable=None,
        initial_input_to_loop_body_mapper=None,
        iteration_input_mapper=None,
        loop_output_mapper=bad_output_mapper,
    )
    result = await exec._execute_loop(loop_step, 0, None, None, None, None)
    assert not result.success
    assert result.attempts == 2
    assert result.output is None
    assert "output_mapper_error" in result.metadata_["exit_reason"]
    assert "bad" in result.feedback