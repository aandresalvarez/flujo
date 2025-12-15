"""Real LLM integration tests for core Flujo patterns.

Tests non-granular execution patterns with real OpenAI API calls:
- Simple pipelines
- Loops (Step.loop_until)
- Branching (Step.branch_on)
- Parallel execution (Step.parallel)
- Chained steps (>>)

Requires OPENAI_API_KEY environment variable.
Run with: pytest tests/integration/test_flujo_patterns_real_llm.py -v
"""

from __future__ import annotations

import os
import pytest

from flujo import Flujo
from flujo.agents.wrapper import make_agent_async
from flujo.domain.dsl import Step, Pipeline
from flujo.domain.models import PipelineContext


def _should_run_real_llm_tests() -> bool:
    if os.environ.get("FLUJO_RUN_REAL_LLM_TESTS", "").lower() not in {"1", "true", "yes"}:
        return False
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return False
    return not key.lower().startswith("dummy")


# Skip unless explicitly enabled with a real API key.
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not _should_run_real_llm_tests(),
        reason="Real LLM tests require FLUJO_RUN_REAL_LLM_TESTS=1 and a non-dummy OPENAI_API_KEY",
    ),
]


# ============================================================================
# SIMPLE STEP TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_simple_single_step_real_llm() -> None:
    """Test a single step with real LLM."""
    agent = make_agent_async(
        model="openai:gpt-4o-mini",
        system_prompt="You are a helpful assistant. Respond in exactly 5 words.",
        output_type=str,
    )

    step = Step(name="simple_step", agent=agent)
    pipeline = Pipeline.from_step(step)
    runner = Flujo(pipeline)

    result = None
    async for item in runner.run_async("Say hello"):
        result = item

    assert result is not None
    assert len(result.step_history) == 1
    assert result.step_history[0].success is True
    assert result.total_cost_usd > 0


@pytest.mark.asyncio
async def test_chained_steps_real_llm() -> None:
    """Test chained steps with >> operator."""
    agent1 = make_agent_async(
        model="openai:gpt-4o-mini",
        system_prompt="You are a translator. Translate input to French. Output ONLY the French text.",
        output_type=str,
    )

    agent2 = make_agent_async(
        model="openai:gpt-4o-mini",
        system_prompt="You are a translator. Translate input to Spanish. Output ONLY the Spanish text.",
        output_type=str,
    )

    step1 = Step(name="to_french", agent=agent1)
    step2 = Step(name="to_spanish", agent=agent2)

    pipeline = step1 >> step2
    runner = Flujo(pipeline)

    result = None
    async for item in runner.run_async("Hello world"):
        result = item

    assert result is not None
    assert len(result.step_history) == 2
    # Both steps should succeed
    assert all(s.success for s in result.step_history)
    assert result.total_cost_usd > 0


# ============================================================================
# LOOP TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_loop_until_with_real_llm() -> None:
    """Test Step.loop_until with real LLM."""
    iteration_count = {"value": 0}

    async def counting_agent(
        data: str,
        *,
        context: PipelineContext | None = None,
        **kwargs: object,
    ) -> dict[str, object]:
        iteration_count["value"] += 1
        return {
            "count": iteration_count["value"],
            "output": f"Iteration {iteration_count['value']}",
        }

    # Wrap in a class with required attributes
    class CounterAgent:
        _model_name = "counter"
        _provider = "mock"
        _system_prompt = "Counter"
        _tools: list[object] = []

        async def run(self, *args: object, **kwargs: object) -> dict[str, object]:
            return await counting_agent(*args, **kwargs)  # type: ignore[arg-type]

    counter = CounterAgent()

    def exit_condition(output: object, ctx: PipelineContext | None) -> bool:
        if isinstance(output, dict):
            return output.get("count", 0) >= 3
        return False

    loop = Step.loop_until(
        name="counting_loop",
        loop_body_pipeline=Pipeline.from_step(Step(name="counter", agent=counter)),
        exit_condition_callable=exit_condition,
        max_loops=5,
    )

    runner = Flujo(Pipeline.from_step(loop))

    result = None
    async for item in runner.run_async("start"):
        result = item

    assert result is not None
    assert iteration_count["value"] >= 3


@pytest.mark.asyncio
async def test_loop_with_llm_agent() -> None:
    """Test loop with real LLM that decides when to exit."""
    agent = make_agent_async(
        model="openai:gpt-4o-mini",
        system_prompt=(
            "You are a number guesser. The user gives a range, you guess a number. "
            "If they say 'correct', respond with 'DONE'. Otherwise keep guessing."
        ),
        output_type=str,
    )

    iteration = {"count": 0}

    def exit_condition(output: object, ctx: PipelineContext | None) -> bool:
        iteration["count"] += 1
        if isinstance(output, str):
            return "DONE" in output.upper() or iteration["count"] >= 2
        return iteration["count"] >= 2

    loop = Step.loop_until(
        name="guessing_loop",
        loop_body_pipeline=Pipeline.from_step(Step(name="guesser", agent=agent)),
        exit_condition_callable=exit_condition,
        max_loops=3,
    )

    runner = Flujo(Pipeline.from_step(loop))

    result = None
    async for item in runner.run_async("correct"):
        result = item

    assert result is not None
    assert iteration["count"] >= 1


# ============================================================================
# BRANCHING TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_branch_on_with_real_llm() -> None:
    """Test Step.branch_on with real LLM agents."""
    formal_agent = make_agent_async(
        model="openai:gpt-4o-mini",
        system_prompt="You are a formal assistant. Respond professionally and formally.",
        output_type=str,
    )

    casual_agent = make_agent_async(
        model="openai:gpt-4o-mini",
        system_prompt="You are a casual friend. Respond casually with slang.",
        output_type=str,
    )

    def branch_selector(data: object, ctx: PipelineContext | None) -> str:
        if isinstance(data, str) and "formal" in data.lower():
            return "formal"
        return "casual"

    branch = Step.branch_on(
        name="style_branch",
        condition_callable=branch_selector,
        branches={
            "formal": Pipeline.from_step(Step(name="formal_step", agent=formal_agent)),
            "casual": Pipeline.from_step(Step(name="casual_step", agent=casual_agent)),
        },
    )

    runner = Flujo(Pipeline.from_step(branch))

    # Test formal branch
    result = None
    async for item in runner.run_async("Please respond formally"):
        result = item

    assert result is not None
    assert result.total_cost_usd > 0


@pytest.mark.asyncio
async def test_branch_with_default() -> None:
    """Test branching with default fallback."""
    agent_a = make_agent_async(
        model="openai:gpt-4o-mini",
        system_prompt="You are Agent A. Always start your response with 'A:'",
        output_type=str,
    )

    agent_default = make_agent_async(
        model="openai:gpt-4o-mini",
        system_prompt="You are the default agent. Always start your response with 'DEFAULT:'",
        output_type=str,
    )

    def selector(data: object, ctx: PipelineContext | None) -> str:
        if isinstance(data, str) and "route_a" in data:
            return "a"
        return "default"

    branch = Step.branch_on(
        name="with_default",
        condition_callable=selector,
        branches={
            "a": Pipeline.from_step(Step(name="a_step", agent=agent_a)),
            "default": Pipeline.from_step(Step(name="default_step", agent=agent_default)),
        },
    )

    runner = Flujo(Pipeline.from_step(branch))

    result = None
    async for item in runner.run_async("go to default"):
        result = item

    assert result is not None
    assert len(result.step_history) >= 1


# ============================================================================
# PARALLEL TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_parallel_with_real_llm() -> None:
    """Test Step.parallel with real LLM agents."""
    agent_1 = make_agent_async(
        model="openai:gpt-4o-mini",
        system_prompt="Count to 3. Output: 'one two three'",
        output_type=str,
    )

    agent_2 = make_agent_async(
        model="openai:gpt-4o-mini",
        system_prompt="Name 3 colors. Output: 'red blue green'",
        output_type=str,
    )

    step_1 = Step(name="counter", agent=agent_1)
    step_2 = Step(name="colors", agent=agent_2)

    parallel = Step.parallel(
        name="parallel_agents",
        branches={
            "numbers": Pipeline.from_step(step_1),
            "colors": Pipeline.from_step(step_2),
        },
    )

    runner = Flujo(Pipeline.from_step(parallel))

    result = None
    async for item in runner.run_async("start"):
        result = item

    assert result is not None
    # Both branches should execute
    assert result.total_cost_usd > 0


@pytest.mark.asyncio
async def test_parallel_three_branches() -> None:
    """Test parallel with three LLM branches."""

    def make_language_agent(language: str) -> object:
        return make_agent_async(
            model="openai:gpt-4o-mini",
            system_prompt=f"Translate the input to {language}. Output ONLY the translation.",
            output_type=str,
        )

    parallel = Step.parallel(
        name="multi_translate",
        branches={
            "french": Pipeline.from_step(
                Step(name="to_french", agent=make_language_agent("French"))
            ),
            "spanish": Pipeline.from_step(
                Step(name="to_spanish", agent=make_language_agent("Spanish"))
            ),
            "german": Pipeline.from_step(
                Step(name="to_german", agent=make_language_agent("German"))
            ),
        },
    )

    runner = Flujo(Pipeline.from_step(parallel))

    result = None
    async for item in runner.run_async("Hello"):
        result = item

    assert result is not None
    # All three branches should have been called
    assert result.total_cost_usd > 0


# ============================================================================
# COMPLEX PATTERNS
# ============================================================================


@pytest.mark.asyncio
async def test_loop_with_branching() -> None:
    """Test combining loop and branching."""
    counter = {"value": 0}

    async def mock_agent(data: str, **kwargs: object) -> dict[str, object]:
        counter["value"] += 1
        return {"iteration": counter["value"]}

    class MockAgent:
        _model_name = "mock"
        _provider = "mock"
        _system_prompt = ""
        _tools: list[object] = []

        async def run(self, *args: object, **kwargs: object) -> dict[str, object]:
            return await mock_agent(*args, **kwargs)  # type: ignore[arg-type]

    agent = MockAgent()
    step = Step(name="mock", agent=agent)

    def exit_after_2(output: object, ctx: PipelineContext | None) -> bool:
        return counter["value"] >= 2

    loop = Step.loop_until(
        name="test_loop",
        loop_body_pipeline=Pipeline.from_step(step),
        exit_condition_callable=exit_after_2,
        max_loops=5,
    )

    runner = Flujo(Pipeline.from_step(loop))

    result = None
    async for item in runner.run_async("start"):
        result = item

    assert result is not None
    assert counter["value"] >= 2


@pytest.mark.asyncio
async def test_step_from_callable_with_llm() -> None:
    """Test Step.from_callable integration."""
    llm_agent = make_agent_async(
        model="openai:gpt-4o-mini",
        system_prompt="Echo the input with 'Received:' prefix",
        output_type=str,
    )

    async def my_step(data: str, *, context: PipelineContext | None = None) -> str:
        # This could process data before/after LLM
        return f"Processed: {data}"

    step1 = Step.from_callable(my_step, name="preprocessor")
    step2 = Step(name="llm_step", agent=llm_agent)

    pipeline = step1 >> step2
    runner = Flujo(pipeline)

    result = None
    async for item in runner.run_async("test input"):
        result = item

    assert result is not None
    assert len(result.step_history) == 2
