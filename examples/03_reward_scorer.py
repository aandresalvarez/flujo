"""
Demonstrates using a model-based reward scorer to evaluate a solution's quality.

This example departs from the default pipeline factory to show a more advanced pattern:
1. Build a simple, custom pipeline with the Flujo engine and DSL (`Step` and `Pipeline`).
2. Run the pipeline to generate a solution.
3. Use the `RewardScorer` to get a final quality score from an LLM judge.
   This is useful for tasks where quality is subjective (e.g., summarization).
"""

import asyncio
from typing import Any, cast

from flujo import Flujo, Step, Pipeline, make_agent_async, init_telemetry
from flujo.domain.scoring import RewardScorer
from flujo.exceptions import RewardModelUnavailable, FeatureDisabled

init_telemetry()


async def main():
    # 1. Define a simple agent. For this example, we only need a solution agent.
    solution_agent = make_agent_async(
        model="openai:gpt-4o-mini",
        system_prompt="You are an expert at explaining complex topics simply. Answer concisely.",
        output_type=str,
    )

    # 2. Define a minimal custom pipeline using the DSL.
    #    For more on the DSL, see docs/pipeline_dsl.md
    pipeline = Pipeline.from_step(
        Step.model_validate({"name": "GenerateExplanation", "agent": cast(Any, solution_agent)})
    )

    # 3. Initialize the Flujo engine with our custom pipeline.
    runner = Flujo(pipeline)

    # 4. Run the pipeline to get a solution.
    print("🧠 Running a simple pipeline to generate a solution...")
    result = None
    async for item in runner.run_async(
        "Explain 'technical debt' in one sentence for a non-technical manager."
    ):
        result = item

    if not result.step_history or not result.step_history[-1].success:
        print("\n❌ Pipeline failed to produce a solution.")
        return

    final_solution = result.step_history[-1].output
    print(f"\n📝 Generated Solution:\n'{final_solution}'")

    # 5. Use the RewardScorer to get a model-based quality score.
    #    The scorer uses an LLM to "judge" the output on a scale of 0.0 to 1.0.
    print("\n🤖 Asking a reward model (LLM judge) to score the solution's quality...")
    try:
        scorer = RewardScorer()
        score = await scorer.score(final_solution)
        print(f"\n🎉 Judged Quality Score: {score:.2f}")
    except (RewardModelUnavailable, FeatureDisabled) as e:
        print(f"\n⚠️ Could not get reward score: {e}")
        print("    Please ensure REWARD_ENABLED=true and OPENAI_API_KEY is set in your .env file.")


if __name__ == "__main__":
    asyncio.run(main())
