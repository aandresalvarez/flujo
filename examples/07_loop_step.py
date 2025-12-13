"""
Demonstrates using a `LoopStep` for iterative refinement.

The pipeline will generate text, then loop through a "review and edit"
cycle until the text meets a quality standard or the max number of
loops is reached. For more details, see docs/pipeline_looping.md.
"""

import asyncio

from pydantic import BaseModel

from flujo import Flujo, Step, Pipeline, step
from flujo.models import PipelineResult


class TextEdit(BaseModel):
    """A model to hold the text and feedback during the loop."""

    text: str
    feedback: str = "No feedback yet."
    is_good_enough: bool = False


# Agents for our iterative workflow
@step(name="GenerateInitialDraft")
async def generate_text_agent(prompt: str) -> TextEdit:
    """Generates the initial draft."""
    print("✍️ Generating initial draft...")
    return TextEdit(text="Python is a language.")


@step(name="EditAndReview")
async def edit_and_review_agent(draft: TextEdit) -> TextEdit:
    """Reviews text. If it's too short, it adds to it and provides feedback."""
    print(f"🧐 Reviewing draft: '{draft.text}'")
    if len(draft.text) < 30:
        print("   -> Draft is too short. Adding more detail.")
        draft.text += " It is known for its simple syntax."
        draft.feedback = "The text was too short, so I expanded it."
        draft.is_good_enough = False
    else:
        print("   -> Draft meets length requirement. Exiting loop.")
        draft.feedback = "The text looks good."
        draft.is_good_enough = True
    return draft


# The body of our loop is a single "edit and review" step.
loop_body_pipeline = Pipeline.from_step(edit_and_review_agent)
# Explicitly type loop body output to avoid generic object chaining complaints
edit_and_review_agent.__step_output_type__ = TextEdit

# The `LoopStep` will run the `loop_body_pipeline` repeatedly.
loop_step = Step.loop_until(
    name="IterativeRefinementLoop",
    loop_body_pipeline=loop_body_pipeline,
    # The `exit_condition_callable` checks the output of the last step in the
    # loop body. It receives that output and the optional pipeline context.
    exit_condition_callable=lambda body_output, ctx: body_output.is_good_enough,
    max_loops=5,  # A safeguard to prevent infinite loops.
)
loop_step.__step_output_type__ = TextEdit
generate_text_agent.__step_output_type__ = TextEdit
loop_step.meta = {"is_adapter": True, "adapter_id": "generic-adapter", "adapter_allow": "generic"}

# The full pipeline: generate an initial version, then enter the refinement loop.
full_pipeline = generate_text_agent >> loop_step

runner = Flujo(full_pipeline)


async def main() -> None:
    print("🚀 Starting iterative refinement pipeline...\n")
    result: PipelineResult | None = None
    async for item in runner.run_async("Write a sentence about Python."):
        result = item

    print("\n✅ Pipeline finished!")
    assert result is not None
    final_result = result.step_history[-1]
    output = final_result.output

    print(f"\nFinal Output: '{output.text}'")
    print(f"Final Feedback: '{output.feedback}'")
    print(f"Iterations (attempts): {final_result.attempts}")


if __name__ == "__main__":
    asyncio.run(main())
