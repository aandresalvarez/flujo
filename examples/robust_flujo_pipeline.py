import asyncio
from flujo import make_agent_async, Step, Task, Pipeline, step, Flujo, init_telemetry
from flujo.domain.models import Checklist
from flujo.domain.scoring import weighted_score, ratio_score
from pydantic import BaseModel

init_telemetry()


# Define a context model for our pipeline
class CodeReviewContext(BaseModel):
    solution: str = ""
    review: Checklist = None
    validation: Checklist = None
    weighted_score: float = 0.0
    ratio_score: float = 0.0


# Create agents using Flujo's make_agent_async
solution_agent = make_agent_async(
    "openai:gpt-4o",
    "You are a Python developer. Write a function as requested.",
    str,
)

review_agent = make_agent_async(
    "openai:gpt-4o",
    (
        "You are a code reviewer. For the given function, return a Checklist with these items: "
        "'Has docstring', 'Handles edge cases', 'Uses idiomatic Python', 'Is correct', 'Has tests'. "
        "For each, set 'passed' to true or false."
    ),
    Checklist,
)

validator_agent = make_agent_async(
    "openai:gpt-4o",
    (
        "You are a strict validator. For the given function, return a Checklist with the same items as the reviewer. "
        "Be strict: only set 'passed' to true if the requirement is fully met."
    ),
    Checklist,
)

repair_agent = make_agent_async(
    "openai:gpt-4o",
    (
        "You are a code repair agent. Given a function and a checklist of failed items, rewrite the function to fix the issues. "
        "Return only the improved code as a string."
    ),
    str,
)


# Define steps using Flujo's @step decorator
@step(name="GenerateSolution")
async def generate_solution(prompt: str) -> CodeReviewContext:
    """Generate initial solution."""
    print("🔧 Generating initial solution...")
    solution = await solution_agent.run(prompt)
    return CodeReviewContext(solution=solution)


@step(name="ReviewCode")
async def review_code(ctx: CodeReviewContext) -> CodeReviewContext:
    """Review the code and update context."""
    print("📋 Reviewing code...")
    review = await review_agent.run(ctx.solution)
    return CodeReviewContext(solution=ctx.solution, review=review)


@step(name="ValidateCode")
async def validate_code(ctx: CodeReviewContext) -> CodeReviewContext:
    """Validate the code and update context."""
    print("✅ Validating code...")
    validation = await validator_agent.run(ctx.solution)
    return CodeReviewContext(solution=ctx.solution, review=ctx.review, validation=validation)


@step(name="RepairCode")
async def repair_code(ctx: CodeReviewContext) -> CodeReviewContext:
    """Repair the code based on validation failures."""
    print("🔧 Repairing code...")
    failed_items = [item.description for item in ctx.validation.items if not item.passed]
    repair_prompt = f"Function:\n{ctx.solution}\n\nFailed items: {failed_items}"
    repaired = await repair_agent.run(repair_prompt)
    return CodeReviewContext(solution=repaired, review=ctx.review, validation=ctx.validation)


@step(name="ScoreSolution")
async def score_solution(ctx: CodeReviewContext) -> CodeReviewContext:
    """Score the final solution."""
    print("📊 Scoring solution...")
    weights = [
        {"item": "Has docstring", "weight": 2.0},
        {"item": "Handles edge cases", "weight": 1.5},
        {"item": "Uses idiomatic Python", "weight": 1.2},
        {"item": "Is correct", "weight": 2.0},
        {"item": "Has tests", "weight": 1.0},
    ]
    weighted = weighted_score(ctx.validation, weights)
    ratio = ratio_score(ctx.validation)
    return CodeReviewContext(
        solution=ctx.solution,
        review=ctx.review,
        validation=ctx.validation,
        weighted_score=weighted,
        ratio_score=ratio,
    )


# Define the repair pipeline
repair_pipeline = Pipeline.from_step(repair_code) >> Pipeline.from_step(validate_code)


# Define the main pipeline with proper Flujo DSL
def needs_repair(ctx: CodeReviewContext) -> bool:
    """Check if repair is needed."""
    return not all(item.passed for item in ctx.validation.items)


def review_passed(result, ctx=None) -> bool:
    """Check if review passed."""
    # Accepts (result, ctx) for Step.branch_on compatibility
    return all(item.passed for item in result.review.items)


@step(name="FailureHandler")
async def failure_handler(ctx: CodeReviewContext) -> CodeReviewContext:
    """Handle the case where review never passes."""
    print("❌ Review loop failed to pass. Feedback:")
    if ctx.review is not None:
        for item in ctx.review.items:
            status = "✅ Passed" if item.passed else "❌ Failed"
            print(f"  - {item.description:<30} {status} | Feedback: {item.feedback}")
    else:
        print("No review feedback available.")
    return CodeReviewContext(
        solution=ctx.solution,
        review=ctx.review,
        validation=None,
        weighted_score=0.0,
        ratio_score=0.0,
    )


@step(name="NoOp")
async def noop_step(ctx: CodeReviewContext) -> CodeReviewContext:
    return ctx


# Replace the main_pipeline definition with a robust branch after the review loop
main_pipeline = (
    generate_solution
    >> Step.loop_until(
        name="ReviewLoop",
        loop_body_pipeline=Pipeline.from_step(review_code),
        exit_condition_callable=lambda result, ctx: review_passed(result),
        max_loops=3,
    )
    >> Step.branch_on(
        name="ReviewPassBranch",
        condition_callable=review_passed,
        branches={
            True: Pipeline.from_step(validate_code),
            False: Pipeline.from_step(failure_handler),
        },
    )
    >> Step.branch_on(
        name="RepairBranch",
        condition_callable=lambda result, ctx: getattr(result, "validation", None) is not None
        and needs_repair(result),
        branches={True: repair_pipeline, False: Pipeline.from_step(noop_step)},
    )
    >> score_solution
)


async def main():
    # Create Flujo runner
    runner = Flujo(main_pipeline)

    # Run the pipeline
    task = Task(prompt="Write a Python function that checks if a string is a palindrome.")
    result = None

    print("🚀 Starting Flujo Pipeline...")
    async for item in runner.run_async(task.prompt):
        result = item

    # Print results
    if result and result.step_history:
        final_step = result.step_history[-1]
        final_output = final_step.output

        print("\n🎉 Flujo Pipeline Complete!")
        print("=" * 50)
        if final_output is not None:
            print(f"Final Solution:\n{getattr(final_output, 'solution', '[No solution]')}")
            print(f"\nWeighted Score: {getattr(final_output, 'weighted_score', 0.0):.2f}")
            print(f"Ratio Score: {getattr(final_output, 'ratio_score', 0.0):.2f}")
            if getattr(final_output, "validation", None) is not None:
                print("\nFinal Checklist:")
                for item in final_output.validation.items:
                    status = "✅ Passed" if item.passed else "❌ Failed"
                    print(f"  - {item.description:<30} {status}")
            else:
                print("\nNo validation results: the pipeline halted before validation.")
        else:
            print("\nNo final output: the pipeline halted early.")


if __name__ == "__main__":
    asyncio.run(main())
