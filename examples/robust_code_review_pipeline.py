import asyncio
from flujo import make_agent_async, Task, init_telemetry
from flujo.domain.models import PipelineContext
from flujo.domain.models import Checklist, Candidate
from flujo.domain.scoring import weighted_score, ratio_score

init_telemetry()


async def main():
    # 1. Solution agent
    solution_agent = make_agent_async(
        "openai:gpt-4o",
        "You are a Python developer. Write a function as requested.",
        str,
    )

    # 2. Review agent (checklist)
    review_agent = make_agent_async(
        "openai:gpt-4o",
        (
            "You are a code reviewer. For the given function, return a Checklist with these items: "
            "'Has docstring', 'Handles edge cases', 'Uses idiomatic Python', 'Is correct', 'Has tests'. "
            "For each, set 'passed' to true or false."
        ),
        Checklist,
    )

    # 3. Validator agent (stricter)
    validator_agent = make_agent_async(
        "openai:gpt-4o",
        (
            "You are a strict validator. For the given function, return a Checklist with the same items as the reviewer. "
            "Be strict: only set 'passed' to true if the requirement is fully met."
        ),
        Checklist,
    )

    # 4. Repair agent (auto-repair)
    repair_agent = make_agent_async(
        "openai:gpt-4o",
        (
            "You are a code repair agent. Given a function and a checklist of failed items, rewrite the function to fix the issues. "
            "Return only the improved code as a string."
        ),
        str,
    )

    # 5. Define a custom pipeline with conditional repair
    async def pipeline(task: Task, context: PipelineContext):
        # Step 1: Generate solution
        solution = await solution_agent.run(task.prompt)
        print("\n--- Initial Solution ---\n", solution)

        # Step 2: Review
        checklist = await review_agent.run(solution)
        print("\n--- Review Checklist ---\n", checklist)

        # Step 3: Validation
        validation = await validator_agent.run(solution)
        print("\n--- Validation Checklist ---\n", validation)

        # Step 4: If validation fails, auto-repair
        if not all(item.passed for item in validation.items):
            print("\nValidation failed. Attempting auto-repair...")
            failed_items = [item.description for item in validation.items if not item.passed]
            repair_prompt = f"Function:\n{solution}\n\nFailed items: {failed_items}"
            repaired = await repair_agent.run(repair_prompt)
            print("\n--- Repaired Solution ---\n", repaired)
            # Re-review and re-validate
            checklist = await review_agent.run(repaired)
            validation = await validator_agent.run(repaired)
            final_solution = repaired
        else:
            final_solution = solution

        # Step 5: Scoring
        weights = [
            {"item": "Has docstring", "weight": 2.0},
            {"item": "Handles edge cases", "weight": 1.5},
            {"item": "Uses idiomatic Python", "weight": 1.2},
            {"item": "Is correct", "weight": 2.0},
            {"item": "Has tests", "weight": 1.0},
        ]
        weighted = weighted_score(validation, weights)
        ratio = ratio_score(validation)
        print(f"\n--- Final Scoring ---\nWeighted Score: {weighted:.2f}\nRatio Score: {ratio:.2f}")

        # Step 6: Print final checklist
        print("\n--- Final Checklist ---")
        for item in validation.items:
            status = "✅ Passed" if item.passed else "❌ Failed"
            print(f"  - {item.description:<30} {status}")

        # Return a Candidate for compatibility
        return Candidate(solution=final_solution, checklist=validation, score=weighted)

    # Run the pipeline
    task = Task(prompt="Write a Python function that checks if a string is a palindrome.")
    context = PipelineContext(run_id="demo_run", initial_prompt=task.prompt)
    result = await pipeline(task, context)
    print("\n--- Final Solution ---\n", result.solution)


if __name__ == "__main__":
    asyncio.run(main())
