"""
00_quickstart.py
----------------
Most basic usage: call the orchestrator once and print the result.
Run with:
    ORCH_OPENAI_API_KEY=sk-... python 00_quickstart.py
"""
from pydantic_ai_orchestrator import Orchestrator, Task
from pydantic_ai_orchestrator.infra.agents import review_agent, solution_agent, validator_agent, get_reflection_agent

# 1️⃣  Create a default orchestrator (uses GPT-4o for all agents)
orch = Orchestrator(
    review_agent,
    solution_agent,
    validator_agent,
    get_reflection_agent()
)

# 2️⃣  Wrap your prompt in a Task (metadata optional)
task = Task(prompt="Write a short motivational haiku about debugging.")

# 3️⃣  Synchronous, blocking call – returns a Candidate object
best = orch.run_sync(task)

# 4️⃣  Inspect the result
print("Score:", best.score)
print("\nSolution:\n", best.solution)
print("\nChecklist:")
for item in best.checklist.items:
    print(f" • {item.description:<40}  =>  {item.passed}") 