from flujo import Step, Pipeline, Flujo
import sys

# Create agent and step
# Using a simple string agent for reproduction if openai is not configured or to avoid api calls if possible
# But the user used "openai:gpt-4o". I'll try to use a mock or simple function if possible,
# but let's stick to the user's example as much as possible.
# If I don't have openai key, I might need to mock it.
# Let's check if I can use a lambda or simple function first to rule out OpenAI network issues.
# The user said "OpenAI API: Working".
# I'll try to use a dummy agent first to see if it hangs without OpenAI.


async def simple_agent(input_str: str) -> str:
    return f"Echo: {input_str}"


step = Step.solution(simple_agent, name="test")
pipeline = Pipeline.from_step(step)

# Create runner and execute
print("Starting runner...")
runner = Flujo(pipeline, state_backend=None, enable_tracing=False)
result = runner.run("Hello")
print("Execution completed")
sys.exit(0)
