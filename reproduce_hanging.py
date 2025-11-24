from flujo import Step, Pipeline, Flujo
import sys


class EchoAgent:
    async def run(self, input_str: str) -> str:
        return f"Echo: {input_str}"


agent = EchoAgent()
step = Step.solution(agent, name="test")
pipeline = Pipeline.from_step(step)

# Create runner and execute
print("Starting runner...")
runner = Flujo(pipeline, state_backend=None, enable_tracing=False)
result = runner.run("Hello")
print("Execution completed")
sys.exit(0)
