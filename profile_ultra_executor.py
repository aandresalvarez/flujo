import asyncio
import cProfile
import pstats
from unittest.mock import Mock
from flujo.application.core.ultra_executor import UltraStepExecutor
from flujo.domain.models import StepResult
from flujo.domain.dsl.step import Step

# Minimal async agent
class DummyAgent:
    async def run(self, data):
        return "ok"

# Minimal step
def make_step():
    step = Mock(spec=Step)
    step.name = "profile_step"
    step.config = Mock()
    step.config.max_retries = 1
    step.config.temperature = None
    step.agent = DummyAgent()
    step.validators = []
    step.plugins = []
    step.fallback_step = None
    return step

# Minimal executor
class SimpleExecutor:
    async def execute_step(self, step, data, context, resources):
        agent = getattr(step, "agent", None)
        raw = await agent.run(data)
        return StepResult(
            name=step.name,
            output=raw,
            success=True,
            attempts=1,
            latency_s=0.0,
        )

async def run_executor(executor, step, n):
    for _ in range(n):
        result = await executor.execute_step(step, data=None, context=None, resources=None)
        assert result.output == "ok"

if __name__ == "__main__":
    N = 10000
    step = make_step()
    ultra = UltraStepExecutor(enable_cache=False)
    simple = SimpleExecutor()

    print("Profiling UltraStepExecutor...")
    prof1 = cProfile.Profile()
    prof1.enable()
    asyncio.run(run_executor(ultra, step, N))
    prof1.disable()
    stats1 = pstats.Stats(prof1).sort_stats("cumtime")
    stats1.print_stats(20)

    print("\nProfiling SimpleExecutor...")
    prof2 = cProfile.Profile()
    prof2.enable()
    asyncio.run(run_executor(simple, step, N))
    prof2.disable()
    stats2 = pstats.Stats(prof2).sort_stats("cumtime")
    stats2.print_stats(20)
