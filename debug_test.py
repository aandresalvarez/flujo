#!/usr/bin/env python3

import asyncio
from flujo.application.core.ultra_executor import ExecutorCore

class TestAgent:
    async def run(self, data, **kwargs):
        return "test success"

async def test_agent_execution():
    executor = ExecutorCore()
    agent = TestAgent()
    
    # Test direct agent execution
    result = await executor._agent_runner.run(
        agent,
        "test data",
        context=None,
        resources=None,
        options={},
        stream=False,
        on_chunk=None,
        breach_event=None,
    )
    
    print(f"Agent execution result: {result}")
    print(f"Result type: {type(result)}")
    return result

if __name__ == "__main__":
    result = asyncio.run(test_agent_execution())
    print(f"Final result: {result}") 