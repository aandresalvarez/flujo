# manual_testing/comprehensive_test.py
"""
Comprehensive test to validate FSD-11 fix.
Tests both stateless and context-aware agents.
"""

import asyncio
from flujo import Flujo, Step, Pipeline
from flujo.domain.models import PipelineContext

# Test Agent 1: Stateless agent (doesn't accept context)
class StatelessAgent:
    """A simple agent that doesn't accept context parameter"""
    def __init__(self, name="StatelessAgent"):
        self.name = name

    async def run(self, data: str) -> str:
        """This method does NOT accept a context parameter"""
        return f"Stateless response to: {data}"

# Test Agent 2: Context-aware agent (accepts context)
class ContextAwareAgent:
    """A context-aware agent that accepts context parameter"""
    def __init__(self, name="ContextAwareAgent"):
        self.name = name

    async def run(self, data: str, context=None) -> str:
        """This method DOES accept a context parameter"""
        context_info = f" (context: {type(context).__name__})" if context else ""
        return f"Context-aware response to: {data}{context_info}"

# Test Agent 3: Flexible agent (accepts **kwargs)
class FlexibleAgent:
    """A flexible agent that accepts **kwargs"""
    def __init__(self, name="FlexibleAgent"):
        self.name = name

    async def run(self, data: str, **kwargs) -> str:
        """This method accepts **kwargs, so it can receive context"""
        context_info = f" (kwargs: {list(kwargs.keys())})" if kwargs else ""
        return f"Flexible response to: {data}{context_info}"

async def test_agent(agent, agent_name, context_model=None):
    """Test a single agent with the Flujo runner"""
    print(f"\n{'='*50}")
    print(f"Testing: {agent_name}")
    print(f"Agent signature: {agent.run.__name__}({agent.run.__code__.co_varnames})")
    print(f"Context model: {context_model.__name__ if context_model else 'None'}")
    print(f"{'='*50}")

    # Create step and pipeline
    step = Step(name=f"Test{agent_name}", agent=agent)
    pipeline = Pipeline.from_step(step)

    # Create runner
    runner = Flujo(
        pipeline,
        context_model=context_model,
        pipeline_name=f"test_{agent_name.lower()}",
        initial_context_data={"initial_prompt": "Test prompt"} if context_model else None
    )

    # Test input
    test_input = "hello world"

    try:
        # Run the pipeline using async version
        async for result in runner.run_async(test_input):
            if result and result.step_history:
                final_step = result.step_history[-1]
                if final_step.success:
                    print(f"✅ SUCCESS: {final_step.output}")
                    return True
                else:
                    print(f"❌ FAILED: {final_step.feedback}")
                    return False
            else:
                print(f"❌ FAILED: No step history")
                return False

    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False

async def main():
    print("FSD-11 COMPREHENSIVE TEST")
    print("Testing signature-aware context injection")
    print("="*60)

    # Test cases
    test_cases = [
        # Test 1: Stateless agent without context
        (StatelessAgent(), "StatelessAgent", None),

        # Test 2: Stateless agent with context (this should work now!)
        (StatelessAgent(), "StatelessAgentWithContext", PipelineContext),

        # Test 3: Context-aware agent with context
        (ContextAwareAgent(), "ContextAwareAgent", PipelineContext),

        # Test 4: Flexible agent with context
        (FlexibleAgent(), "FlexibleAgent", PipelineContext),
    ]

    results = []

    for agent, name, context_model in test_cases:
        success = await test_agent(agent, name, context_model)
        results.append((name, success))

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")

    all_passed = True
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name}: {status}")
        if not success:
            all_passed = False

    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")

    if all_passed:
        print("\n🎉 FSD-11 fix is working correctly!")
        print("   - Stateless agents work without context")
        print("   - Stateless agents work with context (signature-aware injection)")
        print("   - Context-aware agents work with context")
        print("   - Flexible agents work with context")
    else:
        print("\n⚠️  Some tests failed - FSD-11 fix may not be complete")

if __name__ == "__main__":
    asyncio.run(main())
