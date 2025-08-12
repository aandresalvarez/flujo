# manual_testing/test_bug_demonstration.py
"""
This script demonstrates the bug that FSD-11 fixes.
It simulates the old behavior where context was always injected.
"""

import asyncio
import inspect


# Simulate the old buggy behavior
class BuggyAgent:
    """A simple agent that doesn't accept context parameter - this would fail with old code"""

    def __init__(self, name="BuggyAgent"):
        self.name = name

    async def run(self, data: str) -> str:
        """This method does NOT accept a context parameter"""
        return f"Response to: {data}"


# Simulate the old context injection logic (the bug)
async def old_buggy_context_injection(agent, data, context):
    """This simulates the old buggy behavior"""
    print("🔴 OLD BUGGY BEHAVIOR:")
    print(f"   - Agent signature: {inspect.signature(agent.run)}")
    print(f"   - Context available: {context is not None}")
    print("   - Attempting to call with context anyway...")

    try:
        # This is what the old code would do - always pass context
        result = await agent.run(data, context=context)
        print("   ✅ SUCCESS (unexpected - this shouldn't work)")
        return result
    except TypeError as e:
        print(f"   ❌ FAILED with TypeError: {e}")
        print("   This is the bug that FSD-11 fixes!")
        return None


# Simulate the new fixed behavior
async def new_fixed_context_injection(agent, data, context):
    """This simulates the new fixed behavior"""
    print("🟢 NEW FIXED BEHAVIOR:")
    print(f"   - Agent signature: {inspect.signature(agent.run)}")
    print(f"   - Context available: {context is not None}")

    # Check if agent accepts context parameter
    sig = inspect.signature(agent.run)
    accepts_context = "context" in sig.parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )

    print(f"   - Agent accepts context: {accepts_context}")

    try:
        if accepts_context:
            print("   - Passing context to agent...")
            result = await agent.run(data, context=context)
        else:
            print("   - NOT passing context to agent (signature-aware!)")
            result = await agent.run(data)

        print("   ✅ SUCCESS")
        return result
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return None


async def main():
    print("=" * 60)
    print("FSD-11 BUG DEMONSTRATION")
    print("=" * 60)

    # Create a simple agent that doesn't accept context
    agent = BuggyAgent()
    data = "test input"
    context = {"some": "context"}

    print(f"\nTesting with agent: {agent.name}")
    print(f"Agent run method signature: {inspect.signature(agent.run)}")
    print()

    # Test old buggy behavior
    print("1. Testing OLD BUGGY behavior:")
    result1 = await old_buggy_context_injection(agent, data, context)
    print()

    # Test new fixed behavior
    print("2. Testing NEW FIXED behavior:")
    result2 = await new_fixed_context_injection(agent, data, context)
    print()

    print("=" * 60)
    print("SUMMARY:")
    print(f"Old behavior: {'❌ FAILED' if result1 is None else '✅ WORKED (unexpected)'}")
    print(f"New behavior: {'✅ WORKED' if result2 is not None else '❌ FAILED'}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
