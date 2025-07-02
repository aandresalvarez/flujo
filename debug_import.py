#!/usr/bin/env python3

print("=== Debug Import Issue ===")

print("1. Testing basic flujo import...")
try:
    import flujo
    print("✅ flujo imported successfully")
except Exception as e:
    print(f"❌ flujo import failed: {e}")
    exit(1)

print("2. Testing flujo.models import...")
try:
    import flujo.models
    print("✅ flujo.models imported successfully")
except Exception as e:
    print(f"❌ flujo.models import failed: {e}")
    exit(1)

print("3. Testing from flujo.models import...")
try:
    from flujo.models import PipelineContext
    print("✅ PipelineContext imported successfully")
except Exception as e:
    print(f"❌ PipelineContext import failed: {e}")
    exit(1)

print("4. Testing the exact import sequence from the example...")
try:
    from typing import cast
    from flujo.models import PipelineContext
    from flujo import make_agent_async, init_telemetry
    from flujo.recipes import AgenticLoop
    from flujo.domain.commands import AgentCommand, FinishCommand, RunAgentCommand
    print("✅ All imports from example work!")
except Exception as e:
    print(f"❌ Example imports failed: {e}")
    import traceback
    traceback.print_exc()

print("=== Debug Complete ===") 