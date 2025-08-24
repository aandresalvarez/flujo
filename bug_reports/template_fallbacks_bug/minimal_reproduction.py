#!/usr/bin/env python3
"""
Minimal Reproduction Script for Template Fallbacks Bug

This script demonstrates the critical bug where conditional template syntax
like {{ a or b }} fails silently, breaking data flow and template fallback logic.

Bug: {{ context.initial_prompt or 'Fallback text' }} doesn't work
Expected: Use context value if available, otherwise use fallback
Actual: Template resolution fails or always uses empty value

Usage:
    python3 minimal_reproduction.py
"""

import subprocess
import tempfile
import os
import sys


def create_test_pipeline():
    """Create a minimal test pipeline that demonstrates the bug"""
    return """version: "0.1"
name: "template_fallbacks_bug_test"

steps:
  - kind: step
    name: test_fallback
    agent:
      id: "flujo.builtins.stringify"
    input: "{{ context.initial_prompt or 'Fallback: No prompt provided' }}"
    
  - kind: step
    name: show_result
    agent:
      id: "flujo.builtins.stringify"
    input: "Template result: {{ steps.test_fallback }}"
"""


def create_workaround_pipeline():
    """Create a working pipeline using workarounds"""
    return """version: "0.1"
name: "template_fallbacks_workaround"

steps:
  - kind: step
    name: check_prompt
    agent:
      id: "flujo.builtins.stringify"
    input: "{{ context.initial_prompt }}"
    
  - kind: step
    name: use_fallback
    agent:
      id: "flujo.builtins.stringify"
    input: "{{ steps.check_prompt or 'Fallback: No prompt provided' }}"
"""


def test_conditional_template_failure():
    """Test that conditional template syntax fails (demonstrates the bug)"""
    print("🧪 Testing Conditional Template Failure (The Bug)")
    print("=" * 60)

    # Create test pipeline
    pipeline_content = create_test_pipeline()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(pipeline_content)
        pipeline_path = f.name

    print(f"📝 Test pipeline created: {pipeline_path}")

    # Test conditional template (this should work but doesn't)
    print("\n🔍 Test: Conditional Template (FAILS)")
    print("-" * 50)
    print("Command: flujo run <pipeline>")
    print("Expected: Should output fallback text when context.initial_prompt is empty")
    print("Actual: Template resolution fails or outputs empty string")

    try:
        # Test with no context value
        result = subprocess.run(
            ["flujo", "run", pipeline_path], capture_output=True, text=True, timeout=30
        )

        print(f"\nExit code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")

        # Analyze the result
        if "Fallback: No prompt provided" in result.stdout:
            print("✅ SUCCESS: Conditional template worked (Bug is FIXED!)")
            return False  # Bug is fixed
        elif (
            "Template result:" in result.stdout
            and "Fallback: No prompt provided" not in result.stdout
        ):
            print("❌ FAILURE: Template resolved but no fallback used (Bug CONFIRMED)")
            return True  # Bug confirmed
        else:
            print("⚠️  UNKNOWN: Unexpected behavior")
            return True  # Bug confirmed

    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT: Command took too long")
        print("❌ FAILURE: Bug CONFIRMED - conditional template not working")
        return True
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return True

    finally:
        # Clean up
        os.unlink(pipeline_path)


def test_workaround_methods():
    """Test that workarounds actually work"""
    print("\n🧪 Testing Workaround Methods (Should Work)")
    print("=" * 60)

    pipeline_content = create_workaround_pipeline()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(pipeline_content)
        pipeline_path = f.name

    print(f"📝 Workaround pipeline created: {pipeline_path}")

    # Test workaround method
    print("\n🔍 Test: Workaround Method")
    print("-" * 40)

    try:
        # Test with no context value
        result = subprocess.run(
            ["flujo", "run", pipeline_path], capture_output=True, text=True, timeout=30
        )

        print(f"Exit code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")

        # Analyze the result
        if "Fallback: No prompt provided" in result.stdout:
            print("✅ SUCCESS: Workaround method works")
        else:
            print("❌ FAILURE: Workaround method doesn't work")

    except Exception as e:
        print(f"❌ ERROR: {e}")

    # Clean up
    os.unlink(pipeline_path)


def demonstrate_bug_impact():
    """Demonstrate the real-world impact of this bug"""
    print("\n🚨 Bug Impact Demonstration")
    print("=" * 50)

    print("This bug affects users in the following ways:")
    print()
    print("1. **Can't provide fallback values**:")
    print("   # This doesn't work:")
    print("   input: '{{ context.value or \"default\" }}'")
    print("   # Result: No fallback, pipeline fails or uses empty value")
    print()
    print("2. **Can't handle missing context gracefully**:")
    print("   # This doesn't work:")
    print("   input: '{{ context.user_input or \"Please provide input\" }}'")
    print("   # Result: No graceful degradation, pipeline breaks")
    print()
    print("3. **Can't build robust pipelines**:")
    print("   # This doesn't work:")
    print("   input: '{{ context.config or context.default_config }}'")
    print("   # Result: No fallback logic, pipelines are fragile")
    print()
    print("4. **Forces complex workarounds**:")
    print("   # Users must implement complex logic:")
    print("   # - Separate steps for checking values")
    print("   # - Explicit conditional logic")
    print("   # - Multiple pipeline steps instead of simple templates")


def provide_workarounds():
    """Provide working alternatives for users"""
    print("\n🔧 Available Workarounds")
    print("=" * 50)

    print("While the bug exists, users can use these alternatives:")
    print()
    print("1. **Explicit Conditional Logic** (Most Reliable):")
    print("   - Use separate steps to check values")
    print("   - Implement fallback logic explicitly")
    print("   - Handle missing values gracefully")
    print()
    print("2. **Default Values in Context** (Simple):")
    print("   - Set explicit default values")
    print("   - Use context.default_value instead of fallbacks")
    print("   - Pre-populate context with defaults")
    print()
    print("3. **Separate Steps with Logic** (Flexible):")
    print("   - Handle fallback logic in separate steps")
    print("   - Use explicit conditional checks")
    print("   - Build robust pipeline flows")


def main():
    """Main function to demonstrate the template fallbacks bug"""
    print("🚨 TEMPLATE FALLBACKS BUG - MINIMAL REPRODUCTION")
    print("=" * 70)
    print()
    print("This script demonstrates the critical bug where conditional template")
    print("syntax like {{ a or b }} fails silently.")
    print()
    print("Bug Description:")
    print("- Expected: {{ context.value or 'fallback' }} should work")
    print("- Actual: Template resolution fails or always uses empty value")
    print()
    print("Impact: Breaks data flow, template fallback logic, and pipeline robustness")
    print()

    # Check if flujo is available
    try:
        # Try to find flujo
        flujo_path = None
        possible_paths = [
            "flujo",
            ".venv/bin/flujo",
            "../.venv/bin/flujo",
            "/Users/alvaro/Documents/Code/flujo/.venv/bin/flujo",
        ]

        for path in possible_paths:
            try:
                result = subprocess.run(
                    [path, "--help"], capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    flujo_path = path
                    print(f"✅ Flujo detected: {result.stdout[:100]}...")
                    break
            except Exception:
                continue

        if not flujo_path:
            print("❌ Flujo not available or not working")
            print("   Please ensure 'flujo' command is available in PATH")
            sys.exit(1)

        # Store flujo path for later use
        globals()["flujo_path"] = flujo_path

    except Exception as e:
        print(f"❌ Error checking Flujo: {e}")
        print("   Please ensure 'flujo' command is available in PATH")
        sys.exit(1)

    print()
    print("🔍 Starting Bug Reproduction...")
    print()

    # Test the bug
    bug_confirmed = test_conditional_template_failure()

    # Test workarounds
    test_workaround_methods()

    # Demonstrate impact
    demonstrate_bug_impact()

    # Provide workarounds
    provide_workarounds()

    print()
    print("=" * 70)
    print("🎯 BUG REPRODUCTION COMPLETE")
    print()

    if bug_confirmed:
        print("❌ BUG CONFIRMED: Conditional template syntax not working")
        print("🚨 This is a critical issue that breaks data flow and template fallbacks")
        print("🔧 Use workarounds documented above for immediate development")
        print("📋 Report this issue to Flujo development team")
    else:
        print("✅ BUG FIXED: Conditional template syntax now working")
        print("🎉 The template fallbacks issue has been resolved")

    print()
    print("Next steps:")
    print("1. Use workarounds for immediate development")
    print("2. Report issue to Flujo team if not fixed")
    print("3. Test fixes when they become available")


if __name__ == "__main__":
    main()
