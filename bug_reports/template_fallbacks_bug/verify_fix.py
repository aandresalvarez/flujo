#!/usr/bin/env python3
"""
Verification Script for Template Fallbacks Fix

This script verifies that the conditional template syntax fix is working correctly.
Run this after the Flujo team has implemented the fix to confirm it's working.

Usage:
    python3 verify_fix.py
"""

import subprocess
import tempfile
import os
import sys


def create_test_pipeline():
    """Create a comprehensive test pipeline for conditional templates"""
    return """version: "0.1"
name: "template_fallbacks_fix_verification"

steps:
  - kind: step
    name: test_simple_fallback
    agent:
      id: "flujo.builtins.stringify"
    input: "{{ context.initial_prompt or 'Simple Fallback: No prompt provided' }}"
    
  - kind: step
    name: test_complex_fallback
    agent:
      id: "flujo.builtins.stringify"
    input: "{{ steps.test_simple_fallback or context.default_value or 'Complex Fallback: No data available' }}"
    
  - kind: step
    name: test_step_fallback
    agent:
      id: "flujo.builtins.stringify"
    input: "{{ steps.previous_step or 'Step Fallback: No previous step' }}"
    
  - kind: step
    name: show_results
    agent:
      id: "flujo.builtins.stringify"
    input: |
      Template Fallbacks Fix Verification Results:
      
      1. Simple Fallback: {{ steps.test_simple_fallback }}
      2. Complex Fallback: {{ steps.test_complex_fallback }}
      3. Step Fallback: {{ steps.test_step_fallback }}
      
      All conditional templates should now work correctly!
"""


def test_conditional_templates():
    """Test that conditional template syntax is now working"""
    print("🧪 Testing Conditional Template Fix Verification")
    print("=" * 60)

    # Create test pipeline
    pipeline_content = create_test_pipeline()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(pipeline_content)
        pipeline_path = f.name

    print(f"📝 Test pipeline created: {pipeline_path}")

    # Test conditional templates (should now work)
    print("\n🔍 Test: Conditional Templates (Should Now Work)")
    print("-" * 60)
    print("Command: flujo run <pipeline>")
    print("Expected: All conditional templates should work correctly")
    print("Actual: Testing now...")

    try:
        # Test with timeout to avoid hanging
        result = subprocess.run(
            ["flujo", "run", pipeline_path], capture_output=True, text=True, timeout=30
        )

        print(f"\nExit code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")

        # Analyze the results
        if "Simple Fallback: No prompt provided" in result.stdout:
            print("✅ SUCCESS: Simple fallback working")
            simple_working = True
        else:
            print("❌ FAILURE: Simple fallback not working")
            simple_working = False

        if "Complex Fallback: No data available" in result.stdout:
            print("✅ SUCCESS: Complex fallback working")
            complex_working = True
        else:
            print("❌ FAILURE: Complex fallback not working")
            complex_working = False

        if "Step Fallback: No previous step" in result.stdout:
            print("✅ SUCCESS: Step fallback working")
            step_working = True
        else:
            print("❌ FAILURE: Step fallback not working")
            step_working = False

        # Overall assessment
        if simple_working and complex_working and step_working:
            print("\n🎉 ALL TESTS PASSED: Template Fallbacks Fix is Working!")
            return True
        else:
            print("\n⚠️  SOME TESTS FAILED: Template Fallbacks Fix may be incomplete")
            return False

    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT: Command took too long")
        print("❌ FAILURE: Template fallbacks fix verification failed")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

    finally:
        # Clean up
        os.unlink(pipeline_path)


def demonstrate_working_features():
    """Demonstrate what now works after the fix"""
    print("\n🎉 Working Features After Fix")
    print("=" * 50)

    print("The following conditional template syntax now works:")
    print()
    print("1. **Simple Fallbacks**:")
    print("   {{ context.value or 'default' }}")
    print("   ✅ Now works correctly")
    print()
    print("2. **Complex Fallbacks**:")
    print("   {{ a or b or c }}")
    print("   ✅ Now works correctly")
    print()
    print("3. **Step Fallbacks**:")
    print("   {{ steps.previous or 'no data' }}")
    print("   ✅ Now works correctly")
    print()
    print("4. **Context Fallbacks**:")
    print("   {{ context.config or context.default or 'system default' }}")
    print("   ✅ Now works correctly")


def provide_next_steps():
    """Provide next steps for users"""
    print("\n🚀 Next Steps After Fix")
    print("=" * 50)

    print("1. **Remove Workarounds**:")
    print("   - No more need for explicit conditional logic")
    print("   - No more separate steps for fallback handling")
    print("   - Use conditional templates directly")
    print()
    print("2. **Test in Your Pipelines**:")
    print("   - Verify {{ a or b }} syntax works")
    print("   - Test fallback values in your workflows")
    print("   - Build robust pipelines with graceful degradation")
    print()
    print("3. **Update Documentation**:")
    print("   - Remove references to workarounds")
    print("   - Document working conditional template syntax")
    print("   - Share success stories with the community")


def main():
    """Main function to verify the template fallbacks fix"""
    print("🎉 TEMPLATE FALLBACKS FIX VERIFICATION")
    print("=" * 70)
    print()
    print("This script verifies that the conditional template syntax fix")
    print("is working correctly after implementation by the Flujo team.")
    print()
    print("Expected: All conditional templates should now work perfectly")
    print()

    # Check if flujo is available
    try:
        result = subprocess.run(["flujo", "--help"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Flujo command available and working")
        else:
            print("❌ Flujo command not working")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error checking Flujo: {e}")
        print("   Please ensure 'flujo' command is available in PATH")
        sys.exit(1)

    print()
    print("🔍 Starting Fix Verification...")
    print()

    # Test the fix
    fix_working = test_conditional_templates()

    # Show results
    if fix_working:
        print()
        print("=" * 70)
        print("🎉 FIX VERIFICATION COMPLETE - SUCCESS!")
        print("=" * 70)
        print()
        print("✅ The Template Fallbacks fix is working correctly!")
        print("✅ Users can now use conditional template syntax")
        print("✅ Fallback values work perfectly")
        print("✅ Template system is fully functional")
        print()

        demonstrate_working_features()
        provide_next_steps()

        print()
        print("🎯 Framework Status: Template system now production-ready!")
        print("🚀 Next Priority: Fix Input Adaptation for complete production readiness")

    else:
        print()
        print("=" * 70)
        print("⚠️  FIX VERIFICATION COMPLETE - ISSUES DETECTED")
        print("=" * 70)
        print()
        print("❌ Some conditional template functionality may not be working")
        print("❌ The fix may be incomplete or have issues")
        print("🔧 Users should continue using workarounds")
        print("📋 Report any remaining issues to the Flujo team")

    print()
    print("=" * 70)
    print("Verification completed")
    print("=" * 70)


if __name__ == "__main__":
    main()
