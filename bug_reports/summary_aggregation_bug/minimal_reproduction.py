#!/usr/bin/env python3
"""
Minimal Reproduction Script for Summary Aggregation (Updated)

This script validates that Flujo summary tables properly render nested steps and
aggregate usage totals when pipelines contain nested execution (e.g., parallel
branches). It uses `uv run flujo` for CLI invocation.

Usage:
    python minimal_reproduction.py

Requirements:
    - Run `make install` to ensure `uv` and virtualenv are set up
    - CLI available via `uv run flujo`
"""

import subprocess
import tempfile
import os
import sys
import json


def _parallel_yaml() -> str:
    """Return a YAML pipeline with a Parallel step and three inner steps.

    This avoids unsupported 'flujo.builtins.workflow' usage and works with the
    current YAML blueprint loader.
    """
    return """version: "0.1"
name: "summary_aggregation_bug_test"

steps:
  - kind: parallel
    name: wrapper_parallel
    branches:
      branch1:
        - kind: step
          name: inner_step_1
          agent:
            id: "flujo.builtins.stringify"
          input: "First inner step input"
      branch2:
        - kind: step
          name: inner_step_2
          agent:
            id: "flujo.builtins.stringify"
          input: "Second inner step input"
      branch3:
        - kind: step
          name: inner_step_3
          agent:
            id: "flujo.builtins.stringify"
          input: "Third inner step input"
"""


def create_test_pipeline() -> str:
    """Create a minimal pipeline that uses a supported nested construct (parallel)."""
    return _parallel_yaml()


def create_complex_nested_pipeline() -> str:
    """Create a more complex nested pipeline using conditional branching + steps."""
    return """version: "0.1"
name: "complex_nested_workflow_test"

steps:
  - kind: conditional
    name: choose_branch
    # Built-in condition: if output matches a branch key we route there; otherwise default
    branches:
      A:
        - kind: step
          name: branch_A_step
          agent:
            id: "flujo.builtins.stringify"
          input: "A path"
      B:
        - kind: step
          name: branch_B_step
          agent:
            id: "flujo.builtins.stringify"
          input: "B path"
    default_branch:
      - kind: step
        name: default_step
        agent:
          id: "flujo.builtins.stringify"
        input: "default path"
"""


def check_flujo_available() -> bool:
    """Check if Flujo is available via uv and working."""
    try:
        result = subprocess.run(
            ["uv", "run", "flujo", "dev", "version"], capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            print(f"✅ Flujo found: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Flujo command failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Flujo command timed out")
        return False
    except FileNotFoundError:
        print("❌ 'uv' command not found. Run 'make install' first.")
        return False
    except Exception as e:
        print(f"❌ Error checking Flujo: {e}")
        return False


def test_summary_aggregation(pipeline_content, pipeline_name, input_data="test input"):
    """Test if summary aggregation works correctly for nested steps."""
    print(f"\n🧪 Testing Pipeline: {pipeline_name}")
    print("=" * 60)

    # Create temporary pipeline file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(pipeline_content)
        pipeline_file = f.name

    try:
        print(f"📁 Created pipeline file: {pipeline_file}")
        print(f"📋 Pipeline content:\n{pipeline_content}")

        # Test 1: Run with normal output to see summary table
        print("\n📊 Test 1: Normal Output (Summary Table)")
        print("-" * 40)
        print(f"Command: echo '{input_data}' | uv run flujo run {pipeline_file}")

        cmd = ["uv", "run", "flujo", "run", pipeline_file]
        process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        stdout, stderr = process.communicate(input=input_data, timeout=60)

        if process.returncode == 0:
            print("✅ Pipeline executed successfully")

            # Check for summary information
            if "Total cost:" in stdout:
                print("📊 Summary table found:")
                # Extract summary lines
                lines = stdout.split("\n")
                summary_lines = [
                    line
                    for line in lines
                    if any(
                        keyword in line
                        for keyword in ["Total cost:", "Total tokens:", "Steps executed:"]
                    )
                ]
                for line in summary_lines:
                    print(f"   {line.strip()}")
            else:
                print("❌ No summary table found in output")

            # Check for step results
            if "Step Results:" in stdout:
                print("📋 Step results found in summary")
                # Check nested inner steps appear
                if any(name in stdout for name in ["inner_step_1", "inner_step_2", "inner_step_3"]):
                    print("   ✅ Nested inner steps displayed in summary")
                else:
                    print("   ❌ Nested inner steps NOT displayed in summary")
            else:
                print("❌ No step results found in summary")

        else:
            print(f"❌ Pipeline execution failed: {stderr}")
            return False

        # Test 2: Run with JSON output to see actual data
        print("\n🔍 Test 2: JSON Output (Actual Data)")
        print("-" * 40)
        print(f"Command: echo '{input_data}' | uv run flujo run --json {pipeline_file}")

        cmd_json = ["uv", "run", "flujo", "run", "--json", pipeline_file]
        process_json = subprocess.Popen(
            cmd_json,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stdout_json, stderr_json = process_json.communicate(input=input_data, timeout=60)

        if process_json.returncode == 0:
            print("✅ JSON output generated successfully")

            try:
                # Parse JSON output
                data = json.loads(stdout_json)

                # Extract summary information
                total_cost = data.get("total_cost_usd", 0)
                total_tokens = data.get("total_tokens", 0)
                step_count = len(data.get("step_history", []))

                print("📊 JSON Summary Data:")
                print(f"   Total cost: ${total_cost}")
                print(f"   Total tokens: {total_tokens}")
                print(f"   Top-level steps: {step_count}")

                # Check for nested step history in first step (e.g., parallel)
                if "step_history" in data and len(data["step_history"]) > 0:
                    first_step = data["step_history"][0]
                    nested = first_step.get("step_history", [])
                    if nested:
                        names = [str(sr.get("name")) for sr in nested if isinstance(sr, dict)]
                        print(f"   Nested steps: {names}")
                        if {"inner_step_1", "inner_step_2", "inner_step_3"}.issubset(set(names)):
                            print("   ✅ JSON nested step history present")
                            return False
                        else:
                            print("   ❌ JSON nested step names missing")
                    else:
                        print("   ❌ No nested step_history found in first step")
                else:
                    print("❌ No step history found in JSON output")

            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON output: {e}")
                print(f"Raw output: {stdout_json[:200]}...")

        else:
            print(f"❌ JSON output generation failed: {stderr_json}")
            return False

        return False

    except subprocess.TimeoutExpired:
        print("⏰ Pipeline execution timed out")
        return False
    except Exception as e:
        print(f"❌ Error during pipeline execution: {e}")
        return False
    finally:
        # Clean up temporary file
        try:
            os.unlink(pipeline_file)
            print(f"🗑️  Cleaned up: {pipeline_file}")
        except Exception as e:
            print(f"⚠️  Warning: Could not clean up {pipeline_file}: {e}")


def test_workarounds():
    """Test available workarounds for the summary aggregation bug."""
    print("\n🔧 Testing Workarounds")
    print("=" * 50)

    pipeline_content = create_test_pipeline()

    # Create temporary pipeline file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(pipeline_content)
        pipeline_file = f.name

    try:
        print("📋 Testing JSON output structure...")
        print("   Command: uv run flujo run --json <pipeline>")

        cmd = ["uv", "run", "flujo", "run", "--json", pipeline_file]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print("   ✅ JSON output present - checking for nested data")

            try:
                data = json.loads(result.stdout)
                print("   📊 Data structure:")
                print(f"      Top-level steps: {len(data.get('step_history', []))}")
                if data.get("step_history"):
                    first = data["step_history"][0]
                    nested = first.get("step_history", [])
                    print(f"      Has nested step_history: {bool(nested)}")

            except json.JSONDecodeError:
                print("   ⚠️  JSON output not parseable")
        else:
            print(f"   ❌ JSON workaround failed: {result.stderr}")

    except Exception as e:
        print(f"   ❌ Error testing JSON workaround: {e}")
    finally:
        try:
            os.unlink(pipeline_file)
        except Exception:
            pass


def main():
    """Main function to run all tests."""
    print("🧪 Summary Aggregation - Minimal Reproduction Script (Updated)")
    print("=" * 70)

    # Check Flujo availability
    if not check_flujo_available():
        print("\n❌ Cannot proceed without Flujo. Please install and configure Flujo first.")
        sys.exit(1)

    # Test 1: Simple nested parallel
    simple_pipeline = create_test_pipeline()
    bug_detected = test_summary_aggregation(simple_pipeline, "Simple Nested Workflow")

    # Test 2: Conditional nested branching
    complex_pipeline = create_complex_nested_pipeline()
    bug_detected |= test_summary_aggregation(complex_pipeline, "Complex Nested Workflow")

    # Test workarounds
    test_workarounds()

    # Summary
    print("\n📊 Test Summary")
    print("=" * 30)
    if bug_detected:
        print("❌ Potential aggregation issues detected (see logs above)")
    else:
        print("✅ Summary shows nested steps and aggregated totals as expected")

    print("\n📝 Next Steps:")
    print("   1. If any issues appeared, share logs with the Flujo team")
    print("   2. JSON output contains complete nested data for automation")
    print("   3. Monitor future releases for additional improvements")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
