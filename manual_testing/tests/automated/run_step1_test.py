# manual_testing/run_step1_test.py
"""
Simple runner script for Step 1 comprehensive test.
This script sets up the environment and runs the test suite.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_environment():
    """Set up the testing environment."""
    print("🔧 Setting up test environment...")

    # Check if we're in the right directory
    current_dir = Path.cwd()
    if current_dir.name != "manual_testing":
        print(f"⚠️  Current directory: {current_dir}")
        print("   Consider running from manual_testing/ directory for local config")

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        masked_key = (
            f"{'*' * (len(api_key) - 4)}{api_key[-4:]}" if len(api_key) >= 4 else "*" * len(api_key)
        )
        print(f"✅ API key found: {masked_key}")
    else:
        print("⚠️  No OPENAI_API_KEY found - some tests will be skipped")

    # Check for flujo.toml
    config_file = Path("flujo.toml")
    if config_file.exists():
        print("✅ Local flujo.toml configuration found")
    else:
        print("⚠️  No local flujo.toml found - using global config")

    print()


def main():
    """Run the Step 1 comprehensive test."""
    print("=" * 80)
    print("STEP 1 TEST RUNNER")
    print("Running comprehensive test for Core Agentic Step")
    print("=" * 80)

    # Set up environment
    setup_environment()

    try:
        # Import and run the test
        from test_step1_core_agentic import run_comprehensive_test

        asyncio.run(run_comprehensive_test())

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the manual_testing directory")
        return 1
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n🎉 Test runner completed successfully!")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
