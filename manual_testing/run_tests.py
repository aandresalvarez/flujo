# manual_testing/run_tests.py
"""
Main test runner for manual testing.

This script provides easy access to run different types of tests
from the organized folder structure.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def show_menu():
    """Show the available test options."""
    print("=" * 80)
    print("MANUAL TESTING - TEST RUNNER")
    print("=" * 80)
    print("\nAvailable test options:")
    print("\n📋 AUTOMATED TESTS:")
    print("  1. Comprehensive Step 1 test suite (11 tests)")
    print("  2. FSD-11 bug demonstration")
    print("  3. Configuration test")
    print("  4. Comprehensive agent test")
    print("\n🧪 MANUAL TESTS (Real API):")
    print("  5. Basic manual test (2 predefined examples)")
    print("  6. Challenging manual test (4 challenging examples)")
    print("  7. Interactive manual test (input your own)")
    print("\n📚 DOCUMENTATION:")
    print("  8. View manual testing summary")
    print("  9. View Step 1 test summary")
    print("\n🔧 EXAMPLES:")
    print("  10. Run basic pipeline example")
    print("\n  0. Exit")
    print("\n" + "=" * 80)

def run_automated_test(test_name):
    """Run an automated test."""
    print(f"\n🔧 Running automated test: {test_name}")

    if test_name == "comprehensive":
        import sys
        sys.path.append("tests/automated")
        from run_step1_test import main
        return main()
    elif test_name == "bug_demo":
        import sys
        sys.path.append("tests/automated")
        from test_bug_demonstration import main
        return main()
    elif test_name == "config":
        import sys
        sys.path.append("tests/automated")
        from test_config import test_configuration
        import asyncio
        asyncio.run(test_configuration())
        return 0
    elif test_name == "comprehensive_agent":
        import sys
        sys.path.append("tests/automated")
        from comprehensive_test import main
        return main()

def run_manual_test(test_name):
    """Run a manual test."""
    print(f"\n🧪 Running manual test: {test_name}")

    if test_name == "basic":
        import sys
        sys.path.append("tests/manual")
        from manual_test_step1 import main
        import asyncio
        asyncio.run(main())
        return 0
    elif test_name == "challenging":
        import sys
        sys.path.append("tests/manual")
        from manual_test_step1_challenging import main
        import asyncio
        asyncio.run(main())
        return 0
    elif test_name == "interactive":
        import sys
        sys.path.append("tests/manual")
        from interactive_test_step1 import main
        import asyncio
        asyncio.run(main())
        return 0

def show_documentation(doc_name):
    """Show documentation."""
    print(f"\n📚 Showing documentation: {doc_name}")

    if doc_name == "manual_summary":
        doc_path = Path("docs/MANUAL_TESTING_SUMMARY.md")
        if doc_path.exists():
            with open(doc_path, 'r') as f:
                print(f.read())
        else:
            print("❌ Documentation file not found")
        return 0
    elif doc_name == "step1_summary":
        doc_path = Path("docs/TEST_STEP1_SUMMARY.md")
        if doc_path.exists():
            with open(doc_path, 'r') as f:
                print(f.read())
        else:
            print("❌ Documentation file not found")
        return 0

def run_example(example_name):
    """Run an example."""
    print(f"\n🔧 Running example: {example_name}")

    if example_name == "basic_pipeline":
        import sys
        sys.path.append("examples")
        from main import main
        import asyncio
        asyncio.run(main())
        return 0

def main():
    """Main test runner."""
    while True:
        show_menu()

        try:
            choice = input("\nEnter your choice (0-10): ").strip()

            if choice == "0":
                print("\n👋 Goodbye!")
                break
            elif choice == "1":
                run_automated_test("comprehensive")
            elif choice == "2":
                run_automated_test("bug_demo")
            elif choice == "3":
                run_automated_test("config")
            elif choice == "4":
                run_automated_test("comprehensive_agent")
            elif choice == "5":
                run_manual_test("basic")
            elif choice == "6":
                run_manual_test("challenging")
            elif choice == "7":
                run_manual_test("interactive")
            elif choice == "8":
                show_documentation("manual_summary")
            elif choice == "9":
                show_documentation("step1_summary")
            elif choice == "10":
                run_example("basic_pipeline")
            else:
                print("❌ Invalid choice. Please enter a number between 0-10.")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
