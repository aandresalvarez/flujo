#!/usr/bin/env python3
"""
Comprehensive Architect Test Runner

This script runs all architect tests in different categories and provides
detailed reporting on the overall system health and robustness.
"""

import subprocess
import sys
import time
import os
from pathlib import Path
from typing import Dict


class TestRunner:
    """Comprehensive test runner for architect tests."""

    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.results: Dict[str, Dict] = {}
        self.start_time = time.time()

    def run_test_category(self, category: str, test_file: str, description: str) -> Dict:
        """Run a specific test category and return results."""
        print(f"\n{'=' * 60}")
        print(f"Running {category}: {description}")
        print(f"{'=' * 60}")

        test_path = self.test_dir / test_file

        if not test_path.exists():
            return {
                "status": "SKIP",
                "message": f"Test file {test_file} not found",
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "execution_time": 0,
            }

        # Set environment variable for state machine
        env = os.environ.copy()
        env["FLUJO_ARCHITECT_STATE_MACHINE"] = "1"

        start_time = time.time()

        try:
            # Run pytest with verbose output
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                str(test_path),
                "-v",
                "--tb=short",
                "--color=yes",
            ]

            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            execution_time = time.time() - start_time

            # Parse pytest output
            output_lines = result.stdout.split("\n")
            test_summary = None

            for line in output_lines:
                if "passed" in line and "failed" in line:
                    test_summary = line
                    break

            if result.returncode == 0:
                status = "PASS"
                message = "All tests passed"
            else:
                status = "FAIL"
                message = f"Tests failed with return code {result.returncode}"

            # Extract test counts from summary
            tests_run = 0
            tests_passed = 0
            tests_failed = 0

            if test_summary:
                # Parse something like "5 passed, 2 failed in 3.45s"
                import re

                passed_match = re.search(r"(\d+) passed", test_summary)
                failed_match = re.search(r"(\d+) failed", test_summary)

                if passed_match:
                    tests_passed = int(passed_match.group(1))
                if failed_match:
                    tests_failed = int(failed_match.group(1))

                tests_run = tests_passed + tests_failed

            return {
                "status": status,
                "message": message,
                "tests_run": tests_run,
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
                "execution_time": execution_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
            }

        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return {
                "status": "TIMEOUT",
                "message": "Tests timed out after 5 minutes",
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "execution_time": execution_time,
                "stdout": "",
                "stderr": "Timeout expired",
                "return_code": -1,
            }
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "status": "ERROR",
                "message": f"Error running tests: {str(e)}",
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "execution_time": execution_time,
                "stdout": "",
                "stderr": str(e),
                "return_code": -1,
            }

    def run_all_tests(self) -> None:
        """Run all test categories."""
        test_categories = [
            {
                "name": "regression",
                "file": "test_architect_regression_fixes.py",
                "description": "Regression Tests - Prevent critical issues from recurring",
            },
            {
                "name": "edge_cases",
                "file": "test_architect_edge_cases.py",
                "description": "Edge Case Tests - Handle unusual inputs gracefully",
            },
            {
                "name": "performance",
                "file": "test_architect_performance_stress.py",
                "description": "Performance & Stress Tests - Ensure system performance under load",
            },
            {
                "name": "security",
                "file": "test_architect_security_validation.py",
                "description": "Security & Validation Tests - Protect against malicious inputs",
            },
            {
                "name": "happy_path",
                "file": "test_architect_happy_path.py",
                "description": "Happy Path Tests - Verify core functionality works",
            },
        ]

        for category in test_categories:
            self.results[category["name"]] = self.run_test_category(
                category["name"], category["file"], category["description"]
            )

    def generate_report(self) -> str:
        """Generate a comprehensive test report."""
        total_time = time.time() - self.start_time

        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE ARCHITECT TEST REPORT")
        report.append("=" * 80)
        report.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total execution time: {total_time:.2f} seconds")
        report.append("")

        # Summary table
        report.append("TEST CATEGORY SUMMARY")
        report.append("-" * 80)
        report.append(
            f"{'Category':<15} {'Status':<10} {'Tests':<10} {'Passed':<8} {'Failed':<8} {'Time':<8}"
        )
        report.append("-" * 80)

        total_tests = 0
        total_passed = 0
        total_failed = 0

        for category, result in self.results.items():
            status = result["status"]
            tests_run = result["tests_run"]
            tests_passed = result["tests_passed"]
            tests_failed = result["tests_failed"]
            execution_time = result["execution_time"]

            report.append(
                f"{category:<15} {status:<10} {tests_run:<10} {tests_passed:<8} {tests_failed:<8} {execution_time:<8.2f}s"
            )

            total_tests += tests_run
            total_passed += tests_passed
            total_failed += tests_failed

        report.append("-" * 80)
        report.append(
            f"{'TOTAL':<15} {'':<10} {total_tests:<10} {total_passed:<8} {total_failed:<8} {total_time:<8.2f}s"
        )
        report.append("")

        # Overall status
        if total_failed == 0:
            overall_status = "✅ ALL TESTS PASSED"
        else:
            overall_status = f"❌ {total_failed} TESTS FAILED"

        report.append(f"OVERALL STATUS: {overall_status}")
        report.append("")

        # Detailed results
        report.append("DETAILED RESULTS")
        report.append("=" * 80)

        for category, result in self.results.items():
            report.append(f"\n{category.upper()} TESTS")
            report.append("-" * 40)
            report.append(f"Status: {result['status']}")
            report.append(f"Message: {result['message']}")
            report.append(f"Tests Run: {result['tests_run']}")
            report.append(f"Tests Passed: {result['tests_passed']}")
            report.append(f"Tests Failed: {result['tests_failed']}")
            report.append(f"Execution Time: {result['execution_time']:.2f}s")

            if result["status"] == "FAIL" and result["stderr"]:
                report.append(f"Error Details: {result['stderr']}")

        # Recommendations
        report.append("\n" + "=" * 80)
        report.append("RECOMMENDATIONS")
        report.append("=" * 80)

        if total_failed == 0:
            report.append("🎉 Excellent! All tests are passing.")
            report.append("The architect system is robust and ready for production.")
        else:
            report.append("⚠️  Some tests are failing. Recommendations:")
            report.append("1. Review failing tests to understand the issues")
            report.append("2. Fix the root causes of failures")
            report.append("3. Re-run tests to verify fixes")
            report.append("4. Consider adding more specific tests for failing areas")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def save_report(self, report: str, filename: str = None) -> None:
        """Save the test report to a file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"architect_test_report_{timestamp}.txt"

        report_path = self.test_dir / filename

        with open(report_path, "w") as f:
            f.write(report)

        print(f"\nTest report saved to: {report_path}")

    def run(self) -> int:
        """Main execution method."""
        print("🚀 Starting Comprehensive Architect Test Suite")
        print("This will run all test categories to verify system robustness")

        try:
            self.run_all_tests()
            report = self.generate_report()

            # Print report to console
            print(report)

            # Save report to file
            self.save_report(report)

            # Return appropriate exit code
            total_failed = sum(result["tests_failed"] for result in self.results.values())
            return 0 if total_failed == 0 else 1

        except KeyboardInterrupt:
            print("\n\n⚠️  Test execution interrupted by user")
            return 130
        except Exception as e:
            print(f"\n\n❌ Error during test execution: {e}")
            return 1


def main():
    """Main entry point."""
    runner = TestRunner()
    exit_code = runner.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
