#!/usr/bin/env python3
"""
Robustness Test Runner

Comprehensive test runner for Flujo robustness tests that provides detailed
reporting and analysis of system reliability across multiple dimensions.
"""

import subprocess
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import argparse


@dataclass
class TestResult:
    """Result of a test category run."""

    category: str
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    duration: float = 0.0
    output: str = ""
    critical_failures: List[str] = field(default_factory=list)


@dataclass
class RobustnessReport:
    """Comprehensive robustness test report."""

    timestamp: str
    total_duration: float
    overall_passed: bool
    categories: Dict[str, TestResult] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    critical_issues: List[str] = field(default_factory=list)


class RobustnessTestRunner:
    """Runner for comprehensive robustness testing."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_categories = {
            "performance": "tests/robustness/test_performance_regression.py",
            "memory": "tests/robustness/test_memory_leak_detection.py",
            "concurrency": "tests/robustness/test_concurrency_safety.py",
            "error_recovery": "tests/robustness/test_error_recovery.py",
            "configuration": "tests/robustness/test_configuration_robustness.py",
        }

    def run_category_tests(self, category: str, verbose: bool = False) -> TestResult:
        """Run tests for a specific category."""
        if category not in self.test_categories:
            raise ValueError(f"Unknown category: {category}")

        test_file = self.test_categories[category]
        test_path = self.project_root / test_file

        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_path}")

        # Build pytest command
        cmd = [sys.executable, "-m", "pytest", str(test_path)]

        if not verbose:
            cmd.extend(["--tb=short"])  # Don't use -q so we can parse output
        else:
            cmd.extend(["--tb=long", "-v"])

        # Add timing
        cmd.extend(["--durations=10"])

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout per category
            )
        except subprocess.TimeoutExpired:
            return TestResult(
                category=category,
                errors=1,
                duration=time.time() - start_time,
                critical_failures=["Test timed out after 10 minutes"],
            )

        duration = time.time() - start_time

        # Parse results from pytest output
        passed = failed = errors = skipped = 0
        critical_failures = []

        stdout = result.stdout

        # Debug: print stdout if verbose
        if verbose:
            print(f"DEBUG: pytest stdout for {category}:")
            print(repr(stdout))

        # Parse pytest output - look for summary line like "====== 1 passed, 2 failed, 3 errors in 0.12s ======"
        import re

        # Pattern for summary line - try multiple formats
        summary_patterns = [
            r"=+ (\d+) passed(?:, (\d+) failed)?(?:, (\d+) errors?)?(?:, (\d+) skipped)? in",  # Full summary
            r"(\d+) passed(?:, (\d+) failed)?(?:, (\d+) errors?)?(?:, (\d+) skipped)? in",  # Simple summary
        ]

        summary_match = None
        for pattern in summary_patterns:
            summary_match = re.search(pattern, stdout)
            if summary_match:
                break

        if summary_match:
            passed = int(summary_match.group(1))
            if summary_match.group(2):
                failed = int(summary_match.group(2))
            if summary_match.group(3):
                errors = int(summary_match.group(3))
            if summary_match.group(4):
                skipped = int(summary_match.group(4))
        else:
            # Fallback: count dots in the output (each dot represents a test)
            # Look for lines with dots and other test indicators
            lines = stdout.split("\n")
            for line in lines:
                line = line.strip()
                if "." in line and (
                    "PASSED" in stdout
                    or "FAILED" in stdout
                    or "ERROR" in stdout
                    or "SKIPPED" in stdout
                ):
                    # Count dots as passed tests
                    passed = line.count(".")
                    break

        # Check for critical test failures in the output
        lines = stdout.split("\n")
        for line in lines:
            if "FAILED" in line:
                # Extract test name if possible
                test_name = line.strip().split()[-1] if line.strip() else "unknown"
                # Check if this is a critical robustness failure
                if any(
                    keyword in test_name.lower()
                    for keyword in [
                        "leak",
                        "memory",
                        "concurrency",
                        "timeout",
                        "crash",
                        "performance",
                    ]
                ):
                    critical_failures.append(test_name)

        # If we still have no results, try to count individual test lines
        if passed == 0 and failed == 0 and errors == 0:
            for line in lines:
                line = line.strip()
                if line.startswith(("PASSED", "FAILED", "ERROR", "SKIPPED")):
                    if "PASSED" in line:
                        passed += 1
                    elif "FAILED" in line:
                        failed += 1
                        # Try to extract test name
                        parts = line.split()
                        if len(parts) > 1:
                            critical_failures.append(parts[-1])
                    elif "ERROR" in line:
                        errors += 1
                    elif "SKIPPED" in line:
                        skipped += 1

        return TestResult(
            category=category,
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            duration=duration,
            output=result.stdout + result.stderr,
            critical_failures=critical_failures,
        )

    def run_all_tests(
        self, categories: Optional[List[str]] = None, verbose: bool = False
    ) -> RobustnessReport:
        """Run all robustness tests and generate comprehensive report."""
        if categories is None:
            categories = list(self.test_categories.keys())

        report = RobustnessReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), total_duration=0.0, overall_passed=True
        )

        print("🚀 Starting Flujo Robustness Test Suite")
        print("=" * 60)

        total_start_time = time.time()

        for category in categories:
            print(f"\n🔍 Running {category} tests...")
            try:
                result = self.run_category_tests(category, verbose)
                report.categories[category] = result
                report.total_duration += result.duration

                # Report results
                status = "✅" if result.failed == 0 and result.errors == 0 else "❌"
                print(f"   {status} Duration: {result.duration:.2f}s")
                print(
                    f"   📊 Passed: {result.passed}, Failed: {result.failed}, "
                    f"Errors: {result.errors}, Skipped: {result.skipped}"
                )

                if result.critical_failures:
                    print(f"   ⚠️  Critical failures: {len(result.critical_failures)}")
                    report.critical_issues.extend(result.critical_failures)

                if result.failed > 0 or result.errors > 0:
                    report.overall_passed = False

            except Exception as e:
                print(f"❌ Failed to run {category} tests: {e}")
                report.overall_passed = False
                report.critical_issues.append(f"Test execution failed for {category}: {e}")

        report.total_duration = time.time() - total_start_time

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        return report

    def _generate_recommendations(self, report: RobustnessReport) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        # Check for critical failures
        if report.critical_issues:
            recommendations.append(
                f"🔴 CRITICAL: {len(report.critical_issues)} critical robustness failures detected. "
                "Address immediately before deployment."
            )

        # Check performance issues
        if "performance" in report.categories:
            perf_result = report.categories["performance"]
            if perf_result.failed > 0:
                recommendations.append(
                    "🟡 Performance regressions detected. Review recent changes for optimization opportunities."
                )

        # Check memory issues
        if "memory" in report.categories:
            mem_result = report.categories["memory"]
            if mem_result.failed > 0:
                recommendations.append(
                    "🟡 Memory issues detected. Investigate for potential leaks or excessive allocation."
                )

        # Check concurrency issues
        if "concurrency" in report.categories:
            conc_result = report.categories["concurrency"]
            if conc_result.failed > 0:
                recommendations.append(
                    "🟡 Concurrency safety issues detected. Review thread synchronization and shared state access."
                )

        # Check error recovery
        if "error_recovery" in report.categories:
            err_result = report.categories["error_recovery"]
            if err_result.failed > 0:
                recommendations.append(
                    "🟡 Error recovery issues detected. Improve graceful degradation and fallback mechanisms."
                )

        # Overall assessment
        total_failed = sum(r.failed + r.errors for r in report.categories.values())
        if total_failed == 0:
            recommendations.append(
                "✅ All robustness tests passed. System appears stable and reliable."
            )
        elif total_failed < 5:
            recommendations.append(
                "🟡 Minor robustness issues detected. Address before major releases."
            )
        else:
            recommendations.append(
                "🔴 Significant robustness issues detected. Comprehensive review required."
            )

        return recommendations

    def print_report(self, report: RobustnessReport, verbose: bool = False):
        """Print comprehensive test report."""
        print("\n" + "=" * 80)
        print("📋 FLUJO ROBUSTNESS TEST REPORT")
        print("=" * 80)
        print(f"Timestamp: {report.timestamp}")
        status = "✅ PASSED" if report.overall_passed else "❌ FAILED"
        print(f"Overall Status: {status}")
        print()

        print("📊 CATEGORY RESULTS:")
        print("-" * 60)
        for category, result in report.categories.items():
            status_icon = "✅" if result.failed == 0 and result.errors == 0 else "❌"
            print(f"{category}: {status_icon}")
            print(
                f"      Passed: {result.passed}, Failed: {result.failed}, "
                f"Errors: {result.errors}, Skipped: {result.skipped}"
            )

        if report.critical_issues:
            print(f"\n🚨 CRITICAL ISSUES ({len(report.critical_issues)}):")
            print("-" * 60)
            for issue in report.critical_issues[:10]:
                print(f"• {issue}")
            if len(report.critical_issues) > 10:
                print(f"... and {len(report.critical_issues) - 10} more")

        if report.recommendations:
            print(f"\n💡 RECOMMENDATIONS ({len(report.recommendations)}):")
            print("-" * 60)
            for rec in report.recommendations:
                print(f"• {rec}")

        if "performance" in report.categories:
            perf = report.categories["performance"]
            print("\n⚡ PERFORMANCE SUMMARY:")
            print("-" * 60)
            total_perf_tests = perf.passed + perf.failed + perf.errors + perf.skipped
            success_rate = (perf.passed / total_perf_tests * 100) if total_perf_tests > 0 else 0
            print(f"      Passed: {perf.passed}, Failed: {perf.failed}, Errors: {perf.errors}")
            print(f"      Success rate: {success_rate:.1f}%")
        print("=" * 80)


def main():
    """Main entry point for robustness test runner."""
    parser = argparse.ArgumentParser(description="Flujo Robustness Test Runner")
    parser.add_argument(
        "--categories",
        nargs="*",
        choices=["performance", "memory", "concurrency", "error_recovery", "configuration"],
        help="Specific test categories to run (default: all)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output-json", type=str, help="Save report to JSON file")

    args = parser.parse_args()

    # Find project root
    current_path = Path(__file__).resolve()
    project_root = current_path.parent.parent

    # Run tests
    runner = RobustnessTestRunner(project_root)
    report = runner.run_all_tests(args.categories, args.verbose)

    # Print report
    runner.print_report(report, args.verbose)

    # Save JSON report if requested
    if args.output_json:
        report_dict = {
            "timestamp": report.timestamp,
            "total_duration": report.total_duration,
            "overall_passed": report.overall_passed,
            "categories": {
                cat: {
                    "passed": result.passed,
                    "failed": result.failed,
                    "errors": result.errors,
                    "skipped": result.skipped,
                    "duration": result.duration,
                    "critical_failures": result.critical_failures,
                }
                for cat, result in report.categories.items()
            },
            "recommendations": report.recommendations,
            "critical_issues": report.critical_issues,
        }

        with open(args.output_json, "w") as f:
            json.dump(report_dict, f, indent=2)

        print(f"\n📄 Report saved to: {args.output_json}")

    # Exit with appropriate code
    sys.exit(0 if report.overall_passed else 1)


if __name__ == "__main__":
    main()
