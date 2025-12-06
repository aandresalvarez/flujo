#!/usr/bin/env python3
"""
Test Health Monitor for Flujo Test Suite

This script monitors the health of the test suite over time,
tracking metrics like pass rates, execution times, and resource usage.
"""

import json
import datetime
import subprocess
import time

try:
    import psutil
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "psutil is required to run test_health_monitor. Install with `pip install psutil`."
    ) from exc
from pathlib import Path
from typing import Any, Optional


class TestHealthMonitor:
    """Monitor test suite health over time."""

    def __init__(self, health_file: str = "test_health.json"):
        self.health_file = Path(health_file)
        self.load_health_data()

    def load_health_data(self) -> None:
        """Load historical health data."""
        if self.health_file.exists():
            with open(self.health_file) as f:
                self.health_data = json.load(f)
        else:
            self.health_data = {
                "runs": [],
                "metadata": {"created": datetime.datetime.now().isoformat()},
            }

    def record_run(
        self,
        passed: int,
        failed: int,
        skipped: int,
        duration: float,
        memory_usage: Optional[float] = None,
        cpu_usage: Optional[float] = None,
    ) -> None:
        """Record test run results."""
        run_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "duration": duration,
            "total": passed + failed + skipped,
            "success_rate": passed / (passed + failed) if (passed + failed) > 0 else 0,
            "memory_usage_mb": memory_usage,
            "cpu_usage_percent": cpu_usage,
        }

        self.health_data["runs"].append(run_data)

        # Keep only last 100 runs
        if len(self.health_data["runs"]) > 100:
            self.health_data["runs"] = self.health_data["runs"][-100:]

        self.save_health_data()

    def save_health_data(self) -> None:
        """Save health data to file."""
        with open(self.health_file, "w") as f:
            json.dump(self.health_data, f, indent=2)

    def get_health_report(self) -> dict[str, Any]:
        """Generate health report."""
        if not self.health_data["runs"]:
            return {"status": "no_data", "message": "No test runs recorded yet"}

        recent_runs = self.health_data["runs"][-10:]  # Last 10 runs
        all_runs = self.health_data["runs"]

        # Calculate metrics
        avg_success_rate = sum(r["success_rate"] for r in recent_runs) / len(recent_runs)
        avg_duration = sum(r["duration"] for r in recent_runs) / len(recent_runs)
        total_runs = len(all_runs)

        # Determine status
        if avg_success_rate >= 0.99:
            status = "excellent"
        elif avg_success_rate >= 0.95:
            status = "healthy"
        elif avg_success_rate >= 0.90:
            status = "degraded"
        else:
            status = "critical"

        # Calculate trends
        if len(recent_runs) >= 2:
            recent_avg = sum(r["success_rate"] for r in recent_runs[-5:]) / 5
            older_avg = sum(r["success_rate"] for r in recent_runs[:-5]) / 5
            trend = (
                "improving"
                if recent_avg > older_avg
                else "declining"
                if recent_avg < older_avg
                else "stable"
            )
        else:
            trend = "insufficient_data"

        return {
            "status": status,
            "avg_success_rate": avg_success_rate,
            "avg_duration": avg_duration,
            "total_runs": total_runs,
            "recent_runs": len(recent_runs),
            "trend": trend,
            "last_run": recent_runs[-1]["timestamp"] if recent_runs else None,
        }

    def run_test_suite(self, test_command: str = "make test-fast") -> dict[str, Any]:
        """Run test suite and record results."""
        print(f"🧪 Running test suite: {test_command}")

        # Monitor system resources
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu = process.cpu_percent()

        start_time = time.perf_counter()

        try:
            # Run the test command
            result = subprocess.run(
                test_command.split(),
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            duration = time.perf_counter() - start_time

            # Parse results from output
            output_lines = result.stdout.split("\n")
            passed = failed = skipped = 0

            for line in output_lines:
                if "passed" in line and "failed" in line and "skipped" in line:
                    # Extract numbers from line like "2251 passed, 6 skipped, 135 warnings"
                    parts = line.split(",")
                    for part in parts:
                        part = part.strip()
                        if "passed" in part:
                            passed = int(part.split()[0])
                        elif "failed" in part:
                            failed = int(part.split()[0])
                        elif "skipped" in part:
                            skipped = int(part.split()[0])
                    break

            # Get final resource usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            # Record the run
            self.record_run(
                passed=passed,
                failed=failed,
                skipped=skipped,
                duration=duration,
                memory_usage=memory_increase,
                cpu_usage=initial_cpu,
            )

            return {
                "success": result.returncode == 0,
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "duration": duration,
                "memory_increase_mb": memory_increase,
                "output": result.stdout,
                "error": result.stderr,
            }

        except subprocess.TimeoutExpired:
            duration = time.perf_counter() - start_time
            print(f"⏰ Test suite timed out after {duration:.1f} seconds")
            return {"success": False, "error": "Test suite timed out", "duration": duration}
        except Exception as e:
            duration = time.perf_counter() - start_time
            print(f"❌ Error running test suite: {e}")
            return {"success": False, "error": str(e), "duration": duration}


def main():
    """Main function to run test health monitoring."""
    monitor = TestHealthMonitor()

    print("🏥 Flujo Test Health Monitor")
    print("=" * 50)

    # Get current health report
    report = monitor.get_health_report()
    print(f"📊 Current Status: {report['status'].upper()}")
    print(f"📈 Success Rate: {report['avg_success_rate']:.1%}")
    print(f"⏱️  Avg Duration: {report['avg_duration']:.1f}s")
    print(f"📋 Total Runs: {report['total_runs']}")
    print(f"📈 Trend: {report['trend']}")

    print("\n" + "=" * 50)

    # Ask user if they want to run tests
    response = input("Run test suite now? (y/n): ").lower().strip()

    if response in ["y", "yes"]:
        print("\n🚀 Running test suite...")
        result = monitor.run_test_suite()

        print("\n📊 Test Run Results:")
        print(f"✅ Passed: {result.get('passed', 0)}")
        print(f"❌ Failed: {result.get('failed', 0)}")
        print(f"⏭️  Skipped: {result.get('skipped', 0)}")
        print(f"⏱️  Duration: {result.get('duration', 0):.1f}s")
        print(f"🧠 Memory Increase: {result.get('memory_increase_mb', 0):.1f}MB")

        if result["success"]:
            print("✅ Test suite completed successfully!")
        else:
            print(f"❌ Test suite failed: {result.get('error', 'Unknown error')}")

    # Show updated health report
    print("\n" + "=" * 50)
    updated_report = monitor.get_health_report()
    print(f"📊 Updated Status: {updated_report['status'].upper()}")
    print(f"📈 Updated Success Rate: {updated_report['avg_success_rate']:.1%}")


if __name__ == "__main__":
    main()
