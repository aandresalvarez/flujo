from flujo.type_definitions.common import JSONObject

"""Baseline management for performance regression tests.

This module provides utilities for managing performance baselines,
measuring current performance, and detecting regressions.
"""
# ruff: noqa

import json
import time
import statistics
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class BaselineMeasurement:
    """Represents a single baseline measurement."""

    name: str
    value: float
    timestamp: datetime
    environment: str


@dataclass
class BaselineConfig:
    """Configuration for a performance baseline."""

    target: float
    tolerance_percent: float
    description: str
    measured_avg: Optional[float] = None
    measured_std: Optional[float] = None
    sample_size: int = 0
    measurements: Optional[List[float]] = None


class BaselineManager:
    """Manages performance baselines and regression detection."""

    def __init__(self, baseline_file: Path):
        self.baseline_file = baseline_file
        self.baselines: Dict[str, BaselineConfig] = {}
        self.metadata: JSONObject = {}
        self._load_baselines()

    def _load_baselines(self) -> None:
        """Load baselines from file."""
        if not self.baseline_file.exists():
            return

        with open(self.baseline_file, "r") as f:
            data = json.load(f)

        self.metadata = data.get("metadata", {})

        for name, config in data.get("baselines", {}).items():
            # Determine target value key
            if "target_ms" in config:
                target = config["target_ms"]
            elif "target_percent" in config:
                target = config["target_percent"]
            elif "target_ratio" in config:
                target = config["target_ratio"]
            else:
                target = config.get("target", 0)

            self.baselines[name] = BaselineConfig(
                target=target,
                tolerance_percent=config["tolerance_percent"],
                description=config["description"],
                measured_avg=config.get("measured_avg_ms")
                or config.get("measured_avg_percent")
                or config.get("measured_avg_ratio"),
                measured_std=config.get("measured_std_ms")
                or config.get("measured_std_percent")
                or config.get("measured_std_ratio"),
                sample_size=config.get("sample_size", 0),
                measurements=config.get("measurements", []),
            )

    def save_baselines(self) -> None:
        """Save current baselines to file."""
        data = {
            "metadata": {
                **self.metadata,
                "last_updated": datetime.now().strftime("%Y-%m-%d"),
                "version": "1.0",
            },
            "baselines": {},
        }

        for name, config in self.baselines.items():
            baseline_data = {
                "target_ms"
                if "ms" in name
                else "target_percent"
                if "percent" in name
                else "target_ratio"
                if "ratio" in name
                else "target": config.target,
                "tolerance_percent": config.tolerance_percent,
                "description": config.description,
                "sample_size": config.sample_size,
            }

            if config.measured_avg is not None:
                key = (
                    "measured_avg_ms"
                    if "ms" in name
                    else "measured_avg_percent"
                    if "percent" in name
                    else "measured_avg_ratio"
                )
                baseline_data[key] = config.measured_avg

            if config.measured_std is not None:
                key = (
                    "measured_std_ms"
                    if "ms" in name
                    else "measured_std_percent"
                    if "percent" in name
                    else "measured_std_ratio"
                )
                baseline_data[key] = config.measured_std

            if config.measurements:
                baseline_data["measurements"] = config.measurements[
                    -100:
                ]  # Keep last 100 measurements

            data["baselines"][name] = baseline_data

        with open(self.baseline_file, "w") as f:
            json.dump(data, f, indent=2)

    def get_baseline(self, name: str) -> Optional[BaselineConfig]:
        """Get baseline configuration for a measurement."""
        return self.baselines.get(name)

    def add_measurement(self, name: str, value: float) -> None:
        """Add a measurement to a baseline."""
        if name not in self.baselines:
            # Create new baseline with default tolerance
            self.baselines[name] = BaselineConfig(
                target=value,  # Initial target is the first measurement
                tolerance_percent=20.0,
                description=f"Auto-created baseline for {name}",
                measurements=[],
            )

        config = self.baselines[name]
        if config.measurements is None:
            config.measurements = []

        config.measurements.append(value)
        config.sample_size = len(config.measurements)

        # Update statistics if we have enough samples
        if config.sample_size >= 5:
            config.measured_avg = statistics.mean(config.measurements)
            config.measured_std = statistics.stdev(config.measurements)

    def check_regression(self, name: str, current_value: float) -> Tuple[bool, str]:
        """Check if current measurement indicates a regression.

        Returns:
            (is_regression, message)
        """
        config = self.get_baseline(name)
        if not config:
            return False, f"No baseline found for {name}"

        # Use measured average as reference, or target if no measurements
        reference = config.measured_avg if config.measured_avg else config.target

        # Calculate tolerance
        tolerance = reference * (config.tolerance_percent / 100.0)

        # Check for regression (current value worse than reference + tolerance)
        if "ratio" in name:
            # For ratios, higher is better
            is_regression = current_value < (reference - tolerance)
            status = "REGRESSION" if is_regression else "OK"
            message = f"{name}: {current_value:.2f}x ({status}, target: {reference:.2f}x ±{tolerance:.2f})"
        elif "percent" in name:
            # For percentages, lower is better
            is_regression = current_value > (reference + tolerance)
            status = "REGRESSION" if is_regression else "OK"
            message = (
                f"{name}: {current_value:.1f}% ({status}, target: ≤{reference + tolerance:.1f}%)"
            )
        else:
            # For time measurements, lower is better
            is_regression = current_value > (reference + tolerance)
            status = "REGRESSION" if is_regression else "OK"
            message = (
                f"{name}: {current_value:.2f}ms ({status}, target: ≤{reference + tolerance:.2f}ms)"
            )

        return is_regression, message

    def measure_performance(self, func, name: str, iterations: int = 10) -> Tuple[float, float]:
        """Measure performance of a function.

        Returns:
            (average_time_ms, standard_deviation_ms)
        """
        times = []

        for _ in range(iterations):
            start_time = time.perf_counter()
            func()
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)

        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0.0

        # Add measurement to baseline
        self.add_measurement(name, avg_time)

        return avg_time, std_time

    def update_baseline_target(self, name: str, new_target: float) -> None:
        """Update the target value for a baseline."""
        if name in self.baselines:
            self.baselines[name].target = new_target

    def get_baseline_summary(self) -> str:
        """Get a summary of all baselines."""
        lines = ["Performance Baselines Summary:", "=" * 40]

        for name, config in self.baselines.items():
            lines.append(f"{name}:")
            lines.append(f"  Target: {config.target}")
            lines.append(f"  Tolerance: {config.tolerance_percent}%")
            lines.append(f"  Description: {config.description}")

            if config.measured_avg:
                lines.append(f"  Measured Avg: {config.measured_avg:.2f}")
            if config.measured_std:
                lines.append(f"  Measured Std: {config.measured_std:.2f}")
            lines.append(f"  Sample Size: {config.sample_size}")
            lines.append("")

        return "\n".join(lines)


# Global baseline manager instance
_baseline_manager = None


def get_baseline_manager() -> BaselineManager:
    """Get the global baseline manager instance."""
    global _baseline_manager
    if _baseline_manager is None:
        baseline_file = Path(__file__).parent / "baselines.json"
        _baseline_manager = BaselineManager(baseline_file)
    return _baseline_manager


def measure_and_check_regression(
    name: str, func, iterations: int = 10
) -> Tuple[bool, str, float, float]:
    """Measure performance and check for regression.

    Returns:
        (is_regression, message, avg_time, std_dev)
    """
    manager = get_baseline_manager()
    avg_time, std_dev = manager.measure_performance(func, name, iterations)
    is_regression, message = manager.check_regression(name, avg_time)
    return is_regression, message, avg_time, std_dev
