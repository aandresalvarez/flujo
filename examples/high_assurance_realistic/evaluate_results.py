#!/usr/bin/env python3
"""
Evaluation Script for High-Assurance Gene Extraction Results

This script analyzes the results saved by main.py and produces a detailed
evaluation report covering:

1. Search Efficiency: How well did the search explore the space?
2. Invariant Effectiveness: How many violations were caught?
3. Cost Analysis: What were the actual costs vs. estimates?
4. Quality Metrics: How good are the extracted triplets?
5. Comparative Analysis: How does this run compare to others?

================================================================================
USAGE
================================================================================

Evaluate latest run:
  python evaluate_results.py

Evaluate specific run:
  python evaluate_results.py results/run_20251225_155700.json

Compare multiple runs:
  python evaluate_results.py results/run_*.json

================================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EXAMPLE_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = EXAMPLE_DIR.parent
REPO_ROOT = EXAMPLES_DIR.parent
for path in (str(EXAMPLES_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from high_assurance_realistic.models import SearchResults  # noqa: E402


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================


def analyze_search_efficiency(results: SearchResults) -> dict[str, Any]:
    """
    Analyze how efficiently the search explored the space.

    Metrics:
    - Nodes explored vs. theoretical maximum
    - Goal reached?
    - Average score progression

    Args:
        results: SearchResults to analyze

    Returns:
        Dictionary of efficiency metrics
    """
    metrics = {
        "total_nodes": results.total_nodes,
        "goal_reached": results.goal_reached,
        "execution_time_s": results.execution_time_s,
    }

    # Calculate nodes per second
    if results.execution_time_s > 0:
        metrics["nodes_per_second"] = results.total_nodes / results.execution_time_s

    # Check if search was efficient (reached goal quickly)
    if results.goal_reached and results.total_nodes < 15:
        metrics["efficiency_rating"] = "Excellent"
    elif results.goal_reached:
        metrics["efficiency_rating"] = "Good"
    elif results.total_nodes > 20:
        metrics["efficiency_rating"] = "Poor (explored many nodes without reaching goal)"
    else:
        metrics["efficiency_rating"] = "Fair"

    return metrics


def analyze_invariant_effectiveness(results: SearchResults) -> dict[str, Any]:
    """
    Analyze how effective invariants were at preventing contamination.

    Metrics:
    - Total violations caught
    - Violations by rule
    - Pruning rate (violations / total candidates)

    Args:
        results: SearchResults to analyze

    Returns:
        Dictionary of invariant metrics
    """
    violations = results.invariant_violations

    # Count violations by rule
    rule_counts: dict[str, int] = {}
    for violation in violations:
        viols = violation.get("violations", [])
        for v in viols:
            rule = v.get("rule", "unknown")
            rule_counts[rule] = rule_counts.get(rule, 0) + 1

    # Calculate pruning rate
    # Estimate: each node expansion generates ~3 candidates
    estimated_candidates = results.total_nodes * 3
    pruning_rate = len(violations) / estimated_candidates if estimated_candidates > 0 else 0.0

    metrics = {
        "total_violations": len(violations),
        "unique_rules_violated": len(rule_counts),
        "violations_by_rule": rule_counts,
        "estimated_pruning_rate": pruning_rate,
    }

    # Effectiveness rating
    if len(violations) > 5:
        metrics["effectiveness_rating"] = "High (caught many contaminations)"
    elif len(violations) > 0:
        metrics["effectiveness_rating"] = "Medium (caught some contaminations)"
    else:
        metrics["effectiveness_rating"] = "Low (no violations, possibly too lenient)"

    return metrics


def analyze_cost(results: SearchResults) -> dict[str, Any]:
    """
    Analyze the cost of the run.

    Metrics:
    - Total cost
    - Cost per node
    - Cost per token
    - Savings from pruning

    Args:
        results: SearchResults to analyze

    Returns:
        Dictionary of cost metrics
    """
    metrics = {
        "total_cost_usd": results.total_cost_usd,
        "total_tokens": results.total_tokens,
    }

    # Cost per node
    if results.total_nodes > 0:
        metrics["cost_per_node"] = results.total_cost_usd / results.total_nodes

    # Cost per token
    if results.total_tokens > 0:
        metrics["cost_per_1k_tokens"] = (results.total_cost_usd / results.total_tokens) * 1000

    # Estimate savings from pruning
    # Without invariants, we would have evaluated all candidates
    violations = len(results.invariant_violations)
    if violations > 0:
        # Assume each violation saved ~$0.003 (one evaluator call)
        estimated_savings = violations * 0.003
        metrics["estimated_savings_from_pruning"] = estimated_savings
        metrics["savings_percentage"] = (
            estimated_savings / (results.total_cost_usd + estimated_savings)
        ) * 100

    return metrics


def analyze_quality(results: SearchResults) -> dict[str, Any]:
    """
    Analyze the quality of extracted triplets.

    Metrics:
    - Best triplet score
    - Average triplet score
    - Triplets with high confidence

    Args:
        results: SearchResults to analyze

    Returns:
        Dictionary of quality metrics
    """
    metrics = {
        "total_triplets": len(results.all_triplets),
        "best_triplet_found": results.best_triplet is not None,
    }

    if results.best_triplet:
        metrics["best_triplet_confidence"] = results.best_triplet.confidence
        metrics["best_triplet_subject"] = results.best_triplet.subject
        metrics["best_triplet_relation"] = results.best_triplet.relation

    # Analyze all triplets
    if results.all_triplets:
        confidences = [t.confidence for t in results.all_triplets if t.confidence > 0]
        if confidences:
            metrics["average_confidence"] = sum(confidences) / len(confidences)
            metrics["high_confidence_triplets"] = len([c for c in confidences if c >= 0.8])

    # Quality rating
    if results.best_triplet and results.best_triplet.confidence >= 0.9:
        metrics["quality_rating"] = "Excellent"
    elif results.best_triplet and results.best_triplet.confidence >= 0.7:
        metrics["quality_rating"] = "Good"
    elif results.best_triplet:
        metrics["quality_rating"] = "Fair"
    else:
        metrics["quality_rating"] = "Poor (no valid triplet found)"

    return metrics


# ============================================================================
# REPORT GENERATION
# ============================================================================


def generate_report(results: SearchResults) -> str:
    """
    Generate a comprehensive evaluation report.

    Args:
        results: SearchResults to report on

    Returns:
        Formatted report string
    """
    # Run analyses
    efficiency = analyze_search_efficiency(results)
    invariants = analyze_invariant_effectiveness(results)
    cost = analyze_cost(results)
    quality = analyze_quality(results)

    # Build report
    lines = []
    lines.append("=" * 80)
    lines.append("EVALUATION REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Header
    lines.append(f"Run ID: {results.run_id}")
    lines.append(f"Goal: {results.goal}")
    lines.append("")

    # Search Efficiency
    lines.append("SEARCH EFFICIENCY")
    lines.append("-" * 80)
    lines.append(f"Total Nodes Explored: {efficiency['total_nodes']}")
    lines.append(f"Execution Time: {efficiency['execution_time_s']:.2f}s")
    if "nodes_per_second" in efficiency:
        lines.append(f"Nodes/Second: {efficiency['nodes_per_second']:.2f}")
    lines.append(f"Goal Reached: {'✅ Yes' if efficiency['goal_reached'] else '❌ No'}")
    lines.append(f"Efficiency Rating: {efficiency['efficiency_rating']}")
    lines.append("")

    # Invariant Effectiveness
    lines.append("INVARIANT EFFECTIVENESS")
    lines.append("-" * 80)
    lines.append(f"Total Violations Caught: {invariants['total_violations']}")
    lines.append(f"Unique Rules Violated: {invariants['unique_rules_violated']}")
    lines.append(f"Estimated Pruning Rate: {invariants['estimated_pruning_rate']:.1%}")
    lines.append(f"Effectiveness Rating: {invariants['effectiveness_rating']}")

    if invariants["violations_by_rule"]:
        lines.append("")
        lines.append("Top Violated Rules:")
        sorted_rules = sorted(
            invariants["violations_by_rule"].items(), key=lambda x: x[1], reverse=True
        )
        for rule, count in sorted_rules[:5]:
            lines.append(f"  - {rule}: {count} times")
    lines.append("")

    # Cost Analysis
    lines.append("COST ANALYSIS")
    lines.append("-" * 80)
    lines.append(f"Total Cost: ${cost['total_cost_usd']:.4f}")
    lines.append(f"Total Tokens: {cost['total_tokens']:,}")
    if "cost_per_node" in cost:
        lines.append(f"Cost per Node: ${cost['cost_per_node']:.4f}")
    if "estimated_savings_from_pruning" in cost:
        lines.append(
            f"Estimated Savings from Pruning: ${cost['estimated_savings_from_pruning']:.4f}"
        )
        lines.append(f"Savings Percentage: {cost['savings_percentage']:.1f}%")
    lines.append("")

    # Quality Metrics
    lines.append("QUALITY METRICS")
    lines.append("-" * 80)
    lines.append(f"Total Triplets Extracted: {quality['total_triplets']}")
    lines.append(f"Best Triplet Found: {'✅ Yes' if quality['best_triplet_found'] else '❌ No'}")

    if quality["best_triplet_found"]:
        lines.append(f"Best Triplet Confidence: {quality['best_triplet_confidence']:.2f}")
        lines.append(
            f"Best Triplet: {quality['best_triplet_subject']} → {quality['best_triplet_relation']}"
        )

    if "average_confidence" in quality:
        lines.append(f"Average Confidence: {quality['average_confidence']:.2f}")
        lines.append(f"High-Confidence Triplets (≥0.8): {quality['high_confidence_triplets']}")

    lines.append(f"Quality Rating: {quality['quality_rating']}")
    lines.append("")

    # Discovered Invariants
    if results.discovered_invariants:
        lines.append("DISCOVERED INVARIANTS")
        lines.append("-" * 80)
        for inv in results.discovered_invariants:
            lines.append(f"  - {inv}")
        lines.append("")

    # Summary
    lines.append("=" * 80)
    lines.append("SUMMARY")
    lines.append("=" * 80)
    lines.append(f"✅ Efficiency: {efficiency['efficiency_rating']}")
    lines.append(f"✅ Invariants: {invariants['effectiveness_rating']}")
    lines.append(f"✅ Quality: {quality['quality_rating']}")
    lines.append(f"💰 Total Cost: ${cost['total_cost_usd']:.4f}")
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate high-assurance gene extraction results")
    parser.add_argument(
        "results_files",
        nargs="*",
        default=None,
        help="Paths to results JSON files (default: results/latest.json)",
    )
    args = parser.parse_args()

    # Load results
    default_results = EXAMPLE_DIR / "results/latest.json"
    results_files = args.results_files or [str(default_results)]
    results_paths = [Path(path) for path in results_files]
    missing_paths = [path for path in results_paths if not path.exists()]
    if missing_paths:
        for path in missing_paths:
            print(f"❌ Error: Results file not found: {path}")
        sys.exit(1)

    for idx, results_path in enumerate(results_paths, start=1):
        if len(results_paths) > 1:
            print("=" * 80)
            print(f"RUN {idx}/{len(results_paths)}: {results_path}")
            print("=" * 80)

        print(f"Loading results from: {results_path}")
        with open(results_path) as f:
            data = json.load(f)

        results = SearchResults(**data)
        print("✅ Results loaded")
        print()

        # Generate and display report
        report = generate_report(results)
        print(report)

        # Save report
        report_path = results_path.parent / f"report_{results.run_id}.txt"
        with open(report_path, "w") as f:
            f.write(report)

        print(f"📄 Report saved to: {report_path}")
        if len(results_paths) > 1 and idx < len(results_paths):
            print()


if __name__ == "__main__":
    main()
