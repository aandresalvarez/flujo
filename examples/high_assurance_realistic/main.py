"""
High-Assurance Gene Extraction with Real LLMs

This example demonstrates production-ready invariant-based tree search using
REAL OpenAI and Anthropic API calls to extract gene-disease relationships
from biomedical literature.

================================================================================
RUNNING THIS EXAMPLE
================================================================================

METHOD 1: Direct Python execution
    python main.py
    python main.py --use-pipeline
    python main.py --max-depth 5 --beam-width 4

METHOD 2: Flujo CLI (recommended)
    flujo run main.py --input "Extract MED13 facts"
    flujo run main.py --input "Extract MED13 facts" --json

The Flujo CLI provides additional features like state persistence, live progress,
and better error handling.

================================================================================
PEDAGOGICAL WALKTHROUGH
================================================================================

This script is organized into clear sections with detailed explanations:

1. SETUP & CONFIGURATION
   - API key verification
   - Cost estimation
   - Parameter configuration

2. AGENT INITIALIZATION
   - Real LLM agents (GPT-4o + Claude)
   - Invariant definitions
   - Discovery agent setup

3. TREE SEARCH EXECUTION
   - TreeSearchStep configuration
   - Progress tracking
   - Cost monitoring

4. RESULTS ANALYSIS
   - Extract best triplet
   - Save results to JSON
   - Display summary

================================================================================
REQUIREMENTS
================================================================================

Environment Variables:
  - OPENAI_API_KEY: Your OpenAI API key
  - ANTHROPIC_API_KEY: Your Anthropic API key

Dependencies:
  - flujo (this framework)
  - openai
  - anthropic

Cost Estimate:
  - Proposer (GPT-4o): ~10-20 calls x $0.005 = $0.05-0.10
  - Evaluator (Claude): ~10-20 calls x $0.003 = $0.03-0.06
  - Verification (GPT-4o + Claude): ~2 calls x $0.002 = $0.004
  - Discovery (GPT-4o): ~1 call x $0.002 = $0.002
  - TOTAL: ~$0.084-0.164 per run

================================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = EXAMPLE_DIR.parent
REPO_ROOT = EXAMPLES_DIR.parent
for path in (str(EXAMPLES_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from flujo.application.runner import Flujo  # noqa: E402
from flujo.domain.models import Paused  # noqa: E402
from flujo.exceptions import PipelineAbortSignal  # noqa: E402
from flujo.testing.utils import gather_result  # noqa: E402

# Import our modules
from high_assurance_realistic.agents import (  # noqa: E402
    proposer_agent,
    evaluator_agent,
    discovery_agent,
    verify_api_keys,
    create_full_pipeline,
    create_verification_panel,
    create_consensus_gate_step,
)
from high_assurance_realistic.invariants import STRICT_INVARIANTS  # noqa: E402
from high_assurance_realistic.models import (  # noqa: E402
    SearchResults,
    Triplet,
    GeneExtractionContext,
)
from high_assurance_realistic.sample_data import (  # noqa: E402
    get_all_abstracts,
    format_abstract_for_extraction,
)


# ============================================================================
# MODULE-LEVEL PIPELINE (for `flujo run` support)
# ============================================================================

# This allows running with: flujo run main.py --input "Extract MED13 facts"
pipeline = create_full_pipeline()


# ============================================================================
# SECTION 1: SETUP & CONFIGURATION
# ============================================================================


def verify_environment() -> None:
    """
    Verify that the environment is properly configured.

    This checks for:
    1. Required API keys
    2. Output directory existence

    Raises:
        SystemExit: If environment is not properly configured
    """
    print("=" * 80)
    print("ENVIRONMENT VERIFICATION")
    print("=" * 80)

    # Check API keys
    keys_ok, missing_keys = verify_api_keys()
    if not keys_ok:
        print("\n❌ ERROR: Missing required API keys:")
        for key in missing_keys:
            print(f"  - {key}")
        print("\nPlease set these environment variables:")
        print("  export OPENAI_API_KEY='sk-...'")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)

    print("✅ API keys verified")

    # Ensure results directory exists
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    print(f"✅ Results directory: {results_dir}")

    print()


def estimate_cost(max_depth: int, beam_width: int) -> dict[str, float]:
    """
    Estimate the cost of running the search.

    This provides a rough estimate based on typical tree search behavior.
    Actual costs may vary depending on:
    - How quickly the goal is reached
    - How many candidates are pruned by invariants
    - Token counts in prompts and responses

    Args:
        max_depth: Maximum search depth
        beam_width: Beam width for search

    Returns:
        Dictionary with cost estimates
    """
    # Estimate number of nodes explored
    # In practice, invariants will prune many branches
    estimated_nodes = min(max_depth * beam_width, 20)

    # Cost per call (approximate)
    proposer_cost_per_call = 0.005  # GPT-4o, ~500 tokens
    evaluator_cost_per_call = 0.003  # Claude, ~300 tokens
    verification_cost_per_call = 0.002  # Short classification prompts
    discovery_cost = 0.002  # GPT-4o, ~200 tokens, 1 call

    # Total estimates
    proposer_cost = estimated_nodes * proposer_cost_per_call
    evaluator_cost = estimated_nodes * evaluator_cost_per_call
    verification_cost = verification_cost_per_call * 2
    total_cost = proposer_cost + evaluator_cost + discovery_cost + verification_cost

    return {
        "estimated_nodes": estimated_nodes,
        "proposer_cost": proposer_cost,
        "evaluator_cost": evaluator_cost,
        "verification_cost": verification_cost,
        "discovery_cost": discovery_cost,
        "total_cost": total_cost,
    }


async def run_consensus_guard(payload: dict[str, object]) -> dict[str, object]:
    """
    Run the consensus panel before expensive search steps.
    """
    verification_pipeline = create_verification_panel() >> create_consensus_gate_step()
    runner = Flujo(verification_pipeline, context_model=GeneExtractionContext, persist_state=False)
    result = await gather_result(
        runner,
        payload,
        initial_context_data={"current_payload": payload},
    )
    if isinstance(result, Paused):
        raise PipelineAbortSignal(result.message)
    verified_payload = getattr(result, "output", None) or getattr(result, "final_output", None)
    if not isinstance(verified_payload, dict) or not verified_payload:
        raise PipelineAbortSignal("Consensus gate did not return a payload.")
    return verified_payload


# ============================================================================
# SECTION 2: TREE SEARCH EXECUTION
# ============================================================================


async def run_extraction(
    goal: str,
    max_depth: int = 4,
    beam_width: int = 3,
    use_discovery: bool = True,
    use_pipeline: bool = False,
) -> SearchResults:
    """
    Execute high-assurance gene extraction using tree search.

    This demonstrates TWO approaches:

    APPROACH 1 (use_pipeline=False): Direct TreeSearchStep
    - Simple and direct
    - Good for focused tree search
    - Minimal boilerplate

    APPROACH 2 (use_pipeline=True): Full Pipeline Composition
    - Uses @step decorators
    - Pipeline >> composition
    - Shows complete Flujo DSL
    - Better for complex workflows

    Args:
        goal: Extraction goal
        max_depth: Maximum search depth
        beam_width: Beam width for pruning
        use_discovery: Whether to use discovery agent
        use_pipeline: Whether to use full pipeline composition

    Returns:
        SearchResults object with complete run data
    """
    print("=" * 80)
    print("TREE SEARCH EXECUTION")
    print("=" * 80)
    print(f"Goal: {goal}")
    print(f"Max Depth: {max_depth}")
    print(f"Beam Width: {beam_width}")
    print(f"Discovery Agent: {'Enabled' if use_discovery else 'Disabled'}")
    print(f"Mode: {'Pipeline Composition' if use_pipeline else 'Direct TreeSearchStep'}")
    print()

    # ========================================================================
    # APPROACH 1: Direct TreeSearchStep (original approach)
    # ========================================================================

    if not use_pipeline:
        print("Using Direct TreeSearchStep approach...")

        # Import here to show the direct approach
        from flujo.domain.dsl.tree_search import TreeSearchStep

        step = TreeSearchStep(
            name="med13_extraction",
            proposer=proposer_agent,
            evaluator=evaluator_agent,
            discovery_agent=discovery_agent if use_discovery else None,
            static_invariants=STRICT_INVARIANTS,
            branching_factor=3,
            beam_width=beam_width,
            max_depth=max_depth,
            max_iterations=30,
            goal_score_threshold=0.9,
            require_goal=False,
        )

        print("✅ TreeSearchStep configured")
        print()

        # Load data
        print("Loading sample abstracts...")
        abstracts = get_all_abstracts()
        initial_payload = {
            "text": format_abstract_for_extraction(abstracts[0]),
            "pmid": abstracts[0]["pmid"],
            "total_abstracts": len(abstracts),
        }
        initial_payload = await run_consensus_guard(initial_payload)
        initial_text = str(initial_payload.get("text", ""))
        print(f"✅ Starting with PMID {abstracts[0]['pmid']}")
        print()

    # ========================================================================
    # APPROACH 2: Full Pipeline Composition (demonstrates complete DSL)
    # ========================================================================

    else:
        print("Using Full Pipeline Composition approach...")
        print("This demonstrates:")
        print("  - @step decorators")
        print("  - Pipeline >> composition")
        print("  - Context injection")
        print("  - Step chaining")
        print()

        # Create the complete pipeline
        step = create_full_pipeline(
            max_depth=max_depth,
            beam_width=beam_width,
            use_discovery=use_discovery,
        )

        print("✅ Pipeline created with 8 steps:")
        print("  1. load_abstracts (@step decorator)")
        print("  2. gene_verification (ParallelStep)")
        print("  3. gene_consensus_gate (Step.from_callable)")
        print("  4. preprocess_text (Step.from_callable)")
        print("  5. med13_extraction (TreeSearchStep)")
        print("  6. format_results (@step decorator)")
        print("  7. postprocess_results (Step.from_callable)")
        print("  8. cleanup_context (@step decorator)")
        print()

        # For pipeline, initial input is just the goal
        initial_text = goal

    # ========================================================================
    # Execute search (same for both approaches)
    # ========================================================================

    print("Starting execution...")
    print("(This will make real LLM API calls - costs will be tracked)")
    print()

    start_time = time.time()

    runner = Flujo(step, context_model=GeneExtractionContext, persist_state=False)
    result = await gather_result(
        runner,
        initial_text,
        initial_context_data={"initial_prompt": goal},
    )
    if isinstance(result, Paused):
        raise PipelineAbortSignal(result.message)

    execution_time = time.time() - start_time

    print()
    print("✅ Execution completed")
    print(f"  - Execution time: {execution_time:.2f}s")
    print()

    # ========================================================================
    # Extract results (same for both approaches)
    # ========================================================================

    print("Extracting results...")

    best_output = getattr(result, "output", None) or getattr(result, "final_output", None)

    # Handle pipeline output (nested dict) vs direct output
    if use_pipeline and isinstance(best_output, dict):
        # Pipeline returns formatted dict
        best_triplet_data = best_output.get("best_triplet")
        if best_triplet_data:
            best_triplet = Triplet(**best_triplet_data)
        else:
            best_triplet = None
    else:
        # Direct TreeSearchStep returns Triplet
        if isinstance(best_output, Triplet):
            best_triplet = best_output
        elif isinstance(best_output, dict):
            try:
                best_triplet = Triplet(**best_output)
            except Exception:
                best_triplet = None
        else:
            best_triplet = None

    # Rest of extraction logic remains the same...
    context = getattr(result, "context", None)

    # Standard StepOutcome (Approach 2) nesting: Success -> StepResult -> branch_context
    if context is None and hasattr(result, "step_result"):
        step_res = result.step_result
        context = getattr(step_res, "branch_context", None)

    search_state = getattr(context, "tree_search_state", None) if context else None

    all_triplets = []
    if search_state and search_state.nodes:
        for node in search_state.nodes.values():
            if node.output and isinstance(node.output, (dict, Triplet)):
                try:
                    if isinstance(node.output, Triplet):
                        all_triplets.append(node.output)
                    else:
                        all_triplets.append(Triplet(**node.output))
                except Exception:
                    pass

    violations = []
    if search_state:
        violations = [e for e in search_state.trace if e.get("event") == "invariant_violation"]

    discovered = []
    if search_state:
        discovered = search_state.deduced_invariants or []

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = SearchResults(
        run_id=run_id,
        goal=goal,
        best_triplet=best_triplet,
        all_triplets=all_triplets,
        total_nodes=len(search_state.nodes) if search_state else 0,
        total_cost_usd=getattr(result, "total_cost_usd", 0.0),
        total_tokens=getattr(result, "total_tokens", 0),
        invariant_violations=[v for v in violations],
        search_trace=search_state.trace if search_state else [],
        discovered_invariants=discovered,
        goal_reached=best_triplet is not None and getattr(best_triplet, "confidence", 0.0) >= 0.9,
        execution_time_s=execution_time,
    )

    print("✅ Results extracted")
    print()

    return results


# ============================================================================
# SECTION 3: RESULTS DISPLAY & SAVING
# ============================================================================


def display_results(results: SearchResults) -> None:
    """
    Display results in a human-readable format.

    Args:
        results: SearchResults object to display
    """
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()

    # Basic stats
    print(f"Run ID: {results.run_id}")
    print(f"Goal: {results.goal}")
    print(f"Execution Time: {results.execution_time_s:.2f}s")
    print()

    # Search metrics
    print("Search Metrics:")
    print(f"  - Total Nodes Explored: {results.total_nodes}")
    print(f"  - Invariant Violations: {len(results.invariant_violations)}")
    print(f"  - Goal Reached: {'✅ Yes' if results.goal_reached else '❌ No'}")
    print()

    # Cost metrics
    print("Cost Metrics:")
    print(f"  - Total Cost: ${results.total_cost_usd:.4f}")
    print(f"  - Total Tokens: {results.total_tokens:,}")
    print()

    # Discovered invariants
    if results.discovered_invariants:
        print(f"Discovered Invariants ({len(results.discovered_invariants)}):")
        for inv in results.discovered_invariants:
            print(f"  - {inv}")
        print()

    # Best triplet
    if results.best_triplet:
        print("Best Triplet:")
        print(f"  Subject: {results.best_triplet.subject}")
        print(f"  Relation: {results.best_triplet.relation}")
        print(f"  Object: {results.best_triplet.object}")
        print(f"  Evidence: {results.best_triplet.evidence_quote[:100]}...")
        print(f"  Confidence: {results.best_triplet.confidence:.2f}")
    else:
        print("No valid triplet found")
    print()

    # Violation summary
    if results.invariant_violations:
        print(f"Top Violations ({min(3, len(results.invariant_violations))}):")
        for v in results.invariant_violations[:3]:
            viols = v.get("violations", [])
            if viols:
                print(f"  - Node {v.get('node_id')}: {viols[0].get('rule', 'unknown')}")
        print()


def save_results(results: SearchResults) -> Path:
    """
    Save results to JSON file.

    Args:
        results: SearchResults object to save

    Returns:
        Path to saved file
    """
    results_dir = Path(__file__).parent / "results"
    output_file = results_dir / f"run_{results.run_id}.json"

    # Save to JSON
    with open(output_file, "w") as f:
        json.dump(results.model_dump(mode="json"), f, indent=2)

    # Also save as "latest.json" for convenience
    latest_file = results_dir / "latest.json"
    with open(latest_file, "w") as f:
        json.dump(results.model_dump(mode="json"), f, indent=2)

    print(f"✅ Results saved to: {output_file}")
    print(f"✅ Latest results: {latest_file}")
    print()

    return output_file


# ============================================================================
# SECTION 4: MAIN ENTRY POINT
# ============================================================================


async def main() -> None:
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="High-assurance gene extraction with real LLMs")
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Maximum search depth (default: 4)",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=3,
        help="Beam width for search (default: 3)",
    )
    parser.add_argument(
        "--no-discovery",
        action="store_true",
        help="Disable discovery agent",
    )
    parser.add_argument(
        "--use-pipeline",
        action="store_true",
        help="Use full Pipeline composition (demonstrates complete Flujo DSL)",
    )
    args = parser.parse_args()

    # Verify environment
    verify_environment()

    # Show cost estimate
    cost_est = estimate_cost(args.max_depth, args.beam_width)
    print("=" * 80)
    print("COST ESTIMATE")
    print("=" * 80)
    print(f"Estimated nodes: {cost_est['estimated_nodes']}")
    print(f"Proposer cost: ${cost_est['proposer_cost']:.4f}")
    print(f"Evaluator cost: ${cost_est['evaluator_cost']:.4f}")
    print(f"Verification cost: ${cost_est['verification_cost']:.4f}")
    print(f"Discovery cost: ${cost_est['discovery_cost']:.4f}")
    print(f"TOTAL ESTIMATE: ${cost_est['total_cost']:.4f}")
    print()
    print("Note: Actual costs may be lower due to invariant pruning")
    if args.use_pipeline:
        print()
        print("🔧 Using PIPELINE COMPOSITION mode")
        print("   This demonstrates: @step, >>, Step.from_callable, context injection")
    print()

    # Confirm with user
    response = input("Proceed with extraction? (y/n): ")
    if response.lower() != "y":
        print("Aborted")
        return
    print()

    # Run extraction
    goal = (
        "Extract gene-disease relationships for MED13, avoiding contamination from MED13L paralog"
    )
    try:
        results = await run_extraction(
            goal=goal,
            max_depth=args.max_depth,
            beam_width=args.beam_width,
            use_discovery=not args.no_discovery,
            use_pipeline=args.use_pipeline,
        )
    except PipelineAbortSignal as exc:
        print(f"Aborted: {exc}")
        return

    # Display results
    display_results(results)

    # Save results
    output_file = save_results(results)

    # Next steps
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print(f"1. Review results in: {output_file}")
    print(f"2. Run evaluation: python evaluate_results.py {output_file}")
    print("3. Compare with other runs in results/ directory")
    if not args.use_pipeline:
        print()
        print("💡 TIP: Try --use-pipeline to see full Flujo DSL syntax")
        print("   (decorators, pipeline composition, context injection)")
    print()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
