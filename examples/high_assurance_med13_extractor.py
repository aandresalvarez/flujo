#!/usr/bin/env python3
"""
High-Assurance Biomedical Knowledge Extraction with B-Method Inspired Invariants

This example demonstrates how to use Flujo's TreeSearchStep with formal invariants
to build a high-assurance reasoning system for extracting gene-disease relationships
from biomedical literature.

Problem: Extract triplets about the MED13 gene WITHOUT contamination from its paralog MED13L.

Solution: Use static invariants (hard-coded rules) and discovery agents (LLM-deduced rules)
to prune contaminated branches before they cost you money.

Key Concepts:
- Static Invariants: Pre-defined rules checked in Python (0ms cost)
- Discovery Agents: LLMs that analyze goals and deduce safety rules
- Pre-emptive Pruning: Kill bad branches before expensive evaluation
- Backtracking: Return to safe states when invariants are violated
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import BaseModel

from flujo.application.runner import Flujo
from flujo.domain.dsl.tree_search import TreeSearchStep
from flujo.domain.evaluation import EvaluationReport
from flujo.domain.models import PipelineContext
from flujo.testing.utils import gather_result


# ============================================================================
# DOMAIN MODELS
# ============================================================================


class Triplet(BaseModel):
    """A subject-relation-object triplet with evidence."""

    subject: str
    relation: str
    object: str
    evidence_quote: str


# ============================================================================
# MOCK DATA: Simulated PDF chunks with MED13 and MED13L mentions
# ============================================================================

MOCK_CHUNKS = {
    "chunk_1": """
    The MED13 gene encodes a subunit of the Mediator complex. 
    Studies show MED13 is associated with cardiovascular disease.
    Evidence: "MED13 mutations were found in 12% of patients with CHD."
    """,
    "chunk_2": """
    MED13L is a paralog of MED13 with distinct functions.
    MED13L mutations cause intellectual disability and facial dysmorphism.
    Evidence: "MED13L haploinsufficiency leads to developmental delay."
    """,
    "chunk_3": """
    The Mediator complex, including MED13, regulates transcription.
    MED13 interacts with CDK8 to control gene expression.
    Evidence: "MED13-CDK8 interaction is critical for metabolic regulation."
    """,
    "chunk_4": """
    Both MED13 and MED13L are part of the Mediator family.
    However, MED13 specifically regulates cardiac development.
    Evidence: "MED13 knockout mice exhibit severe cardiac defects."
    """,
}


# ============================================================================
# PROPOSER: Extracts candidate triplets from text
# ============================================================================


async def triplet_proposer(prompt: str) -> list[dict[str, Any]]:
    """
    Mock proposer that extracts triplets from the current chunk.
    In a real system, this would be an LLM call.

    The proposer receives a prompt with the current candidate and should
    return next steps to explore from that candidate.
    """
    # Parse the current candidate from the prompt
    current_candidate = None
    try:
        if "Candidate:" in prompt:
            candidate_line = prompt.split("Candidate:")[1].strip().split("\n")[0]
            current_candidate = json.loads(candidate_line)
    except Exception:
        # Root node - no candidate yet
        pass

    # Determine which chunk to process next
    if current_candidate is None:
        # Root: start with chunk_1
        chunk_id = "chunk_1"
    else:
        # Get current chunk and move to next
        current_chunk = current_candidate.get("chunk_id", "chunk_1")
        chunk_map = {
            "chunk_1": "chunk_2",
            "chunk_2": "chunk_3",
            "chunk_3": "chunk_4",
            "chunk_4": None,
        }
        chunk_id = chunk_map.get(current_chunk)

        if chunk_id is None:
            # No more chunks to process
            return []

    text = MOCK_CHUNKS.get(chunk_id, "")
    if not text:
        return []

    # Simulate extraction (in reality, LLM would do this)
    candidates = []

    if "MED13L" in text and "MED13 " not in text.replace("MED13L", ""):
        # Pure MED13L chunk - this should be caught by invariants!
        candidates.append(
            {
                "subject": "MED13L",
                "relation": "causes",
                "object": "intellectual disability",
                "evidence_quote": "MED13L haploinsufficiency leads to developmental delay.",
                "chunk_id": chunk_id,
            }
        )
    elif "MED13 " in text:
        # Extract MED13 triplets
        if "cardiovascular" in text:
            candidates.append(
                {
                    "subject": "MED13",
                    "relation": "associated_with",
                    "object": "cardiovascular disease",
                    "evidence_quote": "MED13 mutations were found in 12% of patients with CHD.",
                    "chunk_id": chunk_id,
                }
            )
        if "CDK8" in text:
            candidates.append(
                {
                    "subject": "MED13",
                    "relation": "interacts_with",
                    "object": "CDK8",
                    "evidence_quote": "MED13-CDK8 interaction is critical for metabolic regulation.",
                    "chunk_id": chunk_id,
                }
            )
        if "cardiac" in text:
            candidates.append(
                {
                    "subject": "MED13",
                    "relation": "regulates",
                    "object": "cardiac development",
                    "evidence_quote": "MED13 knockout mice exhibit severe cardiac defects.",
                    "chunk_id": chunk_id,
                }
            )

    # If both mentioned, might accidentally include MED13L
    if "MED13L" in text and "MED13 " in text:
        # This simulates an LLM mistake - including the wrong gene
        candidates.append(
            {
                "subject": "MED13L",
                "relation": "part_of",
                "object": "Mediator family",
                "evidence_quote": "Both MED13 and MED13L are part of the Mediator family.",
                "chunk_id": chunk_id,
            }
        )

    return candidates


# ============================================================================
# EVALUATOR: Scores candidate triplets
# ============================================================================


async def triplet_evaluator(prompt: str) -> EvaluationReport:
    """
    Mock evaluator that scores triplets based on evidence quality.
    In a real system, this would be an LLM call.
    """
    # Parse the candidate from the prompt
    try:
        candidate_marker = "Candidate:"
        if candidate_marker in prompt:
            candidate_json = prompt.split(candidate_marker)[1].strip().split("\n")[0]
            candidate = json.loads(candidate_json)
        else:
            return EvaluationReport(score=0.0, hard_fail=True)
    except Exception:
        return EvaluationReport(score=0.0, hard_fail=True)

    # Score based on evidence quality
    subject = candidate.get("subject", "")
    evidence = candidate.get("evidence_quote", "")

    # Check for contamination (this is what invariants will catch earlier!)
    if "MED13L" in subject:
        # Evaluator detects contamination, but we've already spent tokens!
        return EvaluationReport(
            score=0.0,
            hard_fail=True,
            metadata={"error": "MED13L_CONTAMINATION"},
        )

    # Score based on evidence specificity
    if "mutations" in evidence or "knockout" in evidence:
        score = 0.95  # Strong evidence
    elif "interaction" in evidence or "regulates" in evidence:
        score = 0.85  # Good evidence
    elif "associated" in evidence:
        score = 0.75  # Moderate evidence
    else:
        score = 0.5  # Weak evidence

    return EvaluationReport(
        score=score,
        hard_fail=False,
        metadata={"subject": subject, "evidence_strength": "high" if score > 0.8 else "medium"},
    )


# ============================================================================
# DISCOVERY AGENT: Deduces invariants from the goal
# ============================================================================


async def discovery_agent(prompt: str) -> list[str]:
    """
    Mock discovery agent that analyzes the goal and deduces safety rules.
    In a real system, this would be an LLM call.
    """
    # Simulate LLM analysis of the goal
    if "MED13" in prompt and "paralog" not in prompt.lower():
        # Deduce that we should avoid the paralog
        return [
            "'MED13L' not in str(output).upper()",
            "'13L' not in str(output)",
            "len(str(output)) > 10",  # No empty results
        ]

    # Generic safety rules
    return [
        "len(str(output)) > 0",  # No empty outputs
    ]


# ============================================================================
# STATIC INVARIANTS: Hard-coded safety rules
# ============================================================================

# Expression-based invariant (checked via compile_expression_to_callable)
STATIC_INVARIANT_EXPR = "'MED13L' not in str(output).upper()"


# Callable-based invariant
def no_med13l_callable(output: object, context: BaseModel | None = None) -> bool:
    """Invariant: Output must not mention MED13L."""
    output_str = str(output).upper()
    return "MED13L" not in output_str and "13L" not in output_str


# Non-empty invariant
def non_empty_output(output: object, context: BaseModel | None = None) -> bool:
    """Invariant: Output must not be empty."""
    if output is None:
        return False
    if isinstance(output, str) and not output.strip():
        return False
    if isinstance(output, (list, dict)) and len(output) == 0:
        return False
    return True


# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================


async def demo_without_invariants() -> None:
    """Demonstrate tree search WITHOUT invariants - allows contamination."""
    print("\n" + "=" * 80)
    print("DEMO 1: Tree Search WITHOUT Invariants")
    print("=" * 80)
    print("This will allow MED13L contamination and waste tokens evaluating bad candidates.\n")

    step = TreeSearchStep(
        name="med13_extractor_no_guards",
        proposer=triplet_proposer,
        evaluator=triplet_evaluator,
        branching_factor=3,
        beam_width=3,
        max_depth=4,  # Allow deeper search
        max_iterations=20,
        goal_score_threshold=0.9,
        require_goal=False,
    )

    runner = Flujo(step, context_model=PipelineContext, persist_state=False)
    result = await gather_result(
        runner,
        "Start extraction",  # Simple initial input
        initial_context_data={
            "initial_prompt": "Extract MED13 gene triplets from biomedical literature"
        },
    )

    output = getattr(result, "output", None) or getattr(result, "final_output", None)
    print(f"Result: {output}")
    print(f"Total cost: ${getattr(result, 'total_cost_usd', 0.0):.4f}")
    print(f"Total tokens: {getattr(result, 'total_tokens', 0)}")

    # Check for contamination in trace
    context = getattr(result, "context", None)
    if context:
        state = getattr(context, "tree_search_state", None)
        if state:
            violations = [e for e in state.trace if e.get("event") == "invariant_violation"]
            print(f"Invariant violations detected: {len(violations)}")
            print(f"Total nodes explored: {len(state.nodes)}")


async def demo_with_static_invariants() -> None:
    """Demonstrate tree search WITH static invariants - prevents contamination."""
    print("\n" + "=" * 80)
    print("DEMO 2: Tree Search WITH Static Invariants")
    print("=" * 80)
    print("Static invariants will catch MED13L mentions BEFORE evaluation, saving tokens.\n")

    step = TreeSearchStep(
        name="med13_extractor_with_guards",
        proposer=triplet_proposer,
        evaluator=triplet_evaluator,
        branching_factor=3,
        beam_width=3,
        max_depth=4,
        max_iterations=20,
        goal_score_threshold=0.9,
        require_goal=False,
    )

    # Add static invariants to the step
    step.static_invariants = [
        STATIC_INVARIANT_EXPR,  # String expression
        no_med13l_callable,  # Callable
        non_empty_output,  # Another callable
    ]

    runner = Flujo(step, context_model=PipelineContext, persist_state=False)
    result = await gather_result(
        runner,
        "Start extraction",
        initial_context_data={
            "initial_prompt": "Extract MED13 gene triplets from biomedical literature"
        },
    )

    output = getattr(result, "output", None) or getattr(result, "final_output", None)
    print(f"Result: {output}")
    print(f"Total cost: ${getattr(result, 'total_cost_usd', 0.0):.4f}")
    print(f"Total tokens: {getattr(result, 'total_tokens', 0)}")

    # Show invariant violations
    context = getattr(result, "context", None)
    if context:
        state = getattr(context, "tree_search_state", None)
        if state:
            violations = [e for e in state.trace if e.get("event") == "invariant_violation"]
            print(f"\nInvariant violations caught: {len(violations)}")
            for v in violations[:3]:  # Show first 3
                viols = v.get("violations", [])
                if viols:
                    print(f"  - Node {v.get('node_id')}: {viols[0].get('rule', 'unknown')}")
            print(f"Total nodes explored: {len(state.nodes)}")


async def demo_with_discovery_agent() -> None:
    """Demonstrate tree search WITH discovery agent - dynamic invariants."""
    print("\n" + "=" * 80)
    print("DEMO 3: Tree Search WITH Discovery Agent")
    print("=" * 80)
    print("Discovery agent will analyze the goal and deduce additional safety rules.\n")

    step = TreeSearchStep(
        name="med13_extractor_with_discovery",
        proposer=triplet_proposer,
        evaluator=triplet_evaluator,
        discovery_agent=discovery_agent,  # Add discovery agent
        branching_factor=3,
        beam_width=3,
        max_depth=4,
        max_iterations=20,
        goal_score_threshold=0.9,
        require_goal=False,
    )

    # Still include static invariants as a baseline
    step.static_invariants = [non_empty_output]

    runner = Flujo(step, context_model=PipelineContext, persist_state=False)
    result = await gather_result(
        runner,
        "Start extraction",
        initial_context_data={
            "initial_prompt": "Extract MED13 gene triplets, avoiding paralog MED13L"
        },
    )

    output = getattr(result, "output", None) or getattr(result, "final_output", None)
    print(f"Result: {output}")
    print(f"Total cost: ${getattr(result, 'total_cost_usd', 0.0):.4f}")
    print(f"Total tokens: {getattr(result, 'total_tokens', 0)}")

    # Show discovered invariants
    context = getattr(result, "context", None)
    if context:
        state = getattr(context, "tree_search_state", None)
        if state:
            discovered = state.deduced_invariants or []
            print(f"\nDiscovered invariants: {len(discovered)}")
            for inv in discovered:
                print(f"  - {inv}")

            violations = [e for e in state.trace if e.get("event") == "invariant_violation"]
            print(f"\nTotal violations caught: {len(violations)}")
            print(f"Total nodes explored: {len(state.nodes)}")


async def main() -> None:
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("HIGH-ASSURANCE BIOMEDICAL KNOWLEDGE EXTRACTION")
    print("Demonstrating B-Method Inspired Invariants in Flujo")
    print("=" * 80)

    # Run demonstrations
    await demo_without_invariants()
    await demo_with_static_invariants()
    await demo_with_discovery_agent()

    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
1. Static Invariants provide 0ms pre-emptive pruning of bad candidates
2. Discovery Agents adapt safety rules to the specific goal
3. Invariant violations are logged in the search trace for debugging
4. Cost savings come from avoiding expensive evaluations of doomed branches
5. Combine both approaches for maximum safety and flexibility
    """)


if __name__ == "__main__":
    asyncio.run(main())
