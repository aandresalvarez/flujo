"""
Invariant definitions for high-assurance gene extraction.

Invariants are formal rules that must ALWAYS hold during tree search.
They provide guardrails to prevent contamination and ensure correctness.

This module defines:
1. Static invariants (pre-defined rules)
2. Helper functions for invariant checking
"""

from __future__ import annotations

from flujo.domain.dsl.step import InvariantRule
from flujo.domain.models import BaseModel


# ============================================================================
# STATIC INVARIANTS: Pre-defined rules checked on every candidate
# ============================================================================


def no_med13l_contamination(output: object, context: BaseModel | None = None) -> bool:
    """
    Invariant: Output must not mention MED13L (paralog gene).

    This is the CRITICAL invariant that prevents contamination.
    MED13L is a different gene with different functions, and mixing
    facts about MED13 and MED13L would corrupt our knowledge base.

    Args:
        output: Candidate output to check
        context: Pipeline context (unused here)

    Returns:
        True if output is clean (no MED13L), False if contaminated
    """
    output_str = str(output).upper()

    # Check for explicit MED13L mentions
    if "MED13L" in output_str:
        return False

    # Check for "13L" pattern (catches "MED 13L", "MED-13L", etc.)
    if "13L" in output_str:
        return False

    return True


def must_mention_med13(output: object, context: BaseModel | None = None) -> bool:
    """
    Invariant: Output must actually mention MED13.

    This ensures we're extracting facts about the target gene,
    not unrelated information.

    Args:
        output: Candidate output to check
        context: Pipeline context (unused here)

    Returns:
        True if output mentions MED13, False otherwise
    """
    output_str = str(output).upper()

    # Must contain "MED13" but NOT "MED13L"
    if "MED13" not in output_str:
        return False

    # Double-check it's not actually MED13L
    if "MED13L" in output_str:
        return False

    return True


def non_empty_output(output: object, context: BaseModel | None = None) -> bool:
    """
    Invariant: Output must not be empty.

    Basic sanity check to avoid wasting evaluator calls on empty results.

    Args:
        output: Candidate output to check
        context: Pipeline context (unused here)

    Returns:
        True if output has content, False if empty
    """
    if output is None:
        return False

    if isinstance(output, str) and not output.strip():
        return False

    if isinstance(output, (list, dict)) and len(output) == 0:
        return False

    # For Triplet objects, check if they have meaningful content
    if hasattr(output, "subject") and hasattr(output, "object"):
        if not output.subject or not output.object:
            return False

    return True


def has_evidence_quote(output: object, context: BaseModel | None = None) -> bool:
    """
    Invariant: Triplets must include evidence quotes.

    We require verbatim quotes to ensure traceability and prevent
    hallucination. Every extracted fact must be grounded in text.

    Args:
        output: Candidate output to check
        context: Pipeline context (unused here)

    Returns:
        True if output has evidence, False otherwise
    """
    # For Triplet objects
    if hasattr(output, "evidence_quote"):
        quote = getattr(output, "evidence_quote", "")
        return bool(quote and len(str(quote).strip()) > 10)

    # For dict representations
    if isinstance(output, dict):
        quote = output.get("evidence_quote", "")
        return bool(quote and len(str(quote).strip()) > 10)

    # For other types, be permissive
    return True


# ============================================================================
# INVARIANT COLLECTIONS: Organized sets of invariants
# ============================================================================

# Core invariants: Always applied
CORE_INVARIANTS: list[InvariantRule] = [
    non_empty_output,
    no_med13l_contamination,
]

# Strict invariants: For high-precision extraction
STRICT_INVARIANTS: list[InvariantRule] = [
    non_empty_output,
    no_med13l_contamination,
    must_mention_med13,
    has_evidence_quote,
]

# Expression-based invariants: Faster to check
EXPRESSION_INVARIANTS: list[str] = [
    "'MED13L' not in str(output).upper()",
    "'13L' not in str(output).upper()",
    "len(str(output)) > 10",
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_invariants_for_mode(mode: str = "strict") -> list[InvariantRule]:
    """
    Get invariants based on extraction mode.

    Args:
        mode: "core" for basic checks, "strict" for comprehensive validation

    Returns:
        List of invariant rules to apply
    """
    if mode == "core":
        return CORE_INVARIANTS
    elif mode == "strict":
        return STRICT_INVARIANTS
    else:
        return CORE_INVARIANTS


def validate_output_manually(output: object) -> tuple[bool, list[str]]:
    """
    Manually validate output against all invariants.

    Useful for debugging or testing invariants outside of tree search.

    Args:
        output: Output to validate

    Returns:
        Tuple of (is_valid, list_of_violations)
    """
    violations = []

    for invariant in STRICT_INVARIANTS:
        try:
            if callable(invariant):
                if not invariant(output, None):
                    name = getattr(invariant, "__name__", str(invariant))
                    violations.append(f"Failed: {name}")
        except Exception as e:
            violations.append(f"Error in {invariant}: {e}")

    return len(violations) == 0, violations
