"""
Domain models for high-assurance gene extraction.

This module defines the data structures used throughout the example:
- Triplet: Gene-disease relationship with evidence
- SearchResults: Complete search results for analysis
"""

from __future__ import annotations

from pydantic import BaseModel as PydanticBaseModel, Field

from flujo.domain.models import PipelineContext


class Triplet(PydanticBaseModel):
    """
    A gene-disease relationship triplet with supporting evidence.

    This represents a single extracted fact from biomedical literature.
    Example: (MED13, causes, congenital_heart_disease)
    """

    subject: str = Field(
        description="Gene name (e.g., 'MED13')",
        examples=["MED13", "BRCA1", "TP53"],
    )

    relation: str = Field(
        description="Relationship type between gene and disease",
        examples=["causes", "associated_with", "regulates", "interacts_with"],
    )

    object: str = Field(
        description="Disease, phenotype, or biological process",
        examples=["congenital heart disease", "breast cancer", "cardiac development"],
    )

    evidence_quote: str = Field(
        description="Verbatim quote from source text supporting this relationship",
        examples=["MED13 mutations were found in 12% of patients with CHD"],
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score from evaluator (0.0-1.0)",
    )

    source_pmid: str | None = Field(
        default=None,
        description="PubMed ID of source article",
    )


class SearchResults(PydanticBaseModel):
    """
    Complete results from a tree search run.

    This is saved to JSON for later analysis by evaluate_results.py.
    """

    run_id: str = Field(
        description="Unique identifier for this run (timestamp-based)",
    )

    goal: str = Field(
        description="The extraction goal/objective",
    )

    best_triplet: Triplet | None = Field(
        default=None,
        description="Highest-scoring triplet found (if any)",
    )

    all_triplets: list[Triplet] = Field(
        default_factory=list,
        description="All valid triplets extracted during search",
    )

    total_nodes: int = Field(
        default=0,
        description="Total number of search nodes explored",
    )

    total_cost_usd: float = Field(
        default=0.0,
        description="Total API cost in USD",
    )

    total_tokens: int = Field(
        default=0,
        description="Total tokens consumed across all LLM calls",
    )

    invariant_violations: list[dict] = Field(
        default_factory=list,
        description="All invariant violations caught during search",
    )

    search_trace: list[dict] = Field(
        default_factory=list,
        description="Complete search trace for debugging",
    )

    discovered_invariants: list[str] = Field(
        default_factory=list,
        description="Invariants deduced by discovery agent",
    )

    goal_reached: bool = Field(
        default=False,
        description="Whether search reached goal threshold",
    )

    execution_time_s: float = Field(
        default=0.0,
        description="Total execution time in seconds",
    )


class GeneVerificationReport(PydanticBaseModel):
    """
    Verdict from a gene verification agent.

    This is used by the consensus panel to decide whether to proceed.
    """

    is_med13: bool = Field(
        description="True when the text is clearly about MED13",
    )
    is_med13l: bool = Field(
        description="True when the text is about MED13L (paralog)",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the gene identity classification (0.0-1.0)",
    )
    rationale: str = Field(
        default="",
        description="Short justification for the classification",
    )


class GeneExtractionContext(PipelineContext):
    """
    Typed runtime context for the high-assurance extraction pipeline.
    """

    total_abstracts: int = 0
    current_abstract_index: int = 0
    validation_history: list[dict[str, object]] = Field(default_factory=list)
    preprocessing_done: bool = False
    original_length: int = 0
    cleaned_length: int = 0
    verification_reports: list[GeneVerificationReport] = Field(default_factory=list)
    verification_passed: bool = False
    current_payload: dict[str, object] = Field(default_factory=dict)
