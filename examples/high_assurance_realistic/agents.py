"""
Enhanced agents with proper Flujo DSL syntax.

This module demonstrates the COMPLETE Flujo DSL including:
- @step decorator for custom steps
- Pipeline composition with >> operator
- Step.from_callable for function wrapping
- ParallelStep consensus guard
- Proper type hints and context injection
"""

from __future__ import annotations

from flujo import make_agent_async, Step, Pipeline
from flujo.domain.agent_result import FlujoAgentResult
from flujo.domain.dsl import adapter_step, step
from flujo.domain.dsl import MergeStrategy
from flujo.domain.evaluation import EvaluationReport
from flujo.exceptions import PipelineAbortSignal

from .models import GeneExtractionContext, GeneVerificationReport, Triplet

ADAPTER_ID = "generic-adapter"
ADAPTER_ALLOW = "generic"


# ============================================================================
# AGENT DEFINITIONS (using make_agent_async)
# ============================================================================

proposer_agent = make_agent_async(
    model="openai:gpt-4o",
    system_prompt="""
You are a biomedical knowledge extraction expert specializing in gene-disease relationships.

Your task is to extract structured triplets from biomedical text. Each triplet represents
a relationship between a gene and a disease/phenotype.

CRITICAL RULES:
1. ONLY extract facts about the MED13 gene
2. DO NOT extract facts about MED13L (this is a DIFFERENT gene - a paralog)
3. DO NOT extract facts about other genes (MED12, MED14, etc.)
4. Every triplet MUST include a verbatim quote as evidence
5. Be precise: distinguish between "causes", "associated_with", "regulates", etc.

Return a JSON array of triplets. If no valid MED13 facts are found, return an empty array.
""",
    output_type=list[Triplet],
)


evaluator_agent = make_agent_async(
    model="anthropic:claude-sonnet-4-5-20250929",
    system_prompt="""
You are a biomedical fact-checking expert. Your job is to evaluate the quality
of gene-disease relationship triplets extracted from scientific literature.

Score triplets on a scale of 0.0 to 1.0 based on:
1. Evidence Strength (40%): Causal > Association > Mention
2. Specificity (30%): Precise > General > Vague
3. Correctness (30%): Definitely MED13 > Ambiguous > Wrong gene

HARD FAIL if triplet is about MED13L or another gene.

Return EvaluationReport with score, hard_fail flag, and metadata.
""",
    output_type=EvaluationReport,
)


discovery_agent = make_agent_async(
    model="openai:gpt-4o",
    system_prompt="""
You are a formal methods expert specializing in constraint deduction.

Analyze the biomedical extraction goal and deduce HARD INVARIANTS that must
ALWAYS be true during extraction.

Only propose invariants that are directly about:
- MED13 vs MED13L gene identity (usually in `output.subject`)
- Presence of evidence quotes (`output.evidence_quote`)
- Output structure (subject/relation/object)

Do NOT invent environment, lab, or personnel rules. Every rule MUST mention `output`
(no context-only rules). Use attribute syntax (output.subject). Never require MED13L
presence; only forbid it (e.g., "'MED13L' not in output.subject.upper()").

ALLOWED CONSTRUCTS:
- Comparisons: ==, !=, <, <=, >, >=
- Boolean ops: and, or, not
- Membership: in, not in
- String methods: .strip(), .lower(), .upper(), .startswith(), .endswith()
- Constants: True, False, None, strings, numbers

FORBIDDEN CONSTRUCTS (Rules using these will be DISCARDED):
- No `isinstance()`
- No `len()`
- No `str()`
- No `type()`
- No `getattr()` / `hasattr()`
- No `implies` (use `or` with `not`)

Guard rules so they pass on raw text inputs (e.g., \"output.subject == None or ...\").

CRITICAL: Do NOT assume 'MED13' must be in the 'object' field. The 'object' is usually 
the disease name.

Return a JSON array of Python expression strings (1-3 invariants).
""",
    output_type=list[str],
)


verification_gpt_agent = make_agent_async(
    model="openai:gpt-4o",
    system_prompt="""
You are verifying gene identity for a high-assurance pipeline.

Given a biomedical abstract, decide whether it is clearly about MED13
and whether it is about MED13L (the paralog).

Rules:
1) If MED13L is mentioned, set is_med13l=True (even if MED13 appears).
2) If MED13 is clearly the focus, set is_med13=True.
3) If uncertain, set is_med13=False and keep confidence low.

Return a GeneVerificationReport with is_med13, is_med13l, confidence, and rationale.
""",
    output_type=GeneVerificationReport,
)


verification_claude_agent = make_agent_async(
    model="anthropic:claude-sonnet-4-5-20250929",
    system_prompt="""
You are verifying gene identity for a high-assurance pipeline.

Given a biomedical abstract, decide whether it is clearly about MED13
and whether it is about MED13L (the paralog).

Rules:
1) If MED13L is mentioned, set is_med13l=True (even if MED13 appears).
2) If MED13 is clearly the focus, set is_med13=True.
3) If uncertain, set is_med13=False and keep confidence low.

Return a GeneVerificationReport with is_med13, is_med13l, confidence, and rationale.
""",
    output_type=GeneVerificationReport,
)


# ============================================================================
# CUSTOM STEPS (using @step decorator)
# ============================================================================


@step(name="load_abstracts")
async def load_abstracts_step(
    goal: str,
    *,
    context: GeneExtractionContext,
) -> dict[str, object]:
    """
    Load PubMed abstracts for extraction.

    This demonstrates the @step decorator for custom processing steps.

    Args:
        goal: Extraction goal
        context: Pipeline context (injected automatically)

    Returns:
        Dictionary with abstracts and metadata
    """
    from .sample_data import get_all_abstracts, format_abstract_for_extraction

    abstracts = get_all_abstracts()

    # Store in context for later steps
    context.total_abstracts = len(abstracts)
    context.current_abstract_index = 0

    # Format first abstract
    formatted_text = format_abstract_for_extraction(abstracts[0])

    payload = {
        "text": formatted_text,
        "pmid": abstracts[0]["pmid"],
        "total_abstracts": len(abstracts),
    }
    context.current_payload = payload
    return payload


@step(name="validate_triplet")
async def validate_triplet_step(
    triplet: Triplet,
    *,
    context: GeneExtractionContext,
) -> dict:
    """
    Validate a triplet against invariants.

    This demonstrates:
    - @step decorator with type hints
    - Context injection
    - Custom validation logic

    Args:
        triplet: Triplet to validate
        context: Pipeline context

    Returns:
        Validation result with is_valid flag
    """
    from .invariants import validate_output_manually

    is_valid, violations = validate_output_manually(triplet)

    # Store validation history in context
    context.validation_history.append(
        {
            "triplet": triplet.model_dump(mode="json"),
            "is_valid": is_valid,
            "violations": violations,
        }
    )

    return {
        "is_valid": is_valid,
        "violations": violations,
        "triplet": triplet,
    }


@adapter_step(
    name="format_results",
    adapter_id=ADAPTER_ID,
    adapter_allow=ADAPTER_ALLOW,
)
async def format_results_step(
    search_output: Triplet,
    *,
    context: GeneExtractionContext,
) -> dict:
    """
    Format search results for output.

    This demonstrates post-processing after tree search.

    Args:
        search_output: Output from TreeSearchStep
        context: Pipeline context

    Returns:
        Formatted results dictionary
    """
    from .models import Triplet

    # Convert to Triplet if needed
    if isinstance(search_output, dict):
        try:
            triplet = Triplet(**search_output)
        except Exception:
            triplet = None
    elif isinstance(search_output, Triplet):
        triplet = search_output
    else:
        triplet = None

    # Extract metadata from context
    total_abstracts = getattr(context, "total_abstracts", 0)
    validation_history = getattr(context, "validation_history", [])

    return {
        "best_triplet": triplet.model_dump() if triplet else None,
        "total_abstracts_processed": total_abstracts,
        "total_validations": len(validation_history),
        "search_metadata": {
            "goal_reached": triplet is not None,
            "confidence": triplet.confidence if triplet else 0.0,
        },
    }


def _normalize_verification_report(raw: object) -> GeneVerificationReport:
    """Normalize branch output into a GeneVerificationReport."""
    if isinstance(raw, GeneVerificationReport):
        return raw
    if isinstance(raw, dict):
        try:
            return GeneVerificationReport(**raw)
        except Exception:
            pass
    return GeneVerificationReport(
        is_med13=False,
        is_med13l=False,
        confidence=0.0,
        rationale=str(raw),
    )


def _normalize_invariants(raw: object) -> list[str]:
    """Normalize discovery output into a list of invariant expressions."""
    import json

    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, dict):
        for key in ("invariants", "rules", "constraints"):
            if key in raw:
                return _normalize_invariants(raw.get(key))
        try:
            return [json.dumps(raw, ensure_ascii=True, sort_keys=True)]
        except Exception:
            return [str(raw)]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            return _normalize_invariants(parsed)
        except Exception:
            lines = [line.strip(" -*\t") for line in text.splitlines() if line.strip()]
            return lines or [text]
    return [str(raw).strip()]


def _requires_med13l_presence(rule_text: str) -> bool:
    """Return True when a rule asserts MED13L presence (positive constraint)."""
    import ast

    token = "MED13L"

    def _has_med13l(node: ast.AST | None) -> bool:
        if node is None:
            return False
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return token in node.value.upper()
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return any(_has_med13l(elt) for elt in node.elts)
        return False

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.positive = False
            self._negated = 0

        def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
            if isinstance(node.op, ast.Not):
                self._negated += 1
                self.visit(node.operand)
                self._negated -= 1
            else:
                self.generic_visit(node)

        def visit_Compare(self, node: ast.Compare) -> None:
            if self.positive:
                return
            has_token = _has_med13l(node.left) or any(
                _has_med13l(comp) for comp in node.comparators
            )
            if has_token:
                for op in node.ops:
                    positive = isinstance(op, (ast.In, ast.Eq))
                    negative = isinstance(op, (ast.NotIn, ast.NotEq))
                    if positive and self._negated % 2 == 0:
                        self.positive = True
                        return
                    if negative and self._negated % 2 == 1:
                        self.positive = True
                        return
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if self.positive:
                return
            if isinstance(node.func, ast.Attribute):
                method = node.func.attr
                if method in ("startswith", "endswith"):
                    if any(_has_med13l(arg) for arg in node.args):
                        if self._negated % 2 == 0:
                            self.positive = True
                            return
            self.generic_visit(node)

    try:
        parsed = ast.parse(rule_text, mode="eval")
    except Exception:
        return False
    visitor = _Visitor()
    visitor.visit(parsed)
    return visitor.positive


def _sanitize_invariants(rules: list[str]) -> list[str]:
    """Guard invariants so they don't fail on raw string nodes."""
    import re
    from flujo.utils.expressions import compile_expression_to_callable

    cleaned: list[str] = []
    for rule in rules:
        rule_text = rule.strip()
        if not rule_text:
            continue

        # Strip forbidden calls immediately
        if any(
            x in rule_text for x in ("isinstance", "len(", "str(", "type(", "getattr", "hasattr")
        ):
            continue

        # Simple replacement for 'implies' if the LLM uses it
        if " implies " in rule_text:
            parts = rule_text.split(" implies ")
            rule_text = f"not ({parts[0]}) or ({parts[1]})"

        # Normalize dict access to attribute access for readability.
        rule_text = re.sub(
            r"(output|context)\[['\"]([A-Za-z_]\w*)['\"]\]",
            r"\1.\2",
            rule_text,
        )
        if "output" not in rule_text:
            continue
        if "output." in rule_text or "output[" in rule_text:
            guard = "output.subject == None"
            if guard not in rule_text:
                rule_text = f"{guard} or ({rule_text})"
        try:
            compile_expression_to_callable(rule_text)
        except Exception:
            continue
        if _requires_med13l_presence(rule_text):
            continue
        cleaned.append(rule_text)
    return cleaned


@step(name="deduce_invariants")
async def deduce_invariants_step(
    objective: str,
    *,
    context: GeneExtractionContext,
) -> list[str]:
    """Run discovery agent and sanitize invariants for tree search."""
    result = await discovery_agent.run(objective, context=context)
    if isinstance(result, FlujoAgentResult):
        result = result.output
    return _sanitize_invariants(_normalize_invariants(result))


@step(name="gpt_verify_gene")
async def gpt_verify_gene_step(
    text: str,
    *,
    context: GeneExtractionContext,
) -> dict:
    result = await verification_gpt_agent.run(text, context=context)
    if isinstance(result, FlujoAgentResult):
        result = result.output
    # Normalize to Pydantic model first
    report = _normalize_verification_report(result)
    # Return JSON-safe dict
    return report.model_dump(mode="json")


@step(name="claude_verify_gene")
async def claude_verify_gene_step(
    text: str,
    *,
    context: GeneExtractionContext,
) -> dict:
    result = await verification_claude_agent.run(text, context=context)
    if isinstance(result, FlujoAgentResult):
        result = result.output
    # Normalize to Pydantic model first
    report = _normalize_verification_report(result)
    # Return JSON-safe dict
    return report.model_dump(mode="json")


def _build_verification_branch(name: str, verify_step: Step) -> Pipeline:
    async def select_text(
        payload: dict[str, object],
        *,
        context: GeneExtractionContext,
    ) -> str:
        text = payload.get("text", "")
        if not isinstance(text, str):
            text = str(text)
        return text

    select_step = Step.from_callable(
        select_text,
        name=f"{name}_select_text",
        updates_context=False,
    )
    return select_step >> verify_step


def create_verification_panel() -> Step:
    """Create a parallel verification panel to confirm gene identity."""
    panel = Step.parallel(
        name="gene_verification",
        branches={
            "gpt": _build_verification_branch("gpt", gpt_verify_gene_step),
            "claude": _build_verification_branch("claude", claude_verify_gene_step),
        },
        merge_strategy=MergeStrategy.NO_MERGE,
    )
    panel.__step_input_type__ = dict
    panel.__step_output_type__ = dict
    return panel


def create_consensus_gate_step() -> Step:
    """Create a consensus gate that blocks MED13L contamination early."""

    @adapter_step(
        name="gene_consensus_gate",
        adapter_id=ADAPTER_ID,
        adapter_allow=ADAPTER_ALLOW,
    )
    async def consensus_gate(
        verification_results: dict[str, object],
        *,
        context: GeneExtractionContext,
    ) -> dict[str, object]:
        reports = [_normalize_verification_report(raw) for raw in verification_results.values()]
        context.verification_reports = reports

        if not reports:
            context.verification_passed = False
            context.pause_message = "Gene verification produced no reports."
            raise PipelineAbortSignal("Gene verification produced no reports.")

        any_med13l = any(report.is_med13l for report in reports)
        all_med13 = all(report.is_med13 for report in reports)

        if any_med13l or not all_med13:
            context.verification_passed = False
            context.pause_message = "Consensus gate rejected: text is not clean MED13."
            raise PipelineAbortSignal("Consensus gate rejected: text is not clean MED13.")

        payload = context.current_payload
        if not payload:
            outputs = getattr(context, "step_outputs", None)
            if isinstance(outputs, dict):
                payload = outputs.get("load_abstracts", {}) or {}
        if not isinstance(payload, dict) or not payload:
            context.pause_message = "Consensus gate could not locate extraction payload."
            raise PipelineAbortSignal("Consensus gate could not locate extraction payload.")

        context.current_payload = payload
        context.verification_passed = True
        return payload

    consensus_gate.__step_input_type__ = dict
    consensus_gate.__step_output_type__ = dict
    return consensus_gate


@step(name="cleanup_context")
async def cleanup_context_step(
    results: dict[str, object],
    *,
    context: GeneExtractionContext,
) -> dict[str, object]:
    """Remove large search state to keep CLI context serialization safe."""
    if hasattr(context, "tree_search_state"):
        try:
            context.tree_search_state = None
        except Exception:
            pass
    return results


# ============================================================================
# PIPELINE COMPOSITION (using >> operator)
# ============================================================================


def create_extraction_pipeline(
    *,
    max_depth: int = 4,
    beam_width: int = 3,
    max_iterations: int = 30,
    use_discovery: bool = True,
    use_consensus: bool = True,
) -> Pipeline:
    """
    Create a complete extraction pipeline using Flujo DSL.

    This demonstrates:
    - Pipeline composition with >> operator
    - Mixing custom steps with TreeSearchStep
    - Step chaining

    Returns:
        Complete extraction pipeline
    """
    from flujo.domain.dsl.tree_search import TreeSearchStep
    from .invariants import STRICT_INVARIANTS

    # Step 1: Load abstracts
    load_step = load_abstracts_step

    async def select_text(
        payload: dict[str, object],
        *,
        context: GeneExtractionContext,
    ) -> str:
        text = payload.get("text", "")
        if not isinstance(text, str):
            text = str(text)
        return text

    text_step = Step.from_callable(
        select_text,
        name="select_text",
        updates_context=False,
    )

    # Step 2: Tree search extraction
    search_step = TreeSearchStep(
        name="med13_extraction",
        proposer=proposer_agent,
        evaluator=evaluator_agent,
        discovery_agent=deduce_invariants_step if use_discovery else None,
        static_invariants=STRICT_INVARIANTS,
        branching_factor=3,
        beam_width=beam_width,
        max_depth=max_depth,
        max_iterations=max_iterations,
        goal_score_threshold=0.9,
        require_goal=False,
    )
    search_step.__step_input_type__ = str
    search_step.__step_output_type__ = Triplet

    # Step 3: Format results
    format_step = format_results_step

    # Compose pipeline with >> operator
    pipeline = load_step
    if use_consensus:
        pipeline = pipeline >> create_verification_panel() >> create_consensus_gate_step()
    pipeline = pipeline >> text_step >> search_step >> format_step >> cleanup_context_step

    return pipeline


# ============================================================================
# ALTERNATIVE: FUNCTIONAL STEPS (using Step.from_callable)
# ============================================================================


def create_preprocessing_step() -> Step:
    """
    Create a preprocessing step using Step.from_callable.

    This demonstrates an alternative to @step decorator.
    """

    async def preprocess_text(
        payload: dict[str, object],
        *,
        context: GeneExtractionContext,
    ) -> str:
        """Clean and prepare text for extraction."""
        text = payload.get("text", "")
        if not isinstance(text, str):
            text = str(text)

        # Remove extra whitespace
        cleaned = " ".join(text.split())

        # Store preprocessing metadata
        context.preprocessing_done = True
        context.original_length = len(text)
        context.cleaned_length = len(cleaned)

        return cleaned

    return Step.from_callable(
        preprocess_text,
        name="preprocess_text",
        updates_context=False,
    )


def create_postprocessing_step() -> Step:
    """
    Create a postprocessing step for final cleanup.
    """

    async def postprocess_results(
        results: dict,
        *,
        context: GeneExtractionContext,
    ) -> dict:
        """Add final metadata and cleanup."""
        # Add execution metadata
        results["execution_metadata"] = {
            "preprocessing_done": getattr(context, "preprocessing_done", False),
            "original_length": getattr(context, "original_length", 0),
            "cleaned_length": getattr(context, "cleaned_length", 0),
        }

        return results

    return Step.from_callable(
        postprocess_results,
        name="postprocess_results",
        updates_context=False,
    )


# ============================================================================
# COMPLETE PIPELINE WITH ALL FEATURES
# ============================================================================


def create_full_pipeline(
    *,
    max_depth: int = 4,
    beam_width: int = 3,
    max_iterations: int = 30,
    use_discovery: bool = True,
    use_consensus: bool = True,
) -> Pipeline:
    """
    Create the complete pipeline demonstrating ALL Flujo DSL features.

    This shows:
    - @step decorated functions
    - Step.from_callable
    - TreeSearchStep
    - Pipeline >> composition
    - Context injection
    - Type hints

    Returns:
        Complete production-ready pipeline
    """
    from flujo.domain.dsl.tree_search import TreeSearchStep
    from .invariants import STRICT_INVARIANTS

    # Build pipeline step by step
    pipeline = load_abstracts_step
    if use_consensus:
        pipeline = pipeline >> create_verification_panel() >> create_consensus_gate_step()
    search_step = TreeSearchStep(
        name="med13_extraction",
        proposer=proposer_agent,
        evaluator=evaluator_agent,
        discovery_agent=deduce_invariants_step if use_discovery else None,
        static_invariants=STRICT_INVARIANTS,
        branching_factor=3,
        beam_width=beam_width,
        max_depth=max_depth,
        max_iterations=max_iterations,
        goal_score_threshold=0.9,
        require_goal=False,
    )
    search_step.__step_input_type__ = str
    search_step.__step_output_type__ = Triplet

    pipeline = (
        pipeline
        >> create_preprocessing_step()  # Step.from_callable
        >> search_step  # Complex step
        >> format_results_step  # @step decorator
        >> create_postprocessing_step()  # Step.from_callable
        >> cleanup_context_step  # @step decorator
    )

    return pipeline


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_all_agents() -> dict[str, object]:
    """Get all agent instances for inspection or testing."""
    return {
        "proposer": proposer_agent,
        "evaluator": evaluator_agent,
        "discovery": discovery_agent,
        "verification_gpt": verification_gpt_agent,
        "verification_claude": verification_claude_agent,
    }


def get_all_steps() -> dict[str, Step]:
    """Get all custom steps for inspection or testing."""
    return {
        "load_abstracts": load_abstracts_step,
        "validate_triplet": validate_triplet_step,
        "format_results": format_results_step,
        "gpt_verify_gene": gpt_verify_gene_step,
        "claude_verify_gene": claude_verify_gene_step,
        "gene_verification": create_verification_panel(),
        "gene_consensus_gate": create_consensus_gate_step(),
        "deduce_invariants": deduce_invariants_step,
        "preprocess": create_preprocessing_step(),
        "postprocess": create_postprocessing_step(),
        "cleanup_context": cleanup_context_step,
    }


def verify_api_keys() -> tuple[bool, list[str]]:
    """Verify that required API keys are set."""
    from flujo.infra.config_manager import get_config_manager

    settings = get_config_manager().get_settings()
    missing = []

    if not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if not settings.anthropic_api_key:
        missing.append("ANTHROPIC_API_KEY")

    return len(missing) == 0, missing
