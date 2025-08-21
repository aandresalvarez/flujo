from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field

from flujo.domain.models import PipelineContext
from .models import ExecutionPlan, ToolSelection, GeneratedYaml


class ArchitectContext(PipelineContext):
    # Legacy/compat input used by some tests and utilities
    initial_prompt: Optional[str] = None
    # Inputs
    user_goal: Optional[str] = None
    project_summary: Optional[str] = None
    refinement_feedback: Optional[str] = None

    # Discovered Capabilities
    flujo_schema: Dict[str, Any] = Field(default_factory=dict)
    available_skills: List[Dict[str, Any]] = Field(default_factory=list)

    # Intermediate Plan
    execution_plan: Optional[List[Dict[str, Any]]] = None
    plan_summary: Optional[str] = None
    plan_mermaid_graph: Optional[str] = None
    plan_estimates: Dict[str, float] = Field(default_factory=dict)
    # Agentic structured variant (Phase 1)
    execution_plan_structured: Optional[ExecutionPlan] = None

    # User Interaction State
    plan_approved: bool = False
    dry_run_requested: bool = False
    sample_input: Optional[str] = None

    # HITL Configuration
    hitl_enabled: bool = False
    non_interactive: bool = True

    # Final Artifact
    generated_yaml: Optional[str] = None
    yaml_text: Optional[str] = None
    validation_report: Optional[Dict[str, Any]] = None
    yaml_is_valid: bool = False
    validation_errors: Optional[str] = None
    # Agentic structured variants (Phase 2)
    tool_selections: List[ToolSelection] = Field(default_factory=list)
    generated_yaml_structured: Optional[GeneratedYaml] = None

    # Pipeline helpers used by existing CLI/tests
    prepared_steps_for_mapping: List[Dict[str, Any]] = Field(default_factory=list)
