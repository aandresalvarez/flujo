from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

from ...exceptions import ConfigurationError
from .schema import AgentModel


class BlueprintError(ConfigurationError):
    """Raised when a blueprint cannot be parsed or validated."""


class _CoercionConfig(BaseModel):
    tolerant_level: int = 0
    max_unescape_depth: int | None = None
    anyof_strategy: str | None = None
    allow: Dict[str, List[str]] | None = None

    @model_validator(mode="after")
    def _validate_values(self) -> "_CoercionConfig":
        if self.tolerant_level not in (0, 1, 2):
            raise ValueError("coercion.tolerant_level must be 0, 1 or 2")
        if self.max_unescape_depth is not None and self.max_unescape_depth < 0:
            raise ValueError("coercion.max_unescape_depth must be >= 0")
        if self.anyof_strategy is not None and str(self.anyof_strategy).lower() not in {
            "first-pass"
        }:
            raise ValueError("coercion.anyof_strategy must be 'first-pass' if set")
        if self.allow:
            valid = {
                "integer": {"str->int"},
                "number": {"str->float"},
                "boolean": {"str->bool"},
                "array": {"str->array"},
            }
            for k, v in self.allow.items():
                if k not in valid:
                    raise ValueError(f"coercion.allow has invalid key '{k}'")
                for t in v:
                    if t not in valid[k]:
                        raise ValueError(f"coercion.allow[{k}] has invalid transform '{t}'")
        return self


class _ReasoningPrecheckConfig(BaseModel):
    enabled: bool = False
    validator_agent: Any | None = None
    agent: Any | None = None
    delimiters: List[str] | None = None
    goal_context_key: str | None = None
    score_threshold: float | None = None
    required_context_keys: List[str] | None = None
    inject_feedback: str | None = None
    retry_guidance_prefix: str | None = None
    context_feedback_key: str | None = None
    consensus_agent: Any | None = None
    consensus_samples: int | None = None
    consensus_threshold: float | None = None

    @model_validator(mode="after")
    def _validate_values(self) -> "_ReasoningPrecheckConfig":
        if self.delimiters is not None and not (
            isinstance(self.delimiters, list) and len(self.delimiters) >= 2
        ):
            raise ValueError("reasoning_precheck.delimiters must be a list of at least 2 strings")
        if self.inject_feedback is not None and str(self.inject_feedback).lower() not in {
            "prepend",
            "context_key",
        }:
            raise ValueError(
                "reasoning_precheck.inject_feedback must be 'prepend' or 'context_key'"
            )
        if self.consensus_samples is not None and self.consensus_samples < 2:
            raise ValueError("reasoning_precheck.consensus_samples must be >= 2")
        return self


class ProcessingConfigModel(BaseModel):
    structured_output: str | None = None
    aop: str | None = None
    coercion: _CoercionConfig | None = None
    output_schema: Dict[str, Any] | None = Field(default=None, alias="schema")
    enforce_grammar: bool | None = None
    reasoning_precheck: _ReasoningPrecheckConfig | None = None

    model_config = {
        "populate_by_name": True,
        # Avoid pydantic BaseModel attribute collision warnings for 'schema' alias
        "protected_namespaces": (),
    }

    @model_validator(mode="after")
    def _normalize(self) -> "ProcessingConfigModel":
        if self.structured_output is not None:
            val = str(self.structured_output).lower()
            if val not in {"off", "auto", "openai_json", "outlines", "xgrammar"}:
                raise ValueError(
                    "processing.structured_output must be one of off|auto|openai_json|outlines|xgrammar"
                )
            self.structured_output = val
        if self.aop is not None:
            val = str(self.aop).lower()
            if val not in {"off", "minimal", "full"}:
                raise ValueError("processing.aop must be one of off|minimal|full")
            self.aop = val
        return self


class BlueprintStepModel(BaseModel):
    """Declarative step spec (minimal v0)."""

    kind: Literal[
        "step",
        "parallel",
        "conditional",
        "loop",
        "map",
        "dynamic_router",
        "hitl",
        "cache",
        "agentic_loop",
    ] = Field(default="step")
    # Accept both 'name' and legacy 'step' keys for step name
    name: str = Field(validation_alias=AliasChoices("name", "step"))
    agent: Optional[Union[str, Dict[str, Any]]] = None
    uses: Optional[str] = None
    input: Optional[Any] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    updates_context: bool = False
    validate_fields: bool = False
    branches: Optional[Dict[str, Any]] = None
    reduce: Optional[Union[str, Dict[str, Any]]] = None
    condition: Optional[str] = None
    condition_expression: Optional[str] = None
    default_branch: Optional[Any] = None
    loop: Optional[Dict[str, Any]] = None
    map: Optional[Dict[str, Any]] = None
    router: Optional[Dict[str, Any]] = None
    fallback: Optional[Dict[str, Any]] = None
    usage_limits: Optional[Dict[str, Any]] = None
    plugins: Optional[List[Union[str, Dict[str, Any]]]] = None
    validators: Optional[List[str]] = None
    merge_strategy: Optional[str] = None
    on_branch_failure: Optional[str] = None
    context_include_keys: Optional[List[str]] = None
    field_mapping: Optional[Dict[str, List[str]]] = None
    ignore_branch_names: Optional[bool] = None
    message: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None
    sink_to: Optional[str] = None
    wrapped_step: Optional[Dict[str, Any]] = None
    planner: Optional[str] = None
    registry: Optional[Union[str, Dict[str, Any]]] = None
    output_template: Optional[str] = None
    processing: Optional[Dict[str, Any]] = None

    @field_validator("uses")
    @classmethod
    def _validate_uses_format(cls, value: Optional[str]) -> Optional[str]:
        """Validate that 'uses' is either 'agents.<name>' or a Python import path."""
        if value is None:
            return value
        uses_spec = value.strip()
        if uses_spec.startswith("agents."):
            m = re.fullmatch(r"agents\.([A-Za-z_][A-Za-z0-9_]*)", uses_spec)
            if not m:
                raise ValueError("uses must be 'agents.<name>' where <name> is a valid identifier")
            return uses_spec
        if uses_spec.startswith("imports."):
            m = re.fullmatch(r"imports\.([A-Za-z_][A-Za-z0-9_]*)", uses_spec)
            if not m:
                raise ValueError(
                    "uses must be 'imports.<alias>' where <alias> is a valid identifier"
                )
            return uses_spec
        import_path_pattern = re.compile(
            r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*(?::[A-Za-z_][A-Za-z0-9_]*)?$"
        )
        if not import_path_pattern.fullmatch(uses_spec):
            raise ValueError(
                "uses must be 'agents.<name>' or a valid Python import path like 'pkg.mod:attr'"
            )
        return uses_spec


class BlueprintPipelineModel(BaseModel):
    version: str = Field(default="0.1")
    name: Optional[str] = None
    steps: List[Dict[str, Any]]
    agents: Optional[Dict[str, "AgentModel"]] = None
    imports: Optional[Dict[str, str]] = None

    @model_validator(mode="after")
    def _validate_agent_references(self) -> "BlueprintPipelineModel":
        if not self.steps:
            return self
        declared_agents = set((self.agents or {}).keys())
        declared_imports = set((self.imports or {}).keys())
        for idx, step in enumerate(self.steps):
            try:
                uses = None
                if isinstance(step, dict):
                    uses = step.get("uses")
                else:
                    uses = getattr(step, "uses", None)
                if isinstance(uses, str) and uses.startswith("agents."):
                    name = uses.split(".", 1)[1]
                    if name not in declared_agents:
                        raise ValueError(
                            f"Unknown declarative agent referenced at steps[{idx}].uses: {uses}"
                        )
                if isinstance(uses, str) and uses.startswith("imports."):
                    alias = uses.split(".", 1)[1]
                    if alias not in declared_imports:
                        raise ValueError(
                            f"Unknown imported pipeline alias at steps[{idx}].uses: {uses}"
                        )
            except Exception:
                pass
        return self


__all__ = [
    "BlueprintError",
    "BlueprintStepModel",
    "BlueprintPipelineModel",
    "ProcessingConfigModel",
]
