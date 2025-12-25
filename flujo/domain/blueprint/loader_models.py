from __future__ import annotations

import re
from typing import Optional, Literal, Union

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

from ...exceptions import ConfigurationError
from .schema import AgentModel
from flujo.type_definitions.common import JSONObject


class BlueprintError(ConfigurationError):
    """Raised when a blueprint cannot be parsed or validated."""


class _CoercionConfig(BaseModel):
    tolerant_level: int = 0
    max_unescape_depth: Optional[int] = None
    anyof_strategy: Optional[str] = None
    allow: Optional[dict[str, list[str]]] = None

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
    validator_agent: object | None = None
    agent: object | None = None
    delimiters: list[str] | None = None
    goal_context_key: Optional[str] = None
    score_threshold: Optional[float] = None
    required_context_keys: list[str] | None = None
    inject_feedback: Optional[str] = None
    retry_guidance_prefix: Optional[str] = None
    context_feedback_key: Optional[str] = None
    consensus_agent: object | None = None
    consensus_samples: Optional[int] = None
    consensus_threshold: Optional[float] = None

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
    structured_output: Optional[str] = None
    aop: Optional[str] = None
    coercion: Optional[_CoercionConfig] = None
    output_schema: Optional[JSONObject] = Field(default=None, alias="schema")
    enforce_grammar: Optional[bool] = None
    reasoning_precheck: Optional[_ReasoningPrecheckConfig] = None

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
        "tree_search",
        "StateMachine",
    ] = Field(default="step")
    # Accept both 'name' and legacy 'step' keys for step name
    name: str = Field(validation_alias=AliasChoices("name", "step"))
    agent: Optional[Union[str, JSONObject]] = None
    uses: Optional[str] = None
    input: object | None = None
    config: JSONObject = Field(default_factory=dict)
    updates_context: bool = False
    validate_fields: bool = False
    branches: Optional[JSONObject] = None
    reduce: Optional[Union[str, JSONObject]] = None
    condition: Optional[str] = None
    condition_expression: Optional[str] = None
    default_branch: object | None = None
    loop: Optional[JSONObject] = None
    map: Optional[JSONObject] = None
    router: Optional[JSONObject] = None
    fallback: Optional[JSONObject] = None
    usage_limits: Optional[JSONObject] = None
    plugins: list[Union[str, JSONObject]] | None = None
    validators: list[str] | None = None
    merge_strategy: Optional[str] = None
    on_branch_failure: Optional[str] = None
    context_include_keys: list[str] | None = None
    field_mapping: Optional[dict[str, list[str]]] = None
    ignore_branch_names: Optional[bool] = None
    message: Optional[str] = None
    input_schema: Optional[JSONObject] = None
    output_schema: Optional[JSONObject] = None
    sink_to: Optional[str] = None
    wrapped_step: Optional[JSONObject] = None
    planner: Optional[str] = None
    registry: Optional[Union[str, JSONObject]] = None
    output_template: Optional[str] = None
    # TreeSearchStep (kind: "tree_search")
    proposer: Optional[Union[str, JSONObject]] = None
    evaluator: Optional[Union[str, JSONObject]] = None
    discovery_agent: Optional[Union[str, JSONObject]] = None
    static_invariants: list[object] | None = None
    cost_function: Optional[Union[str, JSONObject]] = None
    candidate_validator: Optional[Union[str, JSONObject]] = None
    branching_factor: Optional[int] = None
    beam_width: Optional[int] = None
    max_depth: Optional[int] = None
    max_iterations: Optional[int] = None
    path_max_tokens: Optional[int] = None
    goal_score_threshold: Optional[float] = None
    require_goal: Optional[bool] = None
    processing: Optional[ProcessingConfigModel] = None
    meta: Optional[JSONObject] = None
    # StateMachineStep (kind: "StateMachine")
    start_state: Optional[str] = None
    end_states: list[str] | None = None
    states: dict[str, object] | None = None
    transitions: list[object] | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_yaml_11_boolean_keys(cls, data: object) -> object:
        """Normalize YAML 1.1 boolean keys that appear in step specs.

        YAML 1.1 allows unquoted `on`/`off`/`yes`/`no` to parse as booleans. We
        normalize those keys back to their intended string forms in the few
        places where they show up in blueprints.
        """
        if not isinstance(data, dict):
            return data

        kind_val = data.get("kind", "step")
        kind_str = str(kind_val) if kind_val is not None else "step"
        working = dict(data)

        if kind_str == "conditional":
            branches_raw = working.get("branches")
            if isinstance(branches_raw, dict):
                coerced: dict[str, object] = {}
                for k, v in branches_raw.items():
                    if isinstance(k, bool):
                        coerced[str(k).lower()] = v
                    else:
                        coerced[str(k)] = v
                working["branches"] = coerced

        if kind_str == "StateMachine":
            states_raw = working.get("states")
            if isinstance(states_raw, dict):
                working["states"] = {str(k): v for k, v in states_raw.items()}

            end_states_raw = working.get("end_states")
            if end_states_raw is not None and not isinstance(end_states_raw, list):
                working["end_states"] = [end_states_raw]

            transitions_raw = working.get("transitions")
            if isinstance(transitions_raw, list):
                coerced_transitions: list[object] = []
                for rule in transitions_raw:
                    if isinstance(rule, dict):
                        rule_coerced: dict[str, object] = {}
                        for k, v in rule.items():
                            if k is True:
                                rule_coerced["on"] = v
                            elif k is False:
                                rule_coerced["off"] = v
                            else:
                                rule_coerced[str(k)] = v
                        coerced_transitions.append(rule_coerced)
                    else:
                        coerced_transitions.append(rule)
                working["transitions"] = coerced_transitions

        return working

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
    steps: list[JSONObject]
    agents: Optional[dict[str, "AgentModel"]] = None
    imports: Optional[dict[str, str]] = None
    static_invariants: list[object] | None = None

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
