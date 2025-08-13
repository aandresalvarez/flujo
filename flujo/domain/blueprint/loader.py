from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field, ValidationError
import yaml

from ..dsl import (
    Pipeline,
    Step,
    StepConfig,
    ParallelStep,
    MergeStrategy,
    BranchFailureStrategy,
)
from ...exceptions import ConfigurationError


class BlueprintError(ConfigurationError):
	pass


class BlueprintStepModel(BaseModel):
	"""Declarative step spec (minimal v0).

	This intentionally supports only a safe subset to start, then we'll extend.
	"""
	kind: Literal["step", "parallel", "conditional", "loop"] = Field(default="step")
	name: str
	agent: Optional[str] = None
	config: Dict[str, Any] = Field(default_factory=dict)
	# Step flags
	updates_context: bool = False
	validate_fields: bool = False
	# Parallel / Conditional
	branches: Optional[Dict[str, Any]] = None
	# Conditional only (v0: simple string identifier for callable resolution)
	condition: Optional[str] = None
	default_branch: Optional[Any] = None
	# Loop only (v0)
	loop: Optional[Dict[str, Any]] = None  # { body: [...], max_loops: int }
	merge_strategy: Optional[str] = None
	on_branch_failure: Optional[str] = None
	context_include_keys: Optional[List[str]] = None
	field_mapping: Optional[Dict[str, List[str]]] = None
	ignore_branch_names: Optional[bool] = None


class BlueprintPipelineModel(BaseModel):
	version: str = Field(default="0.1")
	steps: List[BlueprintStepModel]


def _normalize_merge_strategy(value: Optional[str]) -> MergeStrategy:
	if value is None:
		return MergeStrategy.CONTEXT_UPDATE
	try:
		return MergeStrategy[value.upper()]
	except Exception as e:
		raise BlueprintError(f"Invalid merge_strategy: {value}") from e


def _normalize_branch_failure(value: Optional[str]) -> BranchFailureStrategy:
	if value is None:
		return BranchFailureStrategy.PROPAGATE
	try:
		return BranchFailureStrategy[value.upper()]
	except Exception as e:
		raise BlueprintError(f"Invalid on_branch_failure: {value}") from e


def _make_step_from_blueprint(model: BlueprintStepModel) -> Step[Any, Any]:
	# For v0: agent may be None; if provided, we resolve later via import string in a follow-up.
	step_config = StepConfig(**model.config) if model.config else StepConfig()
	if model.kind == "parallel":
		if not model.branches:
			raise BlueprintError("parallel step requires branches")
		# Branch values are nested steps or pipelines in YAML; support list-of-steps or single step.
		branches: Dict[str, Pipeline[Any, Any]] = {}
		for branch_name, branch_spec in model.branches.items():
			branches[branch_name] = _build_pipeline_from_branch(branch_spec)
		return ParallelStep(
			name=model.name,
			branches=branches,
			context_include_keys=model.context_include_keys,
			merge_strategy=_normalize_merge_strategy(model.merge_strategy),
			on_branch_failure=_normalize_branch_failure(model.on_branch_failure),
			field_mapping=model.field_mapping,
			ignore_branch_names=bool(model.ignore_branch_names) if model.ignore_branch_names is not None else False,
			config=step_config,
		)
	elif model.kind == "conditional":
		from ..dsl.conditional import ConditionalStep
		if not model.branches:
			raise BlueprintError("conditional step requires branches")
		branches: Dict[Any, Pipeline[Any, Any]] = {}
		for key, branch_spec in model.branches.items():
			branches[key] = _build_pipeline_from_branch(branch_spec)
		# v0: use a simple callable that returns the provided key if input matches string
		def _cond_callable(output: Any, _ctx: Optional[Any]) -> Any:
			# trivial placeholder: pass through output as branch key if present, else use first key
			return output if output in branches else next(iter(branches))
		default_branch = _build_pipeline_from_branch(model.default_branch) if model.default_branch else None
		return ConditionalStep(
			name=model.name,
			condition_callable=_cond_callable,
			branches=branches,
			default_branch_pipeline=default_branch,
			config=step_config,
		)
	elif model.kind == "loop":
		from ..dsl.loop import LoopStep
		if not model.loop or "body" not in model.loop:
			raise BlueprintError("loop step requires loop.body")
		body = _build_pipeline_from_branch(model.loop.get("body"))
		max_loops = model.loop.get("max_loops")
		def _exit_condition(_output: Any, _ctx: Optional[Any], *, _state={"count": 0}) -> bool:  # type: ignore[misc]
			_state["count"] += 1
			if isinstance(max_loops, int) and max_loops > 0:
				return _state["count"] >= max_loops
			return _state["count"] >= 1
		return LoopStep(
			name=model.name,
			loop_body_pipeline=body,
			exit_condition_callable=_exit_condition,  # type: ignore[arg-type]
			max_retries=max(1, int(max_loops)) if isinstance(max_loops, int) else 1,
			config=step_config,
		)
	else:
		# Simple step; resolve agent if provided, otherwise passthrough.
		agent_obj: Any
		if model.agent:
			agent_obj = _resolve_agent(model.agent)
			# If an async callable was provided, use Step.from_callable for richer typing
			if _is_async_callable(agent_obj):
				return Step.from_callable(  # type: ignore[arg-type]
					agent_obj,
					name=model.name,
					updates_context=model.updates_context,
					validate_fields=model.validate_fields,
					**(step_config.model_dump() if hasattr(step_config, "model_dump") else {}),
				)
		else:
			agent_obj = _PassthroughAgent()

		return Step[Any, Any](
			name=model.name,
			agent=agent_obj,
			config=step_config,
			updates_context=model.updates_context,
			validate_fields=model.validate_fields,
		)


def _build_pipeline_from_branch(branch_spec: Any) -> Pipeline[Any, Any]:
	# Accept either a list[BlueprintStepModel-like dicts] or a single dict
	if isinstance(branch_spec, list):
		steps: List[Step[Any, Any]] = []
		for s in branch_spec:
			m = BlueprintStepModel.model_validate(s)
			steps.append(_make_step_from_blueprint(m))
		return Pipeline(steps=steps)  # type: ignore[arg-type]
	elif isinstance(branch_spec, dict):
		m = BlueprintStepModel.model_validate(branch_spec)
		return Pipeline.from_step(_make_step_from_blueprint(m))
	else:
		raise BlueprintError("Invalid branch specification; expected dict or list of dicts")


def build_pipeline_from_blueprint(model: BlueprintPipelineModel) -> Pipeline[Any, Any]:
	steps: List[Step[Any, Any]] = []
	for s in model.steps:
		steps.append(_make_step_from_blueprint(s))
	return Pipeline(steps=steps)  # type: ignore[arg-type]


def dump_pipeline_blueprint_to_yaml(pipeline: Pipeline[Any, Any]) -> str:
	"""Serialize a Pipeline to a minimal YAML blueprint (v0)."""

	def step_to_yaml(step: Any) -> Dict[str, Any]:
		if isinstance(step, ParallelStep):
			branches: Dict[str, Any] = {}
			for k, p in step.branches.items():
				branches[str(k)] = [step_to_yaml(s) for s in p.steps]
			return {
				"kind": "parallel",
				"name": step.name,
				"branches": branches,
				"merge_strategy": getattr(step.merge_strategy, "name", None),
			}
		try:
			from ..dsl.conditional import ConditionalStep
			if isinstance(step, ConditionalStep):
				branches = {str(k): [step_to_yaml(s) for s in p.steps] for k, p in step.branches.items()}
				data: Dict[str, Any] = {"kind": "conditional", "name": step.name, "branches": branches}
				if step.default_branch_pipeline is not None:
					data["default_branch"] = [step_to_yaml(s) for s in step.default_branch_pipeline.steps]
				return data
		except Exception:
			pass
		try:
			from ..dsl.loop import LoopStep
			if isinstance(step, LoopStep):
				return {
					"kind": "loop",
					"name": step.name,
					"loop": {
						"body": [step_to_yaml(s) for s in step.loop_body_pipeline.steps],
						"max_loops": step.max_retries,
					},
				}
		except Exception:
			pass
		return {"kind": "step", "name": getattr(step, "name", "step")}

	data: Dict[str, Any] = {
		"version": "0.1",
		"steps": [step_to_yaml(s) for s in pipeline.steps],
	}
	return yaml.safe_dump(data, sort_keys=False)


def load_pipeline_blueprint_from_yaml(yaml_text: str) -> Pipeline[Any, Any]:
	try:
		data = yaml.safe_load(yaml_text)
		if not isinstance(data, dict) or "steps" not in data:
			raise BlueprintError("YAML blueprint must be a mapping with a 'steps' key")
		bp = BlueprintPipelineModel.model_validate(data)
		return build_pipeline_from_blueprint(bp)
	except ValidationError as ve:
		raise BlueprintError(str(ve)) from ve
	except yaml.YAMLError as ye:
		raise BlueprintError(f"Invalid YAML: {ye}") from ye


# ----------------------------
# Resolution helpers (v1)
# ----------------------------

def _import_object(path: str) -> Any:
    """Import an object from 'module:attr' or 'module.attr' path."""
    import importlib

    module_name: str
    attr_name: Optional[str] = None
    if ":" in path:
        module_name, attr_name = path.split(":", 1)
    else:
        parts = path.split(".")
        if len(parts) < 2:
            # just a module; import it
            return importlib.import_module(path)
        module_name, attr_name = ".".join(parts[:-1]), parts[-1]
    module = importlib.import_module(module_name)
    return getattr(module, attr_name) if attr_name else module


def _is_async_callable(obj: Any) -> bool:
    try:
        import inspect

        return inspect.iscoroutinefunction(obj)
    except Exception:
        return False


class _PassthroughAgent:
    async def run(self, x: Any, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - trivial
        return x


def _resolve_agent(agent_spec: str) -> Any:
    obj = _import_object(agent_spec)
    # If it's a class, try to instantiate with no args
    try:
        import inspect

        if inspect.isclass(obj):
            return obj()
        return obj
    except Exception:
        return obj
