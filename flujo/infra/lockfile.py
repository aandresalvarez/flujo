from __future__ import annotations

import glob
import inspect
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, TypeVar, cast

from pydantic import BaseModel as PydanticBaseModel, Field

from ..domain.dsl.conditional import ConditionalStep
from ..domain.dsl.dynamic_router import DynamicParallelRouterStep
from ..domain.dsl.loop import LoopStep
from ..domain.dsl.parallel import ParallelStep
from ..domain.dsl.pipeline import Pipeline
from ..domain.dsl.tree_search import TreeSearchStep
from ..domain.models import BaseModel, PipelineResult
from ..type_definitions.common import JSONObject
from ..utils.hash import stable_digest


def _iter_steps(obj: object) -> Iterable[object]:
    if isinstance(obj, Pipeline):
        for step in obj.steps:
            yield from _iter_steps(step)
        return
    if isinstance(obj, ParallelStep):
        yield obj
        for branch in obj.branches.values():
            yield from _iter_steps(branch)
        return
    if isinstance(obj, ConditionalStep):
        yield obj
        for branch in obj.branches.values():
            yield from _iter_steps(branch)
        if obj.default_branch_pipeline is not None:
            yield from _iter_steps(obj.default_branch_pipeline)
        return
    if isinstance(obj, DynamicParallelRouterStep):
        yield obj
        for branch in obj.branches.values():
            yield from _iter_steps(branch)
        return
    if isinstance(obj, LoopStep):
        yield obj
        body = getattr(obj, "loop_body_pipeline", None)
        if body is not None:
            yield from _iter_steps(body)
        return
    yield obj


PipelineInT = TypeVar("PipelineInT")
PipelineOutT = TypeVar("PipelineOutT")
ContextT = TypeVar("ContextT", bound=BaseModel)


def _iter_agents(pipeline: Pipeline[PipelineInT, PipelineOutT]) -> Iterable[Tuple[str, object]]:
    for step in _iter_steps(pipeline):
        if isinstance(step, TreeSearchStep):
            yield (f"{step.name}.proposer", step.proposer)
            yield (f"{step.name}.evaluator", step.evaluator)
            continue
        if isinstance(step, DynamicParallelRouterStep):
            yield (f"{step.name}.router", step.router_agent)
        agent = getattr(step, "agent", None)
        if agent is not None:
            yield (getattr(step, "name", "<unnamed>"), agent)


def _unwrap_agent(agent: object) -> object:
    return getattr(agent, "_agent", agent)


def _extract_skill_id(agent: object) -> str | None:
    if isinstance(agent, dict):
        skill_id = agent.get("id") or agent.get("path")
        return str(skill_id) if skill_id else None
    if isinstance(agent, str):
        return agent
    skill_id = getattr(agent, "__flujo_skill_id__", None)
    if isinstance(skill_id, str) and skill_id:
        return skill_id
    unwrapped = _unwrap_agent(agent)
    skill_id = getattr(unwrapped, "__flujo_skill_id__", None)
    if isinstance(skill_id, str) and skill_id:
        return skill_id
    return None


def _extract_prompt(agent: object) -> str | None:
    for attr in ("_original_system_prompt", "system_prompt_template", "system_prompt"):
        val = getattr(agent, attr, None)
        if isinstance(val, str) and val:
            return val
    unwrapped = _unwrap_agent(agent)
    for attr in ("_original_system_prompt", "system_prompt_template", "system_prompt"):
        val = getattr(unwrapped, attr, None)
        if isinstance(val, str) and val:
            return val
    return None


def _source_fingerprint(agent: object) -> str | None:
    target = agent
    if hasattr(agent, "__call__") and not inspect.isfunction(agent):
        target = agent.__call__
    try:
        # inspect.getsource expects a function-like object; some agent wrappers expose
        # __call__ without matching the FunctionType protocols, so we cast to Any and
        # rely on runtime inspection plus the exception handler below.
        return inspect.getsource(cast(Any, target))
    except Exception:
        return None


def _hash_skill(agent: object, skill_id: str) -> str:
    payload = {
        "id": skill_id,
        "module": getattr(agent, "__module__", None),
        "qualname": getattr(agent, "__qualname__", None),
        "source": _source_fingerprint(agent),
    }
    return stable_digest(payload)


def _normalize_payload(value: object) -> object:
    if isinstance(value, BaseModel):
        try:
            return value.model_dump(mode="json")
        except Exception:
            return value.model_dump()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, (list, tuple)):
        return [_normalize_payload(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalize_payload(v) for k, v in value.items()}
    return value


def hash_external_files(
    file_paths: List[Path],
    project_root: Path,
    *,
    strict: bool = False,
) -> List[JSONObject]:
    """Hash external files for inclusion in lockfile.

    Args:
        file_paths: List of file paths to hash (supports glob patterns)
        project_root: Project root directory for resolving relative paths
        strict: If True, raise error on missing files; if False, warn and skip

    Returns:
        List of file hash objects with path, hash, and size

    Raises:
        FileNotFoundError: If strict=True and a file is missing
    """
    results: List[JSONObject] = []
    resolved_paths: List[Path] = []

    # Expand glob patterns and resolve paths
    for pattern in file_paths:
        pattern_path: Path = Path(pattern)
        if not pattern_path.is_absolute():
            pattern_path = project_root / pattern_path

        pattern_str: str = str(pattern_path)
        # Check if pattern contains glob characters
        if "*" in pattern_str or "?" in pattern_str or "[" in pattern_str:
            # Glob pattern - handle recursive patterns with **
            if "**" in pattern_str:
                # For recursive patterns, use glob.glob with recursive=True
                matches: List[Path] = [Path(p) for p in glob.glob(pattern_str, recursive=True)]
                resolved_paths.extend(matches)
            else:
                # Non-recursive glob pattern
                matches = list(pattern_path.parent.glob(pattern_path.name))
                resolved_paths.extend(matches)
        else:
            # Regular path
            resolved_paths.append(pattern_path)

    # Remove duplicates while preserving order
    seen: set[Path] = set()
    unique_paths: List[Path] = []
    for path in resolved_paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)

    # Hash each file
    for file_path in unique_paths:
        try:
            if not file_path.exists():
                if strict:
                    raise FileNotFoundError(f"External file not found: {file_path}")
                continue

            if not file_path.is_file():
                continue

            # Read file content and hash
            file_content: bytes = file_path.read_bytes()
            file_hash: str = stable_digest(file_content)
            file_size: int = file_path.stat().st_size

            # Store relative path
            try:
                relative_path: str = str(file_path.relative_to(project_root))
            except ValueError:
                # Path is outside project root, use absolute path
                relative_path = str(file_path)

            results.append(
                {
                    "path": relative_path,
                    "hash": file_hash,
                    "size": file_size,
                }
            )
        except Exception:
            if strict:
                raise
            # Skip files that can't be read (permissions, etc.)

    return results


def extract_model_info(
    agent: object,
    step_name: str,
) -> Optional[JSONObject]:
    """Extract model information from an agent.

    Args:
        agent: The agent object to extract model info from
        step_name: Name of the step for logging

    Returns:
        JSONObject with model_id, hyperparameters, and hash, or None if no model info found
    """
    try:
        from ..utils.model_utils import extract_model_id

        model_id: Optional[str] = extract_model_id(agent, step_name)
        if model_id is None:
            return None

        # Extract hyperparameters from agent
        hyperparameters: JSONObject = {}

        # Try to get from AsyncAgentWrapper
        unwrapped: object = _unwrap_agent(agent)
        if hasattr(unwrapped, "_agent"):
            inner_agent: object = getattr(unwrapped, "_agent")
            # Try to get model_settings from pydantic-ai Agent
            if hasattr(inner_agent, "model_settings"):
                model_settings: object = getattr(inner_agent, "model_settings")
                if model_settings is not None:
                    if isinstance(model_settings, dict):
                        hyperparameters.update(model_settings)
                    elif hasattr(model_settings, "model_dump"):
                        hyperparameters.update(model_settings.model_dump())

        # Try to get from step config (temperature, max_tokens, etc.)
        # This would require access to the step, which we don't have here
        # So we'll focus on what we can extract from the agent itself

        # Build model config hash
        model_config: JSONObject = {
            "model_id": model_id,
            "hyperparameters": hyperparameters,
        }
        config_hash: str = stable_digest(model_config)

        result: JSONObject = {
            "step": step_name,
            "model_id": model_id,
            "hyperparameters": hyperparameters,
            "hash": config_hash,
        }
        return result
    except Exception:
        # Gracefully handle any extraction errors
        return None


def build_lockfile_data(
    *,
    pipeline: Pipeline[PipelineInT, PipelineOutT],
    result: PipelineResult[ContextT],
    pipeline_name: str | None,
    pipeline_version: str,
    pipeline_id: str,
    run_id: str | None,
    external_files: Optional[List[Path]] = None,
    include_model_info: bool = False,
    project_root: Optional[Path] = None,
) -> JSONObject:
    """Build lockfile data from pipeline and result.

    Args:
        pipeline: The pipeline to generate lockfile for
        result: Pipeline execution result
        pipeline_name: Name of the pipeline
        pipeline_version: Version of the pipeline
        pipeline_id: Unique identifier for the pipeline
        run_id: Optional run ID
        external_files: Optional list of external files to hash
        include_model_info: Whether to include model information
        project_root: Project root for resolving relative paths (required if external_files provided)

    Returns:
        JSONObject containing lockfile data
    """
    skills: list[JSONObject] = []
    prompts: list[JSONObject] = []
    models: list[JSONObject] = []
    external_files_data: list[JSONObject] = []

    # Extract skills and prompts
    for step_name, agent in _iter_agents(pipeline):
        skill_id = _extract_skill_id(agent)
        if skill_id:
            skills.append(
                {
                    "step": step_name,
                    "skill_id": skill_id,
                    "hash": _hash_skill(_unwrap_agent(agent), skill_id),
                }
            )
        prompt = _extract_prompt(agent)
        if prompt:
            prompts.append(
                {
                    "step": step_name,
                    "hash": stable_digest(prompt),
                }
            )

        # Extract model info if requested
        if include_model_info:
            model_info = extract_model_info(agent, step_name)
            if model_info is not None:
                models.append(model_info)

    # Hash external files if provided
    if external_files is not None:
        if project_root is None:
            raise ValueError("project_root is required when external_files is provided")
        external_files_data = hash_external_files(external_files, project_root)

    # Determine schema version based on whether we have new fields
    schema_version: int = 1
    if external_files_data or models:
        schema_version = 2

    data: JSONObject = {
        "schema_version": schema_version,
        "pipeline": {
            "name": pipeline_name,
            "version": pipeline_version,
            "id": pipeline_id,
        },
        "run": {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": getattr(result, "status", None),
        },
        "skills": skills,
        "prompts": prompts,
        "result": _normalize_payload(
            {
                "success": getattr(result, "success", False),
                "final_output": getattr(result, "output", None),
                "steps": [
                    {
                        "name": sr.name,
                        "success": sr.success,
                        "feedback": sr.feedback,
                    }
                    for sr in getattr(result, "step_history", []) or []
                ],
            }
        ),
    }

    # Add new fields only if schema_version is 2
    if schema_version >= 2:
        if external_files_data:
            data["external_files"] = external_files_data
        if models:
            data["models"] = models

    return data


def write_lockfile(
    *,
    path: str | Path,
    pipeline: Pipeline[PipelineInT, PipelineOutT],
    result: PipelineResult[ContextT],
    pipeline_name: str | None,
    pipeline_version: str,
    pipeline_id: str,
    run_id: str | None,
    external_files: Optional[List[Path]] = None,
    include_model_info: bool = False,
    project_root: Optional[Path] = None,
) -> Path:
    """Write lockfile to disk.

    Args:
        path: Path to write lockfile to
        pipeline: The pipeline to generate lockfile for
        result: Pipeline execution result
        pipeline_name: Name of the pipeline
        pipeline_version: Version of the pipeline
        pipeline_id: Unique identifier for the pipeline
        run_id: Optional run ID
        external_files: Optional list of external files to hash
        include_model_info: Whether to include model information
        project_root: Project root for resolving relative paths

    Returns:
        Path to the written lockfile
    """
    data = build_lockfile_data(
        pipeline=pipeline,
        result=result,
        pipeline_name=pipeline_name,
        pipeline_version=pipeline_version,
        pipeline_id=pipeline_id,
        run_id=run_id,
        external_files=external_files,
        include_model_info=include_model_info,
        project_root=project_root,
    )
    target = Path(path)
    target.write_text(
        json.dumps(data, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def load_lockfile(path: str | Path) -> JSONObject:
    """Load lockfile from disk.

    Args:
        path: Path to lockfile

    Returns:
        JSONObject containing lockfile data

    Raises:
        FileNotFoundError: If lockfile doesn't exist
        json.JSONDecodeError: If lockfile is invalid JSON
    """
    lockfile_path = Path(path)
    if not lockfile_path.exists():
        raise FileNotFoundError(f"Lockfile not found: {lockfile_path}")

    content: str = lockfile_path.read_text(encoding="utf-8")
    loaded: object = json.loads(content)
    if not isinstance(loaded, dict):
        raise TypeError(f"Lockfile must contain a JSON object, got {type(loaded).__name__}")
    return cast(JSONObject, loaded)


class LockfileDiff(PydanticBaseModel):
    """Structure for lockfile differences."""

    pipeline_changed: bool = False
    schema_version_changed: bool = False
    prompts_changed: List[JSONObject] = Field(default_factory=list)
    skills_changed: List[JSONObject] = Field(default_factory=list)
    models_changed: List[JSONObject] = Field(default_factory=list)
    external_files_changed: List[JSONObject] = Field(default_factory=list)
    prompts_added: List[JSONObject] = Field(default_factory=list)
    prompts_removed: List[JSONObject] = Field(default_factory=list)
    skills_added: List[JSONObject] = Field(default_factory=list)
    skills_removed: List[JSONObject] = Field(default_factory=list)
    models_added: List[JSONObject] = Field(default_factory=list)
    models_removed: List[JSONObject] = Field(default_factory=list)
    external_files_added: List[JSONObject] = Field(default_factory=list)
    external_files_removed: List[JSONObject] = Field(default_factory=list)
    has_differences: bool = False


def compare_lockfiles(
    lockfile1: JSONObject,
    lockfile2: JSONObject,
    *,
    ignore_fields: Optional[List[str]] = None,
) -> LockfileDiff:
    """Compare two lockfiles and return differences.

    Args:
        lockfile1: First lockfile (typically existing)
        lockfile2: Second lockfile (typically new)
        ignore_fields: Optional list of fields to ignore in comparison (e.g., ["run.timestamp"])

    Returns:
        LockfileDiff object containing all differences
    """
    if ignore_fields is None:
        ignore_fields = []

    diff = LockfileDiff()

    # Helper to check if field should be ignored
    def should_ignore(field_path: str) -> bool:
        for ignore_pattern in ignore_fields:
            if field_path.startswith(ignore_pattern) or ignore_pattern in field_path:
                return True
        return False

    # Compare pipeline metadata
    if not should_ignore("pipeline"):
        pipeline1: JSONObject = lockfile1.get("pipeline", {})
        pipeline2: JSONObject = lockfile2.get("pipeline", {})
        if (
            pipeline1.get("name") != pipeline2.get("name")
            or pipeline1.get("version") != pipeline2.get("version")
            or pipeline1.get("id") != pipeline2.get("id")
        ):
            diff.pipeline_changed = True
            diff.has_differences = True

    # Compare schema version
    if not should_ignore("schema_version"):
        if lockfile1.get("schema_version") != lockfile2.get("schema_version"):
            diff.schema_version_changed = True
            diff.has_differences = True

    # Compare prompts
    prompts1: List[JSONObject] = lockfile1.get("prompts", [])
    prompts2: List[JSONObject] = lockfile2.get("prompts", [])

    prompts1_map: dict[str, JSONObject] = {p.get("step", ""): p for p in prompts1}
    prompts2_map: dict[str, JSONObject] = {p.get("step", ""): p for p in prompts2}

    for step_name, prompt1 in prompts1_map.items():
        if should_ignore(f"prompts.{step_name}"):
            continue
        prompt2 = prompts2_map.get(step_name)
        if prompt2 is None:
            diff.prompts_removed.append(prompt1)
            diff.has_differences = True
        elif prompt1.get("hash") != prompt2.get("hash"):
            if should_ignore(f"prompts.{step_name}.hash"):
                continue
            diff.prompts_changed.append(
                {
                    "step": step_name,
                    "old_hash": prompt1.get("hash"),
                    "new_hash": prompt2.get("hash"),
                }
            )
            diff.has_differences = True

    for step_name, prompt2 in prompts2_map.items():
        if should_ignore(f"prompts.{step_name}"):
            continue
        if step_name not in prompts1_map:
            diff.prompts_added.append(prompt2)
            diff.has_differences = True

    # Compare skills
    skills1: List[JSONObject] = lockfile1.get("skills", [])
    skills2: List[JSONObject] = lockfile2.get("skills", [])

    skills1_map: dict[str, JSONObject] = {s.get("step", ""): s for s in skills1}
    skills2_map: dict[str, JSONObject] = {s.get("step", ""): s for s in skills2}

    for step_name, skill1 in skills1_map.items():
        if should_ignore(f"skills.{step_name}"):
            continue
        skill2 = skills2_map.get(step_name)
        if skill2 is None:
            diff.skills_removed.append(skill1)
            diff.has_differences = True
        elif skill1.get("hash") != skill2.get("hash"):
            if should_ignore(f"skills.{step_name}.hash"):
                continue
            diff.skills_changed.append(
                {
                    "step": step_name,
                    "old_hash": skill1.get("hash"),
                    "new_hash": skill2.get("hash"),
                }
            )
            diff.has_differences = True

    for step_name, skill2 in skills2_map.items():
        if should_ignore(f"skills.{step_name}"):
            continue
        if step_name not in skills1_map:
            diff.skills_added.append(skill2)
            diff.has_differences = True

    # Compare models (if present)
    models1: List[JSONObject] = lockfile1.get("models", [])
    models2: List[JSONObject] = lockfile2.get("models", [])

    if models1 or models2:
        models1_map: dict[str, JSONObject] = {m.get("step", ""): m for m in models1}
        models2_map: dict[str, JSONObject] = {m.get("step", ""): m for m in models2}

        for step_name, model1 in models1_map.items():
            if should_ignore(f"models.{step_name}"):
                continue
            model2 = models2_map.get(step_name)
            if model2 is None:
                diff.models_removed.append(model1)
                diff.has_differences = True
            elif model1.get("hash") != model2.get("hash"):
                if should_ignore(f"models.{step_name}.hash"):
                    continue
                diff.models_changed.append(
                    {
                        "step": step_name,
                        "old_hash": model1.get("hash"),
                        "new_hash": model2.get("hash"),
                    }
                )
                diff.has_differences = True

        for step_name, model2 in models2_map.items():
            if should_ignore(f"models.{step_name}"):
                continue
            if step_name not in models1_map:
                diff.models_added.append(model2)
                diff.has_differences = True

    # Compare external files (if present)
    external_files1: List[JSONObject] = lockfile1.get("external_files", [])
    external_files2: List[JSONObject] = lockfile2.get("external_files", [])

    if external_files1 or external_files2:
        external_files1_map: dict[str, JSONObject] = {f.get("path", ""): f for f in external_files1}
        external_files2_map: dict[str, JSONObject] = {f.get("path", ""): f for f in external_files2}

        for file_path, file1 in external_files1_map.items():
            if should_ignore(f"external_files.{file_path}"):
                continue
            file2 = external_files2_map.get(file_path)
            if file2 is None:
                diff.external_files_removed.append(file1)
                diff.has_differences = True
            elif file1.get("hash") != file2.get("hash"):
                if should_ignore(f"external_files.{file_path}.hash"):
                    continue
                diff.external_files_changed.append(
                    {
                        "path": file_path,
                        "old_hash": file1.get("hash"),
                        "new_hash": file2.get("hash"),
                    }
                )
                diff.has_differences = True

        for file_path, file2 in external_files2_map.items():
            if should_ignore(f"external_files.{file_path}"):
                continue
            if file_path not in external_files1_map:
                diff.external_files_added.append(file2)
                diff.has_differences = True

    return diff


def format_lockfile_diff(diff: LockfileDiff) -> str:
    """Format lockfile differences for human-readable output.

    Args:
        diff: LockfileDiff object containing differences

    Returns:
        Formatted string describing differences
    """
    if not diff.has_differences:
        return "No differences found."

    lines: List[str] = ["Lockfile differences detected:"]

    if diff.pipeline_changed:
        lines.append("\n  Pipeline metadata changed:")
        lines.append("    - Name, version, or ID differs")

    if diff.schema_version_changed:
        lines.append("\n  Schema version changed:")

    if diff.prompts_changed:
        lines.append("\n  Prompts changed:")
        for change in diff.prompts_changed:
            lines.append(f"    - {change.get('step')}: hash changed")

    if diff.prompts_added:
        lines.append("\n  Prompts added:")
        for prompt in diff.prompts_added:
            lines.append(f"    - {prompt.get('step')}")

    if diff.prompts_removed:
        lines.append("\n  Prompts removed:")
        for prompt in diff.prompts_removed:
            lines.append(f"    - {prompt.get('step')}")

    if diff.skills_changed:
        lines.append("\n  Skills changed:")
        for change in diff.skills_changed:
            lines.append(f"    - {change.get('step')}: hash changed")

    if diff.skills_added:
        lines.append("\n  Skills added:")
        for skill in diff.skills_added:
            lines.append(f"    - {skill.get('step')}")

    if diff.skills_removed:
        lines.append("\n  Skills removed:")
        for skill in diff.skills_removed:
            lines.append(f"    - {skill.get('step')}")

    if diff.models_changed:
        lines.append("\n  Models changed:")
        for change in diff.models_changed:
            lines.append(f"    - {change.get('step')}: model config changed")

    if diff.models_added:
        lines.append("\n  Models added:")
        for model in diff.models_added:
            lines.append(f"    - {model.get('step')}: {model.get('model_id')}")

    if diff.models_removed:
        lines.append("\n  Models removed:")
        for model in diff.models_removed:
            lines.append(f"    - {model.get('step')}: {model.get('model_id')}")

    if diff.external_files_changed:
        lines.append("\n  External files changed:")
        for change in diff.external_files_changed:
            lines.append(f"    - {change.get('path')}: hash changed")

    if diff.external_files_added:
        lines.append("\n  External files added:")
        for file_info in diff.external_files_added:
            lines.append(f"    - {file_info.get('path')}")

    if diff.external_files_removed:
        lines.append("\n  External files removed:")
        for file_info in diff.external_files_removed:
            lines.append(f"    - {file_info.get('path')}")

    return "\n".join(lines)


def compute_lockfile_hash(lockfile: JSONObject) -> str:
    """Compute overall hash of lockfile for quick comparison.

    Args:
        lockfile: Lockfile JSONObject

    Returns:
        SHA-256 hash string
    """
    # Create a copy without run-specific fields for stable hashing
    stable_data: JSONObject = {
        "schema_version": lockfile.get("schema_version", 1),
        "pipeline": lockfile.get("pipeline", {}),
        "skills": lockfile.get("skills", []),
        "prompts": lockfile.get("prompts", []),
    }

    # Add optional fields if present
    if "models" in lockfile:
        stable_data["models"] = lockfile["models"]
    if "external_files" in lockfile:
        stable_data["external_files"] = lockfile["external_files"]

    return stable_digest(stable_data)
