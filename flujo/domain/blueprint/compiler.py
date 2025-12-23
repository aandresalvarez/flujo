"""Declarative blueprint compiler for pre-compiling agents and imports.

This module compiles declarative agent definitions and handles blueprint imports
before pipeline construction.

Import Strategy (per FLUJO_TEAM_GUIDE Section 12):
- Top-level imports: Standard library and core dependencies (os, pydantic, etc.)
- TYPE_CHECKING imports: Type-only imports to avoid circular dependencies
- Runtime imports: Used only for circular dependency resolution:
  - load_pipeline_blueprint_from_yaml: Loaded in _compile_imports() to avoid
    circular dependency with loader_parser module

This approach maintains type safety while allowing necessary circular imports
for recursive blueprint compilation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import os
from pydantic import BaseModel

from .loader import (
    BlueprintPipelineModel,
    build_pipeline_from_blueprint,
)
from .model_generator import generate_model_from_schema
from ...agents import make_agent_async, make_templated_agent_async
from flujo.type_definitions.common import JSONObject

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..dsl import Pipeline as _Pipeline
from ...exceptions import ConfigurationError


class DeclarativeAgentModel(BaseModel):
    model: str
    system_prompt: str
    output_schema: JSONObject


class DeclarativeBlueprintCompiler:
    """Compiler that pre-compiles declarative agents and wires steps using 'uses'."""

    def __init__(
        self,
        blueprint: BlueprintPipelineModel,
        base_dir: Optional[str] = None,
        _visited: Optional[list[str]] = None,
    ) -> None:
        self.blueprint = blueprint
        self._compiled_agents: dict[str, object] = {}
        self._compiled_imports: dict[str, _Pipeline[object, object]] = {}
        self._base_dir: Optional[str] = base_dir
        self._visited = _visited

    def _validate_and_coerce_max_retries(
        self, max_retries_opt: object, agent_name: str
    ) -> tuple[int, list[str]]:
        """Validate/coerce max_retries and collect warnings instead of raising.

        Returns a tuple of (value, warnings). Defaults to 3 on invalid input.
        """
        warnings: list[str] = []
        max_retries = 3
        if max_retries_opt is not None:
            try:
                if isinstance(max_retries_opt, str):
                    max_retries = int(max_retries_opt)
                elif isinstance(max_retries_opt, int):
                    max_retries = max_retries_opt
                else:
                    raise ValueError(f"Cannot convert {type(max_retries_opt).__name__} to int")
            except (ValueError, TypeError):
                warnings.append(
                    f"Agent '{agent_name}': Non-integer max_retries value '{max_retries_opt}' (using default 3)."
                )
                max_retries = 3
        if max_retries < 0:
            warnings.append(
                f"Agent '{agent_name}': Negative max_retries '{max_retries}' (using default 3)."
            )
            max_retries = 3
        return max_retries, warnings

    def _validate_and_coerce_timeout(
        self, timeout_opt: object, agent_name: str
    ) -> tuple[Optional[int], list[str]]:
        """Validate/coerce timeout to int seconds; collect warnings. None if invalid."""
        warnings: list[str] = []
        if timeout_opt is None:
            return None, warnings
        try:
            if isinstance(timeout_opt, str):
                # Accept pure integer strings
                timeout_val = int(timeout_opt)
            elif isinstance(timeout_opt, (int,)):
                timeout_val = int(timeout_opt)
            else:
                raise ValueError(f"Cannot convert {type(timeout_opt).__name__} to int")
            if timeout_val <= 0:
                raise ValueError("timeout must be positive")
            return timeout_val, warnings
        except (ValueError, TypeError):
            warnings.append(
                f"Agent '{agent_name}': Invalid timeout value '{timeout_opt}' (ignoring)."
            )
            return None, warnings

    def _compile_agents(self) -> None:
        agents: Optional[dict[str, object]] = getattr(self.blueprint, "agents", None)
        if not agents:
            return
        for name, spec in agents.items():
            # Support dict specs validated via model on loader side
            if isinstance(spec, dict):
                model_name = str(spec.get("model"))
                prompt_spec = spec.get("system_prompt")
                # Resolve system prompt possibly from external file
                system_prompt: str
                if isinstance(prompt_spec, dict) and "from_file" in prompt_spec:
                    rel = str(prompt_spec.get("from_file"))
                    base_dir = self._resolve_base_dir()
                    path = os.path.normpath(os.path.join(base_dir, rel))
                    real_base = os.path.realpath(base_dir)
                    real_path = os.path.realpath(path)
                    # Security: sandbox to base_dir
                    try:
                        if os.path.commonpath([real_base, real_path]) != real_base:
                            raise ConfigurationError(f"Path traversal detected in from_file: {rel}")
                    except ValueError:
                        # On Windows, different drives cause ValueError
                        raise ConfigurationError(f"Path traversal detected in from_file: {rel}")
                    try:
                        with open(real_path, "r", encoding="utf-8") as f:
                            system_prompt = f.read()
                    except FileNotFoundError:
                        raise ConfigurationError(
                            f"Prompt file not found for agent '{name}': {real_path}"
                        )
                    except Exception as e:
                        raise ConfigurationError(f"Error reading prompt file '{real_path}': {e}")
                elif isinstance(prompt_spec, str):
                    system_prompt = prompt_spec
                else:
                    system_prompt = str(prompt_spec) if prompt_spec is not None else ""
                output_schema = spec.get("output_schema") or {}
                # Optional GPT-5 style controls passed through to Agent
                model_settings = spec.get("model_settings") or {}
                # Optional execution controls
                timeout_opt = spec.get("timeout")
                max_retries_opt = spec.get("max_retries")
            else:
                # Already a parsed model-like (fallback)
                model_name = str(getattr(spec, "model"))
                prompt_spec = getattr(spec, "system_prompt")
                if hasattr(prompt_spec, "from_file"):
                    rel = str(getattr(prompt_spec, "from_file"))
                    base_dir = self._resolve_base_dir()
                    path = os.path.normpath(os.path.join(base_dir, rel))
                    real_base = os.path.realpath(base_dir)
                    real_path = os.path.realpath(path)
                    try:
                        if os.path.commonpath([real_base, real_path]) != real_base:
                            raise ConfigurationError(f"Path traversal detected in from_file: {rel}")
                    except ValueError:
                        raise ConfigurationError(f"Path traversal detected in from_file: {rel}")
                    try:
                        with open(real_path, "r", encoding="utf-8") as f:
                            system_prompt = f.read()
                    except FileNotFoundError:
                        raise ConfigurationError(
                            f"Prompt file not found for agent '{name}': {real_path}"
                        )
                    except Exception as e:
                        raise ConfigurationError(f"Error reading prompt file '{real_path}': {e}")
                else:
                    system_prompt = str(prompt_spec)
                output_schema = getattr(spec, "output_schema")
                try:
                    model_settings = getattr(spec, "model_settings", {}) or {}
                except Exception:
                    model_settings = {}
                try:
                    timeout_opt = getattr(spec, "timeout", None)
                except Exception:
                    timeout_opt = None
                try:
                    max_retries_opt = getattr(spec, "max_retries", None)
                except Exception:
                    max_retries_opt = None

            # Lint JSON schema (V-S1) for basic structural issues
            def _lint_schema(sc: JSONObject) -> list[str]:
                warnings: list[str] = []
                try:
                    st = sc.get("type")
                    if st is not None and st not in {
                        "string",
                        "number",
                        "integer",
                        "boolean",
                        "object",
                        "array",
                    }:
                        warnings.append(f"Unknown schema type: {st}")
                    # required outside properties
                    req = sc.get("required")
                    props = sc.get("properties")
                    if isinstance(req, list) and not isinstance(props, dict):
                        warnings.append(
                            "'required' present but 'properties' is missing or not an object"
                        )
                    # array missing items
                    if st == "array" and not isinstance(sc.get("items"), dict):
                        warnings.append("array type without an 'items' schema")
                except Exception:
                    pass
                return warnings

            output_type = generate_model_from_schema(name, output_schema)

            # If from_file is present, always use templated wrapper (variables are optional)
            agent_wrapper: object
            if isinstance(prompt_spec, dict) and "from_file" in prompt_spec:
                # Variables are optional when using from_file - use empty dict if absent
                variables_spec = {}
                if "variables" in prompt_spec:
                    try:
                        variables_spec = dict(prompt_spec["variables"])
                    except Exception as e:
                        raise ConfigurationError(
                            f"Agent '{name}': Failed to convert variables to dict: {prompt_spec['variables']} - {e}"
                        )

                # Validate and coerce controls
                max_retries, warns_mr = self._validate_and_coerce_max_retries(max_retries_opt, name)
                timeout_val, warns_to = self._validate_and_coerce_timeout(timeout_opt, name)
                _coercion_warnings = warns_mr + warns_to

                agent_wrapper = make_templated_agent_async(
                    model=model_name,
                    template_string=system_prompt,
                    variables_spec=variables_spec,
                    output_type=output_type,
                    model_settings=model_settings,
                    timeout=timeout_val,
                    max_retries=max_retries,
                )
            elif (
                hasattr(prompt_spec, "variables") and getattr(prompt_spec, "variables") is not None
            ):
                try:
                    variables_spec2 = dict(getattr(prompt_spec, "variables"))
                except Exception as e:
                    raise ConfigurationError(
                        f"Agent '{name}': Failed to convert variables to dict: {getattr(prompt_spec, 'variables')} - {e}"
                    )

                # Validate and coerce controls
                max_retries, warns_mr = self._validate_and_coerce_max_retries(max_retries_opt, name)
                timeout_val, warns_to = self._validate_and_coerce_timeout(timeout_opt, name)
                _coercion_warnings = warns_mr + warns_to

                agent_wrapper = make_templated_agent_async(
                    model=model_name,
                    template_string=system_prompt,
                    variables_spec=variables_spec2,
                    output_type=output_type,
                    model_settings=model_settings,
                    timeout=timeout_val,
                    max_retries=max_retries,
                )
            else:
                # Validate and coerce controls
                max_retries, warns_mr = self._validate_and_coerce_max_retries(max_retries_opt, name)
                timeout_val, warns_to = self._validate_and_coerce_timeout(timeout_opt, name)
                _coercion_warnings = warns_mr + warns_to

                agent_wrapper = make_agent_async(
                    model=model_name,
                    system_prompt=system_prompt,
                    output_type=output_type,
                    # Pass through provider-specific model settings (e.g., GPT-5 controls)
                    model_settings=model_settings,
                    timeout=timeout_val,
                    max_retries=max_retries,
                )
            self._compiled_agents[name] = agent_wrapper

            # Attach schema warnings for later surfacing during pipeline validation (V-S1)
            try:
                setattr(agent_wrapper, "_schema_warnings", _lint_schema(output_schema))
            except Exception:
                pass
            # Attach declared output schema (for V-S2/V-S3 heuristics)
            try:
                if isinstance(output_schema, dict) and output_schema:
                    setattr(agent_wrapper, "_declared_output_schema", output_schema)
            except Exception:
                pass
            # Attach coercion warnings for validation pass (V-A7)
            try:
                if _coercion_warnings:
                    setattr(agent_wrapper, "_coercion_warnings", list(_coercion_warnings))
            except Exception:
                pass

    def _resolve_base_dir(self) -> str:
        """Resolve the base directory for relative path resolution.

        Returns:
            Base directory string, either from constructor or current working directory
        """
        if self._base_dir:
            return self._base_dir
        # Default to current working directory
        # Note: os is imported at module level to follow FLUJO_TEAM_GUIDE Section 12
        return os.getcwd()

    def _compile_imports(self) -> None:
        """Load and compile imported blueprints into Pipeline objects cached by alias.

        This method handles recursive blueprint imports with cycle detection.
        Runtime import of load_pipeline_blueprint_from_yaml is required to avoid
        circular dependency between compiler and loader_parser modules.
        """
        imports: Optional[dict[str, str]] = getattr(self.blueprint, "imports", None)
        if not imports:
            return
        # Runtime import required to avoid circular dependency with loader_parser module
        from .loader import load_pipeline_blueprint_from_yaml

        base_dir: str = self._resolve_base_dir()
        for alias, rel_path in imports.items():
            try:
                path = rel_path
                if not os.path.isabs(path):
                    path = os.path.normpath(os.path.join(base_dir, path))
                real_path = os.path.realpath(path)
                with open(real_path, "r", encoding="utf-8") as f:
                    text = f.read()
                # Recursively compile with a new compiler instance; pass directory of the imported file
                sub_base_dir = os.path.dirname(real_path)
                # Use loader entrypoint to ensure same validation and compilation path
                sub_pipeline = load_pipeline_blueprint_from_yaml(
                    text,
                    base_dir=sub_base_dir,
                    source_file=real_path,
                    _visited=self._visited,
                )
                self._compiled_imports[alias] = sub_pipeline
            except Exception as e:
                # Fail fast with descriptive message
                raise RuntimeError(
                    f"Failed to compile import '{alias}' from '{rel_path}': {e}"
                ) from e

    def compile_to_pipeline(self) -> "_Pipeline[object, object]":
        # Compile agents and imports first
        self._compile_agents()
        self._compile_imports()
        # Delegate pipeline construction, providing compiled agent and import mapping
        pipeline = build_pipeline_from_blueprint(
            self.blueprint,
            compiled_agents=self._compiled_agents,
            compiled_imports=self._compiled_imports,
        )

        # Propagate top-level blueprint name onto the Pipeline object so downstream
        # components (CLI runner, tracing) can display and persist a meaningful name.
        name_val = getattr(self.blueprint, "name", None)
        if isinstance(name_val, str):
            name_stripped = name_val.strip()
            if name_stripped:
                try:
                    # Prefer direct attribute set; tolerate missing attribute in strict mypy
                    pipeline.name = name_stripped  # type: ignore[attr-defined]
                except (AttributeError, TypeError, ValueError):
                    # Bypass potential attribute guards/frozen models
                    try:
                        object.__setattr__(pipeline, "name", name_stripped)
                    except (AttributeError, TypeError, ValueError):
                        # best-effort only
                        pass

        return pipeline


__all__ = ["DeclarativeBlueprintCompiler", "DeclarativeAgentModel"]
