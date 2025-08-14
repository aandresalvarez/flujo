from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel

from .loader import (
    BlueprintPipelineModel,
    build_pipeline_from_blueprint,
)
from .model_generator import generate_model_from_schema
from ...agents import make_agent_async


class DeclarativeAgentModel(BaseModel):
    model: str
    system_prompt: str
    output_schema: Dict[str, Any]


class DeclarativeBlueprintCompiler:
    """Compiler that pre-compiles declarative agents and wires steps using 'uses'."""

    def __init__(self, blueprint: BlueprintPipelineModel, base_dir: Optional[str] = None) -> None:
        self.blueprint = blueprint
        self._compiled_agents: Dict[str, Any] = {}
        self._compiled_imports: Dict[str, Any] = {}
        self._base_dir: Optional[str] = base_dir

    def _compile_agents(self) -> None:
        agents: Optional[Dict[str, Any]] = getattr(self.blueprint, "agents", None)
        if not agents:
            return
        for name, spec in agents.items():
            # Support dict specs validated via model on loader side
            if isinstance(spec, dict):
                model_name = str(spec.get("model"))
                system_prompt = str(spec.get("system_prompt"))
                output_schema = spec.get("output_schema") or {}
            else:
                # Already a parsed model-like (fallback)
                model_name = str(getattr(spec, "model"))
                system_prompt = str(getattr(spec, "system_prompt"))
                output_schema = getattr(spec, "output_schema")

            output_type = generate_model_from_schema(name, output_schema)
            agent_wrapper = make_agent_async(
                model=model_name,
                system_prompt=system_prompt,
                output_type=output_type,
            )
            self._compiled_agents[name] = agent_wrapper

    def _resolve_base_dir(self) -> str:
        import os

        if self._base_dir:
            return self._base_dir
        # Default to current working directory
        return os.getcwd()

    def _compile_imports(self) -> None:
        """Load and compile imported blueprints into Pipeline objects cached by alias."""
        imports: Optional[Dict[str, str]] = getattr(self.blueprint, "imports", None)
        if not imports:
            return
        import os
        from .loader import load_pipeline_blueprint_from_yaml

        base_dir = self._resolve_base_dir()
        for alias, rel_path in imports.items():
            try:
                path = rel_path
                if not os.path.isabs(path):
                    path = os.path.normpath(os.path.join(base_dir, path))
                with open(path, "r") as f:
                    text = f.read()
                # Recursively compile with a new compiler instance; pass directory of the imported file
                sub_base_dir = os.path.dirname(path)
                # Use loader entrypoint to ensure same validation and compilation path
                sub_pipeline = load_pipeline_blueprint_from_yaml(text, base_dir=sub_base_dir)
                self._compiled_imports[alias] = sub_pipeline
            except Exception as e:
                # Fail fast with descriptive message
                raise RuntimeError(
                    f"Failed to compile import '{alias}' from '{rel_path}': {e}"
                ) from e

    def compile_to_pipeline(self) -> Any:
        # Compile agents and imports first
        self._compile_agents()
        self._compile_imports()
        # Delegate pipeline construction, providing compiled agent and import mapping
        return build_pipeline_from_blueprint(
            self.blueprint,
            compiled_agents=self._compiled_agents,
            compiled_imports=self._compiled_imports,
        )


__all__ = ["DeclarativeBlueprintCompiler", "DeclarativeAgentModel"]
