from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

from ..domain.dsl.pipeline import Pipeline
from ..exceptions import OrchestratorError
from ..infra.registry import PipelineRegistry

RunnerInT = TypeVar("RunnerInT")
RunnerOutT = TypeVar("RunnerOutT")


@dataclass
class RunPlanResolver(Generic[RunnerInT, RunnerOutT]):
    """Resolve pipelines from an optional registry while preserving versioning."""

    pipeline: Pipeline[RunnerInT, RunnerOutT] | None
    registry: Optional[PipelineRegistry]
    pipeline_name: Optional[str]
    pipeline_version: str

    def ensure_pipeline(self) -> Pipeline[RunnerInT, RunnerOutT]:
        """Return a concrete pipeline, loading it from the registry if needed."""
        if self.pipeline is not None:
            return self.pipeline
        if self.registry is None or self.pipeline_name is None:
            raise OrchestratorError("Pipeline not provided and registry missing")
        if self.pipeline_version == "latest":
            version = self.registry.get_latest_version(self.pipeline_name)
            if version is None:
                raise OrchestratorError(f"No pipeline registered under name '{self.pipeline_name}'")
            self.pipeline_version = version
            pipe = self.registry.get(self.pipeline_name, version)
        else:
            pipe = self.registry.get(self.pipeline_name, self.pipeline_version)
        if pipe is None:
            raise OrchestratorError(
                f"Pipeline '{self.pipeline_name}' version '{self.pipeline_version}' not found"
            )
        self.pipeline = pipe
        return pipe
