from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Generic, Optional, TypeVar, Any, Deque
from collections import deque

from ..domain.dsl.pipeline import Pipeline
from ..domain.dsl.step import Step, HumanInTheLoopStep
from ..domain.dsl.loop import LoopStep
from ..domain.dsl.parallel import ParallelStep
from ..domain.dsl.conditional import ConditionalStep
from ..domain.dsl.import_step import ImportStep
from ..domain.dsl.dynamic_router import DynamicParallelRouterStep
from ..exceptions import ConfigurationError, HitlPolicyError
from ..infra.registry import PipelineRegistry

RunnerInT = TypeVar("RunnerInT")
RunnerOutT = TypeVar("RunnerOutT")


@dataclass
class HitlPolicy:
    """Policy for allowing/disallowing HITL steps at runner resolution time."""

    allow_hitl: bool = True

    @classmethod
    def from_env(cls) -> "HitlPolicy":
        """Build a policy honoring FLUJO_ALLOW_HITL (default allows HITL)."""
        raw = os.getenv("FLUJO_ALLOW_HITL", "").strip().lower()
        if raw in {"0", "false", "no", "off"}:
            return cls(allow_hitl=False)
        return cls(allow_hitl=True)

    def enforce(self, pipeline: Pipeline[Any, Any]) -> None:
        """Raise when HITL steps are present but policy forbids them."""
        if self.allow_hitl:
            return
        if _pipeline_contains_hitl(pipeline):
            raise HitlPolicyError(
                "HITL steps are disabled by policy (set FLUJO_ALLOW_HITL=1 to enable)."
            )


def _pipeline_contains_hitl(pipeline: Pipeline[Any, Any]) -> bool:
    """Detect whether a pipeline (including nested branches) contains HITL steps."""
    queue: Deque[Step[Any, Any]] = deque(pipeline.steps)

    def _enqueue(candidate: Any) -> None:
        if isinstance(candidate, Pipeline):
            queue.extend(candidate.steps)
        elif isinstance(candidate, Step):
            queue.append(candidate)

    while queue:
        step = queue.popleft()
        if isinstance(step, HumanInTheLoopStep):
            return True
        if isinstance(step, LoopStep):
            _enqueue(step.loop_body_pipeline)
        if isinstance(step, ImportStep):
            _enqueue(step.pipeline)
        if isinstance(step, ParallelStep):
            for branch in step.branches.values():
                _enqueue(branch)
        if isinstance(step, ConditionalStep):
            for branch in step.branches.values():
                _enqueue(branch)
            _enqueue(step.default_branch_pipeline)
        if isinstance(step, DynamicParallelRouterStep):
            for branch in step.branches.values():
                _enqueue(branch)
    return False


@dataclass
class RunPlanResolver(Generic[RunnerInT, RunnerOutT]):
    """Resolve pipelines from an optional registry while preserving versioning."""

    pipeline: Pipeline[RunnerInT, RunnerOutT] | None
    registry: Optional[PipelineRegistry]
    pipeline_name: Optional[str]
    pipeline_version: str
    hitl_policy: HitlPolicy = field(default_factory=HitlPolicy.from_env)

    def ensure_pipeline(self) -> Pipeline[RunnerInT, RunnerOutT]:
        """Return a concrete pipeline, loading it from the registry if needed."""
        if self.pipeline is not None:
            self.hitl_policy.enforce(self.pipeline)
            return self.pipeline
        if self.registry is None or self.pipeline_name is None:
            raise ConfigurationError("Pipeline not provided and registry missing")
        if self.pipeline_version == "latest":
            version = self.registry.get_latest_version(self.pipeline_name)
            if version is None:
                raise ConfigurationError(
                    f"No pipeline registered under name '{self.pipeline_name}'"
                )
            self.pipeline_version = version
            pipe = self.registry.get(self.pipeline_name, version)
        else:
            pipe = self.registry.get(self.pipeline_name, self.pipeline_version)
        if pipe is None:
            raise ConfigurationError(
                f"Pipeline '{self.pipeline_name}' version '{self.pipeline_version}' not found"
            )
        self.hitl_policy.enforce(pipe)
        self.pipeline = pipe
        return pipe
