from __future__ import annotations
from typing import Any, ClassVar, Generic, Iterator, Optional, Sequence, TypeVar
import logging
from pydantic import ConfigDict, Field, field_validator

from ..pipeline_validation import ValidationFinding, ValidationReport
from ..models import BaseModel
from flujo.domain.models import PipelineResult
from ...exceptions import ConfigurationError
from .step import Step
from ..types import HookCallable
from .pipeline_validation_helpers import (
    aggregate_import_validation,
    apply_fallback_template_lints,
    apply_suppressions_from_meta,
    run_state_machine_lints,
    run_hitl_nesting_validation,
    run_step_validations,
)
from . import pipeline_io

PipeInT = TypeVar("PipeInT")
PipeOutT = TypeVar("PipeOutT")
NewPipeOutT = TypeVar("NewPipeOutT")

__all__ = ["Pipeline"]


class Pipeline(BaseModel, Generic[PipeInT, PipeOutT]):
    """Ordered collection of :class:`Step` objects.

    ``Pipeline`` instances are immutable containers that define the execution
    graph. They can be composed with the ``>>`` operator and validated before
    running. Execution is handled by the :class:`~flujo.application.runner.Flujo`
    class.
    """

    steps: Sequence[Step[Any, Any]]
    hooks: list[HookCallable] = Field(default_factory=list)
    on_finish: list[HookCallable] = Field(default_factory=list)

    model_config: ClassVar[ConfigDict] = {
        "arbitrary_types_allowed": True,
        "revalidate_instances": "never",
    }

    # ------------------------------------------------------------------
    # Construction & composition helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_step(cls, step: Step[PipeInT, PipeOutT]) -> "Pipeline[PipeInT, PipeOutT]":
        return cls.model_construct(steps=[step], hooks=[], on_finish=[])

    # Preserve concrete Step subclasses (e.g., CacheStep, HumanInTheLoopStep)
    @field_validator("steps", mode="before")
    @classmethod
    def _preserve_step_subclasses(cls, v: Any) -> Any:
        try:
            if isinstance(v, (list, tuple)) and all(isinstance(s, Step) for s in v):
                # Return as-is to avoid coercing subclass instances into base Step
                return list(v)
        except Exception:
            pass
        return v

    def __rshift__(
        self, other: Step[PipeOutT, NewPipeOutT] | "Pipeline[PipeOutT, NewPipeOutT]"
    ) -> "Pipeline[PipeInT, NewPipeOutT]":
        base_hooks = list(getattr(self, "hooks", []) or [])
        base_finish = list(getattr(self, "on_finish", []) or [])
        if isinstance(other, Step):
            new_steps = list(self.steps) + [other]
            return Pipeline.model_construct(
                steps=new_steps, hooks=base_hooks, on_finish=base_finish
            )
        if isinstance(other, Pipeline):
            new_steps = list(self.steps) + list(other.steps)
            merged_hooks = base_hooks + list(getattr(other, "hooks", []) or [])
            merged_finish = base_finish + list(getattr(other, "on_finish", []) or [])
            return Pipeline.model_construct(
                steps=new_steps, hooks=merged_hooks, on_finish=merged_finish
            )
        raise TypeError("Can only chain Pipeline with Step or Pipeline")

    # ------------------------------------------------------------------
    # YAML serialization helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, yaml_source: str, *, is_path: bool = True) -> "Pipeline[Any, Any]":
        """Load a Pipeline from YAML. When is_path=True, yaml_source is treated as a file path."""
        return pipeline_io.load_from_yaml(yaml_source, is_path=is_path)

    @classmethod
    def from_yaml_text(cls, yaml_text: str) -> "Pipeline[Any, Any]":
        return pipeline_io.load_from_yaml_text(yaml_text)

    @classmethod
    def from_yaml_file(cls, path: str) -> "Pipeline[Any, Any]":
        return pipeline_io.load_from_yaml_file(path)

    def to_yaml(self) -> str:
        return pipeline_io.dump_to_yaml(self)

    def to_yaml_file(self, path: str) -> None:
        pipeline_io.dump_to_yaml_file(self, path)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def validate_graph(
        self,
        *,
        raise_on_error: bool = False,
        include_imports: bool = False,
        _visited_pipelines: Optional[set[int]] = None,
        _visited_paths: Optional[set[str]] = None,
        _report_cache: Optional[dict[str, "ValidationReport"]] = None,
    ) -> ValidationReport:
        """Validate that all steps have agents, compatible types, and static lints.

        Adds advanced static checks:
        - V-P1: Parallel context merge conflict detection for default CONTEXT_UPDATE without field_mapping
        - V-A5: Unbound output warning when a step's output is unused and it does not update context
        - V-F1: Incompatible fallback signature between step and fallback_step
        """
        # Reset rule override cache to honor current environment/profile for each validation pass.
        try:
            import flujo.validation.linters_base as _lb

            _lb._OVERRIDE_CACHE = None
        except Exception:
            pass
        from typing import Any, get_origin, get_args, Union as TypingUnion
        import types as _types

        def _compatible(a: Any, b: Any) -> bool:
            if a is Any or b is Any:
                return True

            origin_a, origin_b = get_origin(a), get_origin(b)
            _UnionType = getattr(_types, "UnionType", None)

            if origin_b is TypingUnion or (_UnionType is not None and origin_b is _UnionType):
                return any(_compatible(a, arg) for arg in get_args(b))
            if origin_a is TypingUnion or (_UnionType is not None and origin_a is _UnionType):
                return all(_compatible(arg, b) for arg in get_args(a))

            try:
                # Relaxed compatibility for common dict-like bridges
                # Many built-in skills return JSONObject. Allow flowing into object/str inputs,
                # because YAML param templating often selects a concrete field at runtime.
                # Treat both direct dict and typing.Dict origins as dict-like.
                origin_a = get_origin(a)
                origin_b = get_origin(b)
                is_dict_like_a = (a is dict) or (origin_a is dict)
                # Treat Pydantic models as dict-like for validation bridge
                try:
                    from pydantic import BaseModel as _PydanticBaseModel

                    if isinstance(a, type) and issubclass(a, _PydanticBaseModel):
                        is_dict_like_a = True
                except Exception:
                    pass

                if is_dict_like_a and (b is object or b is str or origin_b is dict):
                    return True
                # Avoid issubclass checks with typing constructs (e.g., typing.Dict)
                b_eff = origin_b if origin_b is not None else b
                a_eff = origin_a if origin_a is not None else a
                if not isinstance(b_eff, type) or not isinstance(a_eff, type):
                    return False
                return issubclass(a_eff, b_eff)
            except Exception as e:  # pragma: no cover
                logging.warning("_compatible: issubclass(%s, %s) raised %s", a, b, e)
                return False

        report = ValidationReport()
        # Initialize visited sets/caches to guard recursion/cycles and enable caching
        if _visited_pipelines is None:
            _visited_pipelines = set()
        if _visited_paths is None:
            _visited_paths = set()
        if _report_cache is None:
            _report_cache = {}
        cur_id = id(self)
        if cur_id in _visited_pipelines:
            return report
        _visited_pipelines.add(cur_id)
        try:
            import os as _os_path

            cur_path = getattr(self, "_source_file", None)
            if isinstance(cur_path, str):
                cur_path = _os_path.path.realpath(cur_path)
                if cur_path in _visited_paths:
                    return report
                _visited_paths.add(cur_path)
        except Exception:
            pass

        run_hitl_nesting_validation(self, report, raise_on_error=raise_on_error)
        run_step_validations(self, report, raise_on_error=raise_on_error)

        run_state_machine_lints(self, report)

        # Agent coercion lints moved to AgentLinter (V-A7)

        # Template lints moved to TemplateLinter (V-T1..V-T6)

        # Advanced Check 3.2.1: Context Merge Conflict Detection for ParallelStep (V-P1)
        # Use runtime imports with fallbacks while keeping mypy satisfied by typing as Any
        from typing import Any as _Any

        ParallelStep: _Any
        MergeStrategy: _Any
        try:
            from .parallel import ParallelStep as _ParallelStep  # local import to avoid circular
            from .step import MergeStrategy as _MergeStrategy  # enum

            ParallelStep = _ParallelStep
            MergeStrategy = _MergeStrategy
        except Exception:  # pragma: no cover - defensive import failure
            ParallelStep = None
            MergeStrategy = None

        if ParallelStep is not None and MergeStrategy is not None:
            for st in self.steps:
                if isinstance(st, ParallelStep):
                    # Parallel merge conflict checks moved to OrchestrationLinter (V-P1/V-P1-W)
                    pass

        if raise_on_error and report.errors:
            raise ConfigurationError(
                "Pipeline validation failed: " + report.model_dump_json(indent=2)
            )

        aggregate_import_validation(
            self,
            report,
            include_imports=include_imports,
            visited_pipelines=_visited_pipelines,
            visited_paths=_visited_paths,
            report_cache=_report_cache,
        )

        # Optional: run pluggable linters and merge (deduplicated)
        try:
            from ...validation.linters import run_linters as _run_linters

            lr = _run_linters(self)
            if lr and (lr.errors or lr.warnings):
                merged_errs = report.errors + lr.errors
                merged_warns = report.warnings + lr.warnings

                def _dedupe(arr: list[ValidationFinding]) -> list[ValidationFinding]:
                    seen: set[tuple[str, str | None, str]] = set()
                    out: list[ValidationFinding] = []
                    for it in arr:
                        key = (
                            str(getattr(it, "rule_id", "")),
                            getattr(it, "step_name", None),
                            str(getattr(it, "message", "")),
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        out.append(it)
                    return out

                report.errors = _dedupe(merged_errs)
                report.warnings = _dedupe(merged_warns)
        except Exception:
            pass

        apply_fallback_template_lints(self, report)

        apply_suppressions_from_meta(self, report)

        return report

    # ------------------------------------------------------------------
    # Iteration helpers & visualization methods (delegated mostly)
    # ------------------------------------------------------------------

    def iter_steps(self) -> Iterator[Step[Any, Any]]:
        return iter(self.steps)

    # ------------------------------------------------------------------
    # Visualization helpers (Mermaid generation) – copied from legacy implementation
    # ------------------------------------------------------------------

    def to_mermaid(self) -> str:  # noqa: D401
        """Generate a Mermaid graph definition for visualizing this pipeline."""
        from . import pipeline_mermaid

        return pipeline_mermaid.to_mermaid(self)

    def to_mermaid_with_detail_level(self, detail_level: str = "auto") -> str:  # noqa: D401
        """Generate a Mermaid graph definition with configurable detail levels."""
        from . import pipeline_mermaid

        return pipeline_mermaid.to_mermaid_with_detail_level(self, detail_level)

    # ---------------------- internal visualization utils --------------------

    def _determine_optimal_detail_level(self) -> str:
        """Heuristic to pick a detail level based on pipeline complexity."""
        from . import pipeline_mermaid

        return pipeline_mermaid._determine_optimal_detail_level(self)

    def _calculate_complexity_score(self) -> int:
        from . import pipeline_mermaid

        return pipeline_mermaid._calculate_complexity_score(self)

    # High / medium / low detail graph generators – directly migrated from legacy

    def _generate_high_detail_mermaid(self) -> str:  # noqa: C901 – complexity inherited
        from . import pipeline_mermaid

        return pipeline_mermaid._generate_high_detail_mermaid(self)

    def _generate_medium_detail_mermaid(self) -> str:
        from . import pipeline_mermaid

        return pipeline_mermaid._generate_medium_detail_mermaid(self)

    def _generate_low_detail_mermaid(self) -> str:
        from . import pipeline_mermaid

        return pipeline_mermaid._generate_low_detail_mermaid(self)

    def as_step(
        self, name: str, *, inherit_context: bool = True, **kwargs: Any
    ) -> "Step[PipeInT, PipelineResult[Any]]":
        """Wrap this pipeline as a composable Step, delegating to Flujo runner's as_step."""
        from flujo.application.runner import Flujo
        from flujo.domain.models import PipelineContext

        runner: Flujo[PipeInT, PipeOutT, PipelineContext] = Flujo(self)
        return runner.as_step(name, inherit_context=inherit_context, **kwargs)


# Resolve forward references for hook payloads
try:
    from ..events import HookPayload as _HookPayload  # pragma: no cover

    Pipeline.model_rebuild(_types_namespace={"HookPayload": _HookPayload})
except Exception:  # pragma: no cover - defensive fallback
    Pipeline.model_rebuild()
