from __future__ import annotations
from typing import (
    Any,
    ClassVar,
    Generic,
    Iterator,
    Optional,
    Sequence,
    TypeVar,
    TypeAlias,
    TYPE_CHECKING,
)
from pydantic import ConfigDict, Field, PrivateAttr, field_validator

from ..pipeline_validation import ValidationFinding, ValidationReport
from ..models import BaseModel
from flujo.domain.models import PipelineResult
from ...exceptions import ConfigurationError
from .step import Step, InvariantRule
from ..types import HookCallable
from .pipeline_validation_helpers import (
    aggregate_import_validation,
    apply_fallback_template_lints,
    apply_suppressions_from_meta,
    run_state_machine_lints,
    run_hitl_nesting_validation,
)
from .pipeline_step_validations import run_step_validations
from . import pipeline_io

PipeInT = TypeVar("PipeInT")
PipeOutT = TypeVar("PipeOutT")
NewPipeOutT = TypeVar("NewPipeOutT")

if TYPE_CHECKING:
    AnyStep: TypeAlias = Step[Any, Any]
else:
    AnyStep = Step  # type: ignore[misc]

__all__ = ["Pipeline"]


class Pipeline(BaseModel, Generic[PipeInT, PipeOutT]):
    """Ordered collection of :class:`Step` objects.

    ``Pipeline`` instances are immutable containers that define the execution
    graph. They can be composed with the ``>>`` operator and validated before
    running. Execution is handled by the :class:`~flujo.application.runner.Flujo`
    class.
    """

    steps: Sequence[AnyStep]
    hooks: list[HookCallable] = Field(default_factory=list)
    on_finish: list[HookCallable] = Field(default_factory=list)
    static_invariants: list[InvariantRule] = Field(
        default_factory=list,
        description="Hard invariants that must hold for the pipeline context.",
    )

    # Capture head/tail typing for the composed pipeline to aid static/dynamic checks.
    _input_type: Any = PrivateAttr(default=Any)
    _output_type: Any = PrivateAttr(default=Any)

    model_config: ClassVar[ConfigDict] = {
        "revalidate_instances": "never",
    }

    # -----------------------------
    # Internal helpers
    # -----------------------------

    @staticmethod
    def _extract_step_io(step: AnyStep) -> tuple[object, object]:
        """Return the declared input/output types for a step."""
        return (
            getattr(step, "__step_input_type__", Any),
            getattr(step, "__step_output_type__", Any),
        )

    @staticmethod
    def _compatible_types(a: Any, b: Any, *, is_adapter: bool = False) -> bool:
        """Strict type compatibility for adjacent steps.

        Mirrors V-A2/V-A2-STRICT/V-A2-TYPE logic: disallow Any/object fallthrough and
        require adapters for unsafe bridges (e.g., Pydantic->dict).
        """
        from typing import get_origin, get_args, Union as TypingUnion
        import types as _types

        if a in (Any, object, None, type(None)) or b in (Any, object, None, type(None)):  # noqa: E721
            return False

        origin_a, origin_b = get_origin(a), get_origin(b)
        try:
            from pydantic import BaseModel as _PydanticBaseModel

            if isinstance(a, type) and issubclass(a, _PydanticBaseModel):
                # Allow Pydantic model outputs to flow into dict expectations only via adapters.
                if b is dict or origin_b is dict:
                    return is_adapter
        except Exception:
            pass
        _UnionType = getattr(_types, "UnionType", None)

        if origin_b is TypingUnion or (_UnionType is not None and origin_b is _UnionType):
            return any(
                Pipeline._compatible_types(a, arg, is_adapter=is_adapter) for arg in get_args(b)
            )
        if origin_a is TypingUnion or (_UnionType is not None and origin_a is _UnionType):
            return all(
                Pipeline._compatible_types(arg, b, is_adapter=is_adapter) for arg in get_args(a)
            )

        try:
            b_eff = origin_b if origin_b is not None else b
            a_eff = origin_a if origin_a is not None else a
            if not isinstance(b_eff, type) or not isinstance(a_eff, type):
                return False
            return issubclass(a_eff, b_eff)
        except Exception:
            return False

    def _assert_adjacent_types_strict(self) -> None:
        try:
            from ...infra.settings import get_settings as _get_settings

            strict_mode = bool(getattr(_get_settings(), "strict_dsl", True))
        except Exception:
            strict_mode = True

        if not strict_mode:
            return

        prev_step: AnyStep | None = None
        prev_out_type: object | None = None
        for step in self.steps:
            in_type = getattr(step, "__step_input_type__", Any)
            out_type = getattr(step, "__step_output_type__", Any)
            meta = getattr(step, "meta", {}) or {}
            templated_input_present = (
                isinstance(meta, dict) and meta.get("templated_input") is not None
            )
            is_adapter_step = isinstance(meta, dict) and bool(meta.get("is_adapter", False))
            if prev_step is not None and prev_out_type is not None:
                generic_types = (Any, object, None, type(None))
                if (
                    not templated_input_present
                    and not is_adapter_step
                    and prev_out_type not in generic_types  # noqa: E721
                    and in_type not in generic_types  # noqa: E721
                    and not self._compatible_types(
                        prev_out_type, in_type, is_adapter=is_adapter_step
                    )
                ):
                    raise ValueError(
                        f"Type mismatch between steps '{prev_step.name}' (returns {prev_out_type}) and "
                        f"'{step.name}' (expects {in_type}). Insert an adapter step or adjust types."
                    )
            prev_step = step
            prev_out_type = out_type

    def _initialize_io_types(self) -> None:
        """Populate pipeline head/tail types from contained steps."""
        if not getattr(self, "steps", None):
            self._input_type = Any
            self._output_type = Any
            return
        self._assert_adjacent_types_strict()
        first_step = self.steps[0]
        last_step = self.steps[-1]
        self._input_type = self._extract_step_io(first_step)[0]
        self._output_type = self._extract_step_io(last_step)[1]

    @property
    def input_type(self) -> Any:
        """Head input type for the pipeline (derived from the first step)."""
        return self._input_type

    @property
    def output_type(self) -> Any:
        """Tail output type for the pipeline (derived from the last step)."""
        return self._output_type

    # ------------------------------------------------------------------
    # Construction & composition helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_step(cls, step: Step[PipeInT, PipeOutT]) -> "Pipeline[PipeInT, PipeOutT]":
        pipeline = cls.model_construct(steps=[step], hooks=[], on_finish=[])
        pipeline._initialize_io_types()
        return pipeline

    @classmethod
    def model_validate(cls, obj: Any, *args: Any, **kwargs: Any) -> "Pipeline[Any, Any]":
        """
        Preserve concrete Step subclasses (e.g., ParallelStep) when instances are provided.

        When callers pass already-constructed Step/ParallelStep objects, bypass Pydantic
        re-validation/coercion to avoid losing subclass-specific fields (like branches).
        """
        try:
            if isinstance(obj, dict):
                steps_val = obj.get("steps")
                if isinstance(steps_val, (list, tuple)) and all(
                    isinstance(s, Step) for s in steps_val
                ):
                    pipeline = cls.model_construct(
                        steps=list(steps_val),
                        hooks=list(obj.get("hooks", []) or []),
                        on_finish=list(obj.get("on_finish", []) or []),
                    )
                    pipeline._initialize_io_types()
                    return pipeline
        except Exception:
            pass
        pipeline = super().model_validate(obj, *args, **kwargs)
        try:
            pipeline._initialize_io_types()
        except Exception:
            pass
        return pipeline

    def model_post_init(self, __context: Any) -> None:  # noqa: D401
        """Ensure head/tail types are initialized after construction."""
        self._initialize_io_types()

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
            pipeline: Pipeline[PipeInT, NewPipeOutT] = Pipeline.model_construct(
                steps=new_steps, hooks=base_hooks, on_finish=base_finish
            )
            pipeline._initialize_io_types()
            return pipeline
        if isinstance(other, Pipeline):
            new_steps = list(self.steps) + list(other.steps)
            merged_hooks = base_hooks + list(getattr(other, "hooks", []) or [])
            merged_finish = base_finish + list(getattr(other, "on_finish", []) or [])
            combined_pipeline: Pipeline[PipeInT, NewPipeOutT] = Pipeline.model_construct(
                steps=new_steps, hooks=merged_hooks, on_finish=merged_finish
            )
            combined_pipeline._initialize_io_types()
            return combined_pipeline
        raise TypeError("Can only chain Pipeline with Step or Pipeline")

    # ------------------------------------------------------------------
    # YAML serialization helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, yaml_source: str, *, is_path: bool = True) -> "Pipeline[object, object]":
        """Load a Pipeline from YAML. When is_path=True, yaml_source is treated as a file path."""
        return pipeline_io.load_from_yaml(yaml_source, is_path=is_path)

    @classmethod
    def from_yaml_text(cls, yaml_text: str) -> "Pipeline[object, object]":
        return pipeline_io.load_from_yaml_text(yaml_text)

    @classmethod
    def from_yaml_file(cls, path: str) -> "Pipeline[object, object]":
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
    # Iteration helpers
    # ------------------------------------------------------------------

    def iter_steps(self) -> Iterator[Step[Any, Any]]:
        return iter(self.steps)

    def as_step(
        self, name: str, *, inherit_context: bool = True, **kwargs: Any
    ) -> "Step[PipeInT, PipelineResult[Any]]":
        """Wrap this pipeline as a composable Step, delegating to Flujo runner's as_step."""
        # Use deferred factory to avoid circular import of Flujo runner
        from ..interfaces import RunnerLike, get_runner_factory

        factory = get_runner_factory()
        runner: RunnerLike
        if factory is None:
            # Fallback for direct low-level usage without full package init
            from flujo.application.runner import Flujo

            runner = Flujo(self)
        else:
            runner = factory(self)

        return runner.as_step(name, inherit_context=inherit_context, **kwargs)  # type: ignore[no-any-return]


# Resolve forward references for hook payloads
try:
    from ..events import HookPayload as _HookPayload  # pragma: no cover

    Pipeline.model_rebuild(_types_namespace={"HookPayload": _HookPayload})
except Exception:  # pragma: no cover - defensive fallback
    Pipeline.model_rebuild()
