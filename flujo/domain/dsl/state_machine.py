from __future__ import annotations

from typing import Any, ClassVar, Dict, List

from pydantic import Field, model_validator

from .pipeline import Pipeline
from .step import Step
from .conditional import ConditionalStep


class StateMachineStep(Step[Any, Any]):
    """A high-level DSL primitive for orchestrating pipelines via named states.

    The step holds a mapping of state name → Pipeline and metadata about the
    start and terminal states. Execution semantics are provided by a policy
    executor which iterates states until an end state is reached.
    """

    kind: ClassVar[str] = "StateMachine"

    # Map of state name → Pipeline to run for that state
    states: Dict[str, Pipeline[Any, Any]] = Field(default_factory=dict)
    start_state: str
    end_states: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_states_to_pipelines(cls, data: Any) -> Any:
        try:
            if isinstance(data, dict):
                states_in = data.get("states")
                from flujo.domain.blueprint.loader import _build_pipeline_from_branch as _mk

                if isinstance(states_in, dict):
                    coerced: Dict[str, Pipeline[Any, Any]] = {}
                    for k, v in states_in.items():
                        # Accept a list[step-spec] or a single step-spec dict
                        coerced[str(k)] = _mk(v)
                    data["states"] = coerced
        except Exception:
            # Best-effort; let Pydantic raise if still invalid
            pass
        return data

    @property
    def is_complex(self) -> bool:  # noqa: D401
        """Signal to the executor that this is a complex orchestration step."""
        return True

    def build_internal_pipeline(self) -> Pipeline[Any, Any]:
        """Build a simple conditional wrapper over the state's pipelines.

        This does not implement the iteration semantics; the policy executor
        drives state transitions and termination. The conditional simply
        selects the pipeline for the current state.
        """

        def _select_branch(_out: Any = None, ctx: Any | None = None) -> str:
            try:
                if ctx is not None and hasattr(ctx, "scratchpad"):
                    sp = getattr(ctx, "scratchpad", {})
                    key = sp.get("current_state") if isinstance(sp, dict) else None
                    if isinstance(key, str) and key in self.states:
                        return key
            except Exception:
                pass
            return self.start_state

        from typing import Any as _Any

        cond: ConditionalStep[_Any] = ConditionalStep(
            name=f"{getattr(self, 'name', 'StateMachine')}:branch",
            condition_callable=_select_branch,
            branches=self.states,
            default_branch_pipeline=self.states.get(self.start_state),
        )
        return Pipeline.from_step(cond)
