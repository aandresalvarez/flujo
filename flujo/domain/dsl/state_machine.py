from __future__ import annotations

from typing import Callable, ClassVar, Literal

from pydantic import Field, model_validator, PrivateAttr

from .pipeline import Pipeline
from .step import Step
from .conditional import ConditionalStep
from ..base_model import BaseModel


class TransitionRule(BaseModel):
    """Declarative transition rule for StateMachineStep.

    YAML fields:
      - from: source state name or "*" wildcard
      - on: one of {success, failure, pause}
      - to: destination state name
      - when: optional expression evaluated with (output, context)
    """

    from_state: str = Field(alias="from")
    on: Literal["success", "failure", "pause"]
    to: str
    when: str | None = None

    # Compiled predicate (not serialized)
    _when_fn: Callable[[object, object | None], object] | None = PrivateAttr(default=None)


class StateMachineStep(Step[object, object]):
    """A high-level DSL primitive for orchestrating pipelines via named states.

    The step holds a mapping of state name → Pipeline and metadata about the
    start and terminal states. Execution semantics are provided by a policy
    executor which iterates states until an end state is reached.
    """

    kind: ClassVar[str] = "StateMachine"

    # Map of state name → Pipeline to run for that state
    states: dict[str, Pipeline[object, object]] = Field(default_factory=dict)
    start_state: str
    end_states: list[str] = Field(default_factory=list)
    transitions: list[TransitionRule] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_states_to_pipelines(cls, data: object) -> object:
        try:
            if isinstance(data, dict):
                states_in = data.get("states")
                if isinstance(states_in, dict):
                    coerced: dict[str, object] = {}
                    for k, v in states_in.items():
                        if isinstance(v, Pipeline):
                            coerced[str(k)] = v
                        elif isinstance(v, Step):
                            coerced[str(k)] = Pipeline.from_step(v)
                        else:
                            # Best-effort; blueprint parsing happens in the blueprint loader.
                            coerced[str(k)] = v
                    data["states"] = coerced
        except Exception:
            # Best-effort; let Pydantic raise if still invalid
            pass
        return data

    @model_validator(mode="after")
    def _validate_and_compile_transitions(self) -> "StateMachineStep":
        try:
            # Build lookup sets
            state_keys = set(self.states.keys())
            end_keys = set(self.end_states or [])

            # Compile expressions
            if self.transitions:
                from flujo.utils.expressions import compile_expression_to_callable

                for tr in self.transitions:
                    # Validate 'from'
                    if tr.from_state != "*" and tr.from_state not in state_keys:
                        raise ValueError(
                            f"Transition 'from' references unknown state: {tr.from_state!r}"
                        )
                    # Validate 'to'
                    if tr.to not in state_keys and tr.to not in end_keys:
                        raise ValueError(
                            f"Transition 'to' must be an existing state or an end state: {tr.to!r}"
                        )
                    # Precompile predicate
                    if tr.when is not None and isinstance(tr.when, str) and tr.when.strip():
                        try:
                            tr._when_fn = compile_expression_to_callable(tr.when)
                        except Exception as e:
                            # Raise a clear error so blueprint validation fails early
                            raise ValueError(f"Invalid transition 'when' expression: {e}")
        except Exception:
            # Let Pydantic surface the validation error
            raise
        return self

    @property
    def is_complex(self) -> bool:  # noqa: D401
        """Signal to the executor that this is a complex orchestration step."""
        return True

    def build_internal_pipeline(self) -> Pipeline[object, object]:
        """Build a simple conditional wrapper over the state's pipelines.

        This does not implement the iteration semantics; the policy executor
        drives state transitions and termination. The conditional simply
        selects the pipeline for the current state.
        """

        def _select_branch(_out: object | None = None, ctx: object | None = None) -> str:
            try:
                if ctx is not None:
                    key = getattr(ctx, "current_state", None)
                    if isinstance(key, str) and key in self.states:
                        return key
            except Exception:
                pass
            return self.start_state

        cond: ConditionalStep[BaseModel] = ConditionalStep(
            name=f"{getattr(self, 'name', 'StateMachine')}:branch",
            condition_callable=_select_branch,
            branches=self.states,
            default_branch_pipeline=self.states.get(self.start_state),
        )
        return Pipeline.from_step(cond)
