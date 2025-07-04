from __future__ import annotations

from typing import Any, Literal, Optional, Union, TYPE_CHECKING

from pydantic import BaseModel

from .models import PipelineResult, StepResult

if TYPE_CHECKING:
    pass
from .resources import AppResources


class PreRunPayload(BaseModel):
    """Payload for pre-run hooks.

    Contains the initial input, context object, and resources that will be used
    during pipeline execution.
    """

    event_name: Literal["pre_run"]
    initial_input: Any
    context: Optional[BaseModel] = None
    resources: Optional[AppResources] = None


class PostRunPayload(BaseModel):
    """Payload for post-run hooks.

    Contains the final pipeline result, context object, and resources used
    during pipeline execution.
    """

    event_name: Literal["post_run"]
    pipeline_result: PipelineResult[Any]
    context: Optional[BaseModel] = None
    resources: Optional[AppResources] = None


class PreStepPayload(BaseModel):
    """Payload for pre-step hooks.

    Contains the step that is about to be executed, its input data, context object,
    and resources available for the step.
    """

    event_name: Literal["pre_step"]
    step: Any  # Step[Any, Any] - using Any to avoid forward reference issues
    step_input: Any
    context: Optional[BaseModel] = None
    resources: Optional[AppResources] = None

    # Runtime validation to ensure step is a Step instance
    @classmethod
    def model_validate(cls, *args: Any, **kwargs: Any) -> "PreStepPayload":
        from .dsl.step import Step

        step = kwargs.get("step")
        if step is not None and not isinstance(step, Step):
            raise ValueError(f"step must be a Step instance, got {type(step)}")
        return super().model_validate(*args, **kwargs)


class PostStepPayload(BaseModel):
    """Payload for post-step hooks.

    Contains the step result, context object, and resources that were used
    during step execution.
    """

    event_name: Literal["post_step"]
    step_result: StepResult
    context: Optional[BaseModel] = None
    resources: Optional[AppResources] = None


class OnStepFailurePayload(BaseModel):
    """Payload for step failure hooks.

    Contains the step result (with failure details), context object, and resources
    that were used during the failed step execution.
    """

    event_name: Literal["on_step_failure"]
    step_result: StepResult
    context: Optional[BaseModel] = None
    resources: Optional[AppResources] = None


HookPayload = Union[
    PreRunPayload,
    PostRunPayload,
    PreStepPayload,
    PostStepPayload,
    OnStepFailurePayload,
]
