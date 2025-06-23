from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel

from .models import PipelineResult, StepResult
from .pipeline_dsl import Step
from .resources import AppResources


class PreRunPayload(BaseModel):
    event_name: Literal["pre_run"]
    initial_input: Any
    pipeline_context: Optional[BaseModel] = None
    resources: Optional[AppResources] = None


class PostRunPayload(BaseModel):
    event_name: Literal["post_run"]
    pipeline_result: PipelineResult[Any]
    pipeline_context: Optional[BaseModel] = None
    resources: Optional[AppResources] = None


class PreStepPayload(BaseModel):
    event_name: Literal["pre_step"]
    step: Step[Any, Any]
    step_input: Any
    pipeline_context: Optional[BaseModel] = None
    resources: Optional[AppResources] = None


class PostStepPayload(BaseModel):
    event_name: Literal["post_step"]
    step_result: StepResult
    pipeline_context: Optional[BaseModel] = None
    resources: Optional[AppResources] = None


class OnStepFailurePayload(BaseModel):
    event_name: Literal["on_step_failure"]
    step_result: StepResult
    pipeline_context: Optional[BaseModel] = None
    resources: Optional[AppResources] = None


HookPayload = Union[
    PreRunPayload,
    PostRunPayload,
    PreStepPayload,
    PostStepPayload,
    OnStepFailurePayload,
]
