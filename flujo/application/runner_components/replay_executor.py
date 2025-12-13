from __future__ import annotations

from typing import Any, Awaitable, Callable, Generic, List, TypeVar

from ...domain.models import PipelineContext, PipelineResult
from ...exceptions import ReplayError
from ...testing.replay import ReplayAgent
from ...type_definitions.common import JSONObject

_CtxT = TypeVar("_CtxT", bound=PipelineContext)


class ReplayExecutor(Generic[_CtxT]):
    """Replays a prior run deterministically using recorded trace and responses."""

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    async def replay_from_trace(self, run_id: str) -> PipelineResult[_CtxT]:
        runner = self._runner
        if runner.state_backend is None:
            raise ReplayError("Replay requires a state_backend with trace support")

        stored = await runner.state_backend.get_run_details(run_id)
        steps = await runner.state_backend.list_run_steps(run_id)
        trace = await runner.state_backend.get_trace(run_id)

        if stored is None:
            raise ReplayError(f"No stored run metadata for run_id={run_id}")
        if steps is None:
            steps = []

        initial_input: Any = None
        initial_context_data: JSONObject = {}
        try:
            if isinstance(trace, dict):
                attrs = trace.get("attributes", {}) if trace else {}
                initial_input = attrs.get("flujo.input", None)
        except Exception:
            initial_input = None
        try:
            loaded_state = await runner.state_backend.load_state(run_id)
            if loaded_state is not None:
                initial_context_data = loaded_state.get("pipeline_context") or {}
        except Exception:
            initial_context_data = {}

        response_map = self._build_response_map(steps)
        human_inputs = self._collect_human_inputs(trace)

        runner._ensure_pipeline()
        assert runner.pipeline is not None
        for st in runner.pipeline.steps:
            try:
                setattr(st, "agent", ReplayAgent(response_map))
            except Exception:
                pass

        original_resume = runner.resume_async
        runner.resume_async = self._patched_resume(original_resume, human_inputs)

        try:
            final_result = await self._run_with_replay(runner, initial_input, initial_context_data)
            final_result = await self._drain_pauses(
                runner,
                final_result,
                human_inputs,
                original_resume,
            )
            return final_result
        finally:
            runner.resume_async = original_resume

    def _build_response_map(self, steps: List[JSONObject]) -> JSONObject:
        response_map: JSONObject = {}
        for s in steps:
            step_name = s.get("step_name", "")
            key = f"{step_name}:attempt_1"
            raw_resp = s.get("raw_response")
            if raw_resp is None:
                raw_resp = s.get("output")
            response_map[key] = raw_resp
        return response_map

    def _collect_human_inputs(self, trace: Any) -> List[Any]:
        human_inputs: list[Any] = []

        def _collect_events(span: JSONObject) -> None:
            try:
                for ev in span.get("events", []) or []:
                    if ev.get("name") == "flujo.resumed":
                        human_inputs.append(ev.get("attributes", {}).get("human_input"))
                for ch in span.get("children", []) or []:
                    _collect_events(ch)
            except Exception:
                pass

        if isinstance(trace, dict):
            _collect_events(trace)
        return human_inputs

    def _patched_resume(
        self,
        original_resume: Callable[[PipelineResult[_CtxT], Any], Awaitable[PipelineResult[_CtxT]]],
        human_inputs: List[Any],
    ) -> Callable[[PipelineResult[_CtxT], Any], Awaitable[PipelineResult[_CtxT]]]:
        async def _resume_patched(
            paused_result: PipelineResult[_CtxT], human_input: Any
        ) -> PipelineResult[_CtxT]:
            if not human_inputs:
                raise ReplayError("ReplayError: no recorded human input available for resume")
            next_input = human_inputs.pop(0)
            return await original_resume(paused_result, next_input)

        return _resume_patched

    async def _run_with_replay(
        self,
        runner: Any,
        initial_input: Any,
        initial_context_data: JSONObject,
    ) -> PipelineResult[_CtxT]:
        final_result: PipelineResult[_CtxT] | None = None
        async for item in runner.run_async(
            initial_input, initial_context_data=initial_context_data
        ):
            final_result = item
        if final_result is None:
            raise ReplayError("ReplayError: run_async produced no result")
        return final_result

    async def _drain_pauses(
        self,
        runner: Any,
        final_result: PipelineResult[_CtxT],
        human_inputs: List[Any],
        original_resume: Any,
    ) -> PipelineResult[_CtxT]:
        from ...domain.models import PipelineContext as _PipelineContext

        while True:
            ctx = getattr(final_result, "final_pipeline_context", None)
            is_paused = False
            if _PipelineContext is not None and isinstance(ctx, _PipelineContext):
                status = getattr(ctx, "status", None)
                is_paused = status == "paused"
            if not is_paused:
                break
            if not human_inputs:
                raise ReplayError("ReplayError: no recorded human input available for resume")
            next_human_input = human_inputs.pop(0)
            final_result = await original_resume(final_result, next_human_input)
        return final_result
