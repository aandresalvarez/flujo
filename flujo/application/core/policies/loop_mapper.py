from __future__ import annotations

from typing import Protocol

from ._shared import (
    StepOutcome,
    StepResult,
    telemetry,
    time,
    to_outcome,
)
from .common import DefaultAgentResultUnpacker


class _NamedStep(Protocol):
    name: str


def map_initial_input(
    *,
    loop_step: _NamedStep,
    current_data: object,
    current_context: object | None,
    start_time: float,
) -> tuple[object, StepOutcome[StepResult] | None]:
    mapper = (
        loop_step.get_initial_input_to_loop_body_mapper()
        if hasattr(loop_step, "get_initial_input_to_loop_body_mapper")
        else getattr(loop_step, "initial_input_to_loop_body_mapper", None)
    )
    if not mapper:
        return current_data, None
    try:
        return mapper(current_data, current_context), None
    except Exception as e:
        sr = StepResult(
            name=loop_step.name,
            success=False,
            output=None,
            attempts=0,
            latency_s=time.monotonic() - start_time,
            token_counts=0,
            cost_usd=0.0,
            feedback=f"Error in initial_input_to_loop_body_mapper for LoopStep '{loop_step.name}': {e}",
            branch_context=current_context,
            metadata_={"iterations": 0, "exit_reason": "initial_input_mapper_error"},
            step_history=[],
        )
        return current_data, to_outcome(sr)


def apply_iteration_input_mapper(
    *,
    loop_step: _NamedStep,
    current_data: object,
    current_context: object | None,
    iteration_count: int,
    max_loops: int,
    start_time: float,
    cumulative_tokens: int,
    cumulative_cost: float,
    iteration_results_all: list[StepResult],
    iteration_results: list[StepResult],
) -> tuple[object, StepOutcome[StepResult] | None]:
    iter_mapper = (
        loop_step.get_iteration_input_mapper()
        if hasattr(loop_step, "get_iteration_input_mapper")
        else getattr(loop_step, "iteration_input_mapper", None)
    )
    if not iter_mapper or iteration_count > max_loops:
        return current_data, None
    try:
        mapped = iter_mapper(current_data, current_context, iteration_count - 1)
        return mapped, None
    except Exception as e:
        telemetry.logfire.error(
            f"Error in iteration_input_mapper for LoopStep '{loop_step.name}' at iteration {iteration_count}: {e}"
        )
        completed_before_mapper_error = iteration_count - 1
        sr = StepResult(
            name=loop_step.name,
            success=False,
            output=None,
            attempts=completed_before_mapper_error,
            latency_s=time.monotonic() - start_time,
            token_counts=cumulative_tokens,
            cost_usd=cumulative_cost,
            feedback=f"Error in iteration_input_mapper for LoopStep '{loop_step.name}': {e}",
            branch_context=current_context,
            metadata_={
                "iterations": completed_before_mapper_error,
                "exit_reason": "iteration_input_mapper_error",
            },
            step_history=iteration_results_all + iteration_results,
        )
        return current_data, to_outcome(sr)


def finalize_loop_output(
    *,
    loop_step: _NamedStep,
    core: object,
    current_data: object,
    final_output: object,
    current_context: object | None,
    iteration_count: int,
    cumulative_tokens: int,
    cumulative_cost: float,
    iteration_results_all: list[StepResult],
    is_map_step: bool,
    start_time: float,
) -> tuple[object, StepOutcome[StepResult] | None]:
    output_mapper = (
        loop_step.get_loop_output_mapper()
        if hasattr(loop_step, "get_loop_output_mapper")
        else getattr(loop_step, "loop_output_mapper", None)
    )
    if is_map_step:
        try:
            if hasattr(loop_step, "_results_var"):
                results = list(getattr(loop_step, "_results_var").get())
            else:
                results = list(getattr(loop_step, "results", []) or [])
        except Exception:
            results = list(getattr(loop_step, "results", []) or [])
        if current_data is not None:
            results.append(current_data)
        final_output = results
        output_mapper = None
        try:
            meta = getattr(loop_step, "meta", {})
            if isinstance(meta, dict):
                candidate = meta.get("map_finalize_mapper")
                if callable(candidate):
                    output_mapper = candidate
        except Exception:
            output_mapper = None
    if not output_mapper:
        return final_output, None
    try:
        try:
            unpacker = getattr(core, "unpacker", DefaultAgentResultUnpacker())
        except Exception:
            unpacker = DefaultAgentResultUnpacker()
        try:
            meta = getattr(loop_step, "meta", {})
            use_final = bool(
                is_map_step
                and isinstance(meta, dict)
                and meta.get("map_finalize_mapper") is not None
            )
        except Exception:
            use_final = False
        unpacked_data = unpacker.unpack(final_output if use_final else current_data)
        mapped = output_mapper(unpacked_data, current_context)
        try:
            if current_context is not None:
                try:
                    object.__setattr__(current_context, "_last_loop_iterations", iteration_count)
                except Exception:
                    setattr(current_context, "_last_loop_iterations", iteration_count)
        except Exception:
            pass
        return mapped, None
    except Exception as e:
        sr = StepResult(
            name=loop_step.name,
            success=False,
            output=None,
            attempts=iteration_count,
            latency_s=time.monotonic() - start_time,
            token_counts=cumulative_tokens,
            cost_usd=cumulative_cost,
            feedback=str(e),
            branch_context=current_context,
            metadata_={
                "iterations": iteration_count,
                "exit_reason": "loop_output_mapper_error",
            },
            step_history=iteration_results_all,
        )
        return final_output, to_outcome(sr)
