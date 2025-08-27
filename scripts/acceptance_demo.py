"""
Acceptance Demo Script for Conversational Loops (FSD-033)

Usage:
  1) Generate a conversational loop via the wizard or use an existing pipeline.yaml
  2) Run the pipeline once to reach a HITL pause (uv run flujo run --input "Goal")
  3) Execute this script to resume with a human response

Example:
  uv run python scripts/acceptance_demo.py --run-id <run_id> --reply "Tomorrow"
"""

from __future__ import annotations

import argparse
import asyncio

from flujo.application.runner import Flujo
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.models import PipelineContext, PipelineResult
from flujo.application.core.state_manager import StateManager
from flujo.state.backends.sqlite import SQLiteBackend


async def resume_run(
    run_id: str, reply: str, pipeline_path: str = "pipeline.yaml"
) -> PipelineResult[PipelineContext]:
    pipeline = Pipeline.from_yaml_file(pipeline_path)
    runner = Flujo(pipeline=pipeline)
    backend = SQLiteBackend()
    sm: StateManager[PipelineContext] = StateManager(backend)
    ctx, last_output, idx, created_at, pname, pver, step_history = await sm.load_workflow_state(
        run_id, PipelineContext
    )
    if ctx is None:
        raise SystemExit(f"No persisted context found for run_id={run_id}")
    paused = PipelineResult(step_history=step_history, final_pipeline_context=ctx)
    final = await runner.resume_async(paused, human_input=reply)
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resume a paused conversational run")
    parser.add_argument("--run-id", required=True, help="Run ID to resume")
    parser.add_argument("--reply", required=True, help="Human reply to resume with")
    parser.add_argument("--pipeline", default="pipeline.yaml", help="Path to pipeline.yaml")
    args = parser.parse_args(argv)
    final = asyncio.run(resume_run(args.run_id, args.reply, pipeline_path=args.pipeline))
    print("Final success:", final.success)
    return 0 if final.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
