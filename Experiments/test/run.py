from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from flujo.cli.helpers import (
    create_flujo_runner,
    display_pipeline_results,
    execute_pipeline_with_output_handling,
    load_pipeline_from_file,
    load_pipeline_from_yaml_file,
)


def _resolve_pipeline(root: Path, pipeline_path: Path):
    if pipeline_path.suffix in {".yaml", ".yml"}:
        return load_pipeline_from_yaml_file(str(pipeline_path))
    pipeline_obj, _ = load_pipeline_from_file(str(pipeline_path))
    return pipeline_obj


def run_pipeline(pipeline_name: str, goal: Optional[str]) -> None:
    root = Path(__file__).resolve().parent
    pipeline_path = root / "pipelines" / pipeline_name
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")

    pipeline = _resolve_pipeline(root, pipeline_path)
    runner = create_flujo_runner(
        pipeline,
        context_model_class=None,
        initial_context_data=None,
    )

    result = execute_pipeline_with_output_handling(
        runner=runner,
        input_data=goal or "",
        run_id=None,
        json_output=False,
    )

    display_pipeline_results(
        result=result,
        run_id=None,
        json_output=False,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run evidence-first experimental pipelines with Flujo.",
    )
    parser.add_argument(
        "--pipeline",
        default="score_free_controller.py",
        help="Pipeline file under Experiments/test/pipelines (YAML or Python).",
    )
    parser.add_argument(
        "--goal",
        default=None,
        help="Initial goal or prompt to feed into the pipeline.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    run_pipeline(args.pipeline, args.goal)


if __name__ == "__main__":
    main()
