"""Lockfile CLI commands for Flujo."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, List, Optional

import typer
from typing_extensions import Annotated

from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.models import PipelineResult
from flujo.infra.config_manager import get_cli_defaults
from flujo.infra.lockfile import (
    build_lockfile_data,
    compare_lockfiles,
    compute_lockfile_hash,
    format_lockfile_diff,
    load_lockfile,
    write_lockfile,
)
from flujo.type_definitions.common import JSONObject

from .exit_codes import EX_RUNTIME_ERROR, EX_VALIDATION_FAILED
from .helpers import find_project_root, load_pipeline_from_yaml_file, print_rich_or_typer

lock_app = typer.Typer(
    name="lock",
    help="Generate, verify, and compare Flujo lockfiles",
    rich_markup_mode="markdown",
)


def _create_dummy_result() -> PipelineResult[Any]:
    """Create a dummy PipelineResult for lockfile generation without execution.

    Returns:
        PipelineResult with minimal data for lockfile generation
    """
    return PipelineResult(
        success=True,
        output=None,
        step_history=[],
        status="completed",
    )


def _get_pipeline_metadata(
    pipeline: Pipeline[Any, Any],
    pipeline_file: Optional[str] = None,
) -> tuple[str, str, str]:
    """Extract pipeline metadata (name, version, id) from pipeline or defaults.

    Args:
        pipeline: The pipeline object
        pipeline_file: Optional path to pipeline file for fallback naming

    Returns:
        Tuple of (pipeline_name, pipeline_version, pipeline_id)
    """
    # Try to get name from pipeline attribute (set by YAML loader)
    pipeline_name: Optional[str] = getattr(pipeline, "name", None)
    if not pipeline_name and pipeline_file:
        # Fallback to filename without extension
        pipeline_name = Path(pipeline_file).stem

    if not pipeline_name:
        pipeline_name = "pipeline"

    # Version defaults to "0.1" (YAML default)
    pipeline_version: str = getattr(pipeline, "version", "0.1")

    # Generate a stable ID based on pipeline name
    pipeline_id: str = str(uuid.uuid5(uuid.NAMESPACE_DNS, pipeline_name))

    return pipeline_name, pipeline_version, pipeline_id


def _resolve_external_files(
    external_files: Optional[List[str]],
    project_root: Path,
    config: JSONObject,
) -> List[Path]:
    """Resolve external file paths from CLI and config.

    Args:
        external_files: CLI-provided file paths
        project_root: Project root directory for resolving relative paths
        config: Configuration from ConfigManager

    Returns:
        List of resolved Path objects
    """
    resolved: List[Path] = []

    # Get from config
    config_files: List[str] = config.get("external_files", [])
    if isinstance(config_files, list):
        for file_pattern in config_files:
            if isinstance(file_pattern, str):
                pattern_path = Path(file_pattern)
                if not pattern_path.is_absolute():
                    pattern_path = project_root / pattern_path
                resolved.append(pattern_path)

    # Add CLI-provided files (takes precedence)
    if external_files:
        for file_pattern in external_files:
            pattern_path = Path(file_pattern)
            if not pattern_path.is_absolute():
                pattern_path = project_root / pattern_path
            resolved.append(pattern_path)

    return resolved


@lock_app.command("generate")
def generate(
    pipeline: Annotated[
        Optional[str],
        typer.Option("--pipeline", help="Path to pipeline.yaml (default: project pipeline.yaml)"),
    ] = None,
    output: Annotated[
        Optional[str],
        typer.Option("--output", "-o", help="Output lockfile path (default: ./flujo.lock)"),
    ] = None,
    include_external: Annotated[
        Optional[List[str]],
        typer.Option(
            "--include-external",
            help="Additional external files to hash (repeatable, supports glob patterns)",
        ),
    ] = None,
    model_info: Annotated[
        bool,
        typer.Option(
            "--model-info/--no-model-info", help="Include model versions and hyperparameters"
        ),
    ] = True,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite existing lockfile"),
    ] = False,
) -> None:
    """Generate a lockfile from pipeline definition without executing it.

    Examples:
        flujo lock generate
        flujo lock generate --include-external data/*.json --include-external schemas/api.yaml
        flujo lock generate --output custom.lock --no-model-info
    """
    try:
        # Find project root
        project_root: Path = find_project_root()

        # Load configuration
        config: JSONObject = get_cli_defaults("lock")
        if not isinstance(config, dict):
            config = {}

        # Determine pipeline file
        if pipeline is None:
            pipeline_file: str = str((project_root / "pipeline.yaml").resolve())
        else:
            pipeline_file = pipeline

        if not Path(pipeline_file).exists():
            print_rich_or_typer(
                f"[red]Pipeline file not found: {pipeline_file}[/red]",
                stderr=True,
            )
            raise typer.Exit(EX_RUNTIME_ERROR)

        # Load pipeline
        pipeline_obj: Pipeline[Any, Any] = load_pipeline_from_yaml_file(pipeline_file)

        # Get pipeline metadata
        pipeline_name, pipeline_version, pipeline_id = _get_pipeline_metadata(
            pipeline_obj, pipeline_file
        )

        # Resolve external files
        external_files_list: List[Path] = _resolve_external_files(
            include_external, project_root, config
        )

        # Determine output path
        if output is None:
            output_path: Path = project_root / "flujo.lock"
        else:
            output_path = Path(output)

        # Check if file exists
        if output_path.exists() and not force:
            print_rich_or_typer(
                f"[yellow]Lockfile already exists: {output_path}[/yellow]\n"
                "Use --force to overwrite.",
                stderr=True,
            )
            raise typer.Exit(EX_RUNTIME_ERROR)

        # Create dummy result for lockfile generation
        dummy_result: PipelineResult[Any] = _create_dummy_result()

        # Generate lockfile
        write_lockfile(
            path=output_path,
            pipeline=pipeline_obj,
            result=dummy_result,
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
            pipeline_id=pipeline_id,
            run_id=None,
            external_files=external_files_list if external_files_list else None,
            include_model_info=model_info,
            project_root=project_root,
        )

        print_rich_or_typer(f"[green]Lockfile generated: {output_path}[/green]")

    except typer.Exit:
        raise
    except Exception as e:
        print_rich_or_typer(
            f"[red]Failed to generate lockfile: {type(e).__name__}: {e}[/red]",
            stderr=True,
        )
        raise typer.Exit(EX_RUNTIME_ERROR) from e


@lock_app.command("verify")
def verify(
    lockfile: Annotated[
        Optional[str],
        typer.Option("--lockfile", "-l", help="Path to lockfile (default: ./flujo.lock)"),
    ] = None,
    pipeline: Annotated[
        Optional[str],
        typer.Option("--pipeline", help="Path to pipeline.yaml (default: project pipeline.yaml)"),
    ] = None,
    strict: Annotated[
        bool,
        typer.Option("--strict/--no-strict", help="Exit non-zero on any difference"),
    ] = False,
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            "--output-format",
            help="Output format: text, json (default: text)",
        ),
    ] = "text",
    ignore_fields: Annotated[
        Optional[str],
        typer.Option(
            "--ignore-fields",
            help="Comma-separated fields to ignore (e.g., 'run.timestamp')",
        ),
    ] = None,
) -> None:
    """Verify that current pipeline matches the lockfile.

    Examples:
        flujo lock verify
        flujo lock verify --strict
        flujo lock verify --ignore-fields run.timestamp
    """
    try:
        # Find project root
        project_root: Path = find_project_root()

        # Determine lockfile path
        if lockfile is None:
            lockfile_path: Path = project_root / "flujo.lock"
        else:
            lockfile_path = Path(lockfile)

        if not lockfile_path.exists():
            print_rich_or_typer(
                f"[red]Lockfile not found: {lockfile_path}[/red]",
                stderr=True,
            )
            raise typer.Exit(EX_VALIDATION_FAILED)

        # Determine pipeline file
        if pipeline is None:
            pipeline_file: str = str((project_root / "pipeline.yaml").resolve())
        else:
            pipeline_file = pipeline

        if not Path(pipeline_file).exists():
            print_rich_or_typer(
                f"[red]Pipeline file not found: {pipeline_file}[/red]",
                stderr=True,
            )
            raise typer.Exit(EX_RUNTIME_ERROR)

        # Load existing lockfile
        existing_lockfile: JSONObject = load_lockfile(lockfile_path)

        # Load pipeline
        pipeline_obj: Pipeline[Any, Any] = load_pipeline_from_yaml_file(pipeline_file)

        # Get pipeline metadata
        pipeline_name, pipeline_version, pipeline_id = _get_pipeline_metadata(
            pipeline_obj, pipeline_file
        )

        # Load configuration
        config: JSONObject = get_cli_defaults("lock")
        if not isinstance(config, dict):
            config = {}

        # Resolve external files
        external_files_list: List[Path] = _resolve_external_files(None, project_root, config)

        # Generate temporary lockfile from current pipeline
        dummy_result: PipelineResult[Any] = _create_dummy_result()
        current_lockfile: JSONObject = build_lockfile_data(
            pipeline=pipeline_obj,
            result=dummy_result,
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
            pipeline_id=pipeline_id,
            run_id=None,
            external_files=external_files_list if external_files_list else None,
            include_model_info=config.get("include_model_info", True),
            project_root=project_root,
        )

        # Parse ignore fields
        ignore_list: Optional[List[str]] = None
        if ignore_fields:
            ignore_list = [f.strip() for f in ignore_fields.split(",") if f.strip()]

        # Compare lockfiles
        diff = compare_lockfiles(existing_lockfile, current_lockfile, ignore_fields=ignore_list)

        # Output results
        if output_format == "json":
            output_data: JSONObject = {
                "has_differences": diff.has_differences,
                "prompts_changed": diff.prompts_changed,
                "prompts_added": diff.prompts_added,
                "prompts_removed": diff.prompts_removed,
                "skills_changed": diff.skills_changed,
                "skills_added": diff.skills_added,
                "skills_removed": diff.skills_removed,
                "models_changed": diff.models_changed,
                "models_added": diff.models_added,
                "models_removed": diff.models_removed,
                "external_files_changed": diff.external_files_changed,
                "external_files_added": diff.external_files_added,
                "external_files_removed": diff.external_files_removed,
            }
            typer.echo(json.dumps(output_data, indent=2))
        else:
            formatted_diff: str = format_lockfile_diff(diff)
            if diff.has_differences:
                print_rich_or_typer(f"[yellow]{formatted_diff}[/yellow]", stderr=True)
            else:
                print_rich_or_typer(
                    "[green]Lockfile verification passed: no differences found.[/green]"
                )

        # Exit with appropriate code
        if diff.has_differences and strict:
            raise typer.Exit(EX_VALIDATION_FAILED)

    except typer.Exit:
        raise
    except Exception as e:
        print_rich_or_typer(
            f"[red]Failed to verify lockfile: {type(e).__name__}: {e}[/red]",
            stderr=True,
        )
        raise typer.Exit(EX_RUNTIME_ERROR) from e


@lock_app.command("compare")
def compare(
    lockfile1: Annotated[str, typer.Argument(help="Path to first lockfile")],
    lockfile2: Annotated[str, typer.Argument(help="Path to second lockfile")],
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            "--output-format",
            help="Output format: text, json, markdown (default: text)",
        ),
    ] = "text",
    detailed: Annotated[
        bool,
        typer.Option("--detailed/--no-detailed", help="Show full hashes and detailed diffs"),
    ] = False,
    ignore_fields: Annotated[
        Optional[str],
        typer.Option(
            "--ignore-fields",
            help="Comma-separated fields to ignore (e.g., 'run.timestamp')",
        ),
    ] = None,
) -> None:
    """Compare two lockfiles and show differences.

    Examples:
        flujo lock compare flujo.lock flujo.lock.backup
        flujo lock compare flujo.lock flujo.lock.backup --format json --detailed
    """
    try:
        # Load both lockfiles
        lockfile1_data: JSONObject = load_lockfile(lockfile1)
        lockfile2_data: JSONObject = load_lockfile(lockfile2)

        # Parse ignore fields
        ignore_list: Optional[List[str]] = None
        if ignore_fields:
            ignore_list = [f.strip() for f in ignore_fields.split(",") if f.strip()]

        # Compare
        diff = compare_lockfiles(lockfile1_data, lockfile2_data, ignore_fields=ignore_list)

        # Output results
        if output_format == "json":
            output_data: JSONObject = {
                "has_differences": diff.has_differences,
                "prompts_changed": diff.prompts_changed,
                "prompts_added": diff.prompts_added,
                "prompts_removed": diff.prompts_removed,
                "skills_changed": diff.skills_changed,
                "skills_added": diff.skills_added,
                "skills_removed": diff.skills_removed,
                "models_changed": diff.models_changed,
                "models_added": diff.models_added,
                "models_removed": diff.models_removed,
                "external_files_changed": diff.external_files_changed,
                "external_files_added": diff.external_files_added,
                "external_files_removed": diff.external_files_removed,
            }
            if detailed:
                output_data["lockfile1_hash"] = compute_lockfile_hash(lockfile1_data)
                output_data["lockfile2_hash"] = compute_lockfile_hash(lockfile2_data)
            typer.echo(json.dumps(output_data, indent=2))
        elif output_format == "markdown":
            lines: List[str] = ["# Lockfile Comparison\n"]
            if diff.has_differences:
                lines.append("## Differences Found\n")
                lines.append(format_lockfile_diff(diff))
            else:
                lines.append("## No Differences\n")
                lines.append("The lockfiles are identical.")
            if detailed:
                lines.append("\n## Hashes\n")
                lines.append(f"- Lockfile 1: `{compute_lockfile_hash(lockfile1_data)}`")
                lines.append(f"- Lockfile 2: `{compute_lockfile_hash(lockfile2_data)}`")
            typer.echo("\n".join(lines))
        else:
            # Text format
            formatted_diff: str = format_lockfile_diff(diff)
            typer.echo(formatted_diff)
            if detailed:
                typer.echo(f"\nLockfile 1 hash: {compute_lockfile_hash(lockfile1_data)}")
                typer.echo(f"Lockfile 2 hash: {compute_lockfile_hash(lockfile2_data)}")

    except typer.Exit:
        raise
    except Exception as e:
        print_rich_or_typer(
            f"[red]Failed to compare lockfiles: {type(e).__name__}: {e}[/red]",
            stderr=True,
        )
        raise typer.Exit(EX_RUNTIME_ERROR) from e


@lock_app.command("show")
def show(
    lockfile: Annotated[
        Optional[str],
        typer.Option("--lockfile", "-l", help="Path to lockfile (default: ./flujo.lock)"),
    ] = None,
    format: Annotated[
        str,
        typer.Option(
            "--format",
            help="Output format: table, json, yaml (default: table)",
        ),
    ] = "table",
) -> None:
    """Display lockfile information in a human-readable format.

    Examples:
        flujo lock show
        flujo lock show --format json
        flujo lock show --lockfile custom.lock --format yaml
    """
    try:
        # Find project root
        project_root: Path = find_project_root()

        # Determine lockfile path
        if lockfile is None:
            lockfile_path: Path = project_root / "flujo.lock"
        else:
            lockfile_path = Path(lockfile)

        if not lockfile_path.exists():
            print_rich_or_typer(
                f"[red]Lockfile not found: {lockfile_path}[/red]",
                stderr=True,
            )
            raise typer.Exit(EX_RUNTIME_ERROR)

        # Load lockfile
        lockfile_data: JSONObject = load_lockfile(lockfile_path)

        # Output based on format
        if format == "json":
            typer.echo(json.dumps(lockfile_data, indent=2))
        elif format == "yaml":
            try:
                import yaml

                typer.echo(yaml.safe_dump(lockfile_data, default_flow_style=False))
            except ImportError:
                print_rich_or_typer(
                    "[red]YAML output requires PyYAML. Install with: pip install pyyaml[/red]",
                    stderr=True,
                )
                raise typer.Exit(EX_RUNTIME_ERROR) from None
        else:
            # Table format
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(title=f"Lockfile: {lockfile_path.name}")

            table.add_column("Field", style="cyan")
            table.add_column("Value", style="green")

            # Pipeline info
            pipeline_info: JSONObject = lockfile_data.get("pipeline", {})
            table.add_row("Pipeline Name", str(pipeline_info.get("name", "N/A")))
            table.add_row("Pipeline Version", str(pipeline_info.get("version", "N/A")))
            table.add_row("Pipeline ID", str(pipeline_info.get("id", "N/A")))
            table.add_row("Schema Version", str(lockfile_data.get("schema_version", 1)))

            # Counts
            prompts: List[JSONObject] = lockfile_data.get("prompts", [])
            skills: List[JSONObject] = lockfile_data.get("skills", [])
            models: List[JSONObject] = lockfile_data.get("models", [])
            external_files: List[JSONObject] = lockfile_data.get("external_files", [])

            table.add_row("Prompts", str(len(prompts)))
            table.add_row("Skills", str(len(skills)))
            if models:
                table.add_row("Models", str(len(models)))
            if external_files:
                table.add_row("External Files", str(len(external_files)))

            # Run info
            run_info: JSONObject = lockfile_data.get("run", {})
            if run_info.get("run_id"):
                table.add_row("Run ID", str(run_info.get("run_id")))
            if run_info.get("timestamp"):
                table.add_row("Timestamp", str(run_info.get("timestamp")))

            console.print(table)

    except typer.Exit:
        raise
    except Exception as e:
        print_rich_or_typer(
            f"[red]Failed to show lockfile: {type(e).__name__}: {e}[/red]",
            stderr=True,
        )
        raise typer.Exit(EX_RUNTIME_ERROR) from e


__all__ = ["lock_app"]
