"""CLI app construction and command registration."""

from __future__ import annotations

import typer

from . import dev_commands
from .lens import register_lens_app
from .lock_command import lock_app
from .project_command import demo as demo_cmd, init as init_cmd
from .run_command import run as run_cmd
from .status_command import status as status_cmd
from .validate_command import validate as validate_cmd
from .architect_command import create as create_cmd
from .migrate_command import migrate as migrate_cmd


def create_cli_app() -> typer.Typer:
    """Create the main CLI Typer app instance.

    Returns:
        Configured Typer app with help text but no commands registered yet.
    """
    return typer.Typer(
        rich_markup_mode="markdown",
        help=(
            "A project-based server to build, run, and debug Flujo AI workflows.\n\n"
            "Common Commands:\n"
            "- `init` / `demo`: scaffold a project or a demo\n"
            "- `run`: run the current project's pipeline (YAML or Python)\n"
            "- `validate`: validate a pipeline file\n"
            "- `lens`: inspect past runs (`list`, `show`, `trace`, `from-file`)\n"
            "- `dev`: developer tools (`version`, `show-config`, `validate`, `visualize`, `budgets`)\n\n"
            "Debugging Flags for `run`:\n"
            "- `--debug`: step-by-step trace tree (safe previews)\n"
            "- `--trace-preview-len N`: preview size for prompts/responses\n"
            "- `--debug-prompts`: include full prompts/responses in trace (unsafe)\n"
            "- `--debug-export PATH`: write full debug JSON (trace + steps + context).\n"
            "  If omitted with `--debug`, auto-writes to `./debug/<timestamp>_<run_id>.json`.\n\n"
            "Visualization Flags for `run`:\n"
            "- `--live`/`--progress`: live step panels (compact)\n"
            "- `--summary`: only final output, totals, and run id\n"
            "- `--no-steps` / `--no-context`: hide table/context\n"
            "- `--output-preview-len N`: trim table outputs (default 100)\n"
            "- `--final-output-format`: auto|raw|json|yaml|md\n"
            "- `--pager`: page long output in terminal\n\n"
            "Project Root: pass `--project` or set `FLUJO_PROJECT_ROOT`.\n"
            "Verbose Errors: add `-v`/`--verbose` or `--trace`.\n"
            "Stable exit codes: see flujo.cli.exit_codes."
        ),
    )


def register_all_commands(app: typer.Typer) -> None:
    """Register all CLI commands and sub-apps with the main app.

    Args:
        app: The main Typer app instance to register commands with.
    """
    # Register lens app
    register_lens_app(app)

    # Register lock app
    app.add_typer(lock_app, name="lock")

    # Register top-level commands
    app.command(name="run", help="🚀 Run the workflow in the current project.")(run_cmd)
    app.command(name="status", help="Show provider readiness and SQLite state configuration.")(
        status_cmd
    )
    app.command(name="migrate", help="Apply database migrations for the state backend.")(
        migrate_cmd
    )
    app.command(name="validate")(validate_cmd)
    app.command(name="init", help="Initialize a new Flujo project in the current directory.")(
        init_cmd
    )
    app.command(name="demo", help="Create a demo Flujo project in the current directory.")(demo_cmd)
    # Architect create command
    app.command(
        name="create",
        help=(
            "🤖 Start a conversation with the AI Architect to build your workflow.\n\n"
            "By default this uses the full conversational state machine. Set FLUJO_ARCHITECT_MINIMAL=1"
            " to use the legacy minimal generator.\n\n"
            "Tip: pass --allow-side-effects to permit pipelines that reference side-effect skills."
        ),
    )(create_cmd)

    # Create developer sub-app and nested experimental group
    dev_app: typer.Typer = typer.Typer(
        rich_markup_mode=None,
        help="🛠️  Access advanced developer and diagnostic tools (e.g., version, show-config, visualize).",
    )
    experimental_app: typer.Typer = typer.Typer(
        rich_markup_mode=None, help="(Advanced) Experimental and diagnostic commands."
    )
    dev_app.add_typer(experimental_app, name="experimental")

    # Budgets live under the dev group
    budgets_app: typer.Typer = typer.Typer(rich_markup_mode=None, help="Budget governance commands")
    dev_app.add_typer(budgets_app, name="budgets")

    # Register developer app at top level
    app.add_typer(dev_app, name="dev")

    # Register developer and experimental commands
    dev_commands.register_commands(dev_app, experimental_app, budgets_app)

    # Register validate command again (per FSD-021) with explicit help
    try:
        app.command(
            name="validate",
            help="✅ Validate the project's pipeline.yaml file.",
        )(validate_cmd)
    except Exception:
        pass
