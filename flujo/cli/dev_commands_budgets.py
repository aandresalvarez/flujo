from __future__ import annotations

import typer

from .helpers import print_rich_or_typer


def budgets_show(pipeline_name: str) -> None:
    """Print the effective budget for a pipeline and its resolution source."""
    try:
        from flujo.infra.config_manager import ConfigManager
        from flujo.infra.budget_resolver import resolve_limits_for_pipeline

        cfg = ConfigManager().load_config()
        limits, src = resolve_limits_for_pipeline(getattr(cfg, "budgets", None), pipeline_name)

        if limits is None:
            typer.echo("No budget configured (unlimited). Source: none")
            return

        cost = (
            f"${limits.total_cost_usd_limit:.2f}"
            if limits.total_cost_usd_limit is not None
            else "unlimited"
        )
        tokens = (
            f"{limits.total_tokens_limit}" if limits.total_tokens_limit is not None else "unlimited"
        )
        origin = src.source if src.pattern is None else f"{src.source}[{src.pattern}]"
        typer.echo(f"Effective budget for '{pipeline_name}':")
        typer.echo(f"  - total_cost_usd_limit: {cost}")
        typer.echo(f"  - total_tokens_limit: {tokens}")
        typer.echo(f"Resolved from {origin} in flujo.toml")
    except Exception as e:  # noqa: BLE001
        print_rich_or_typer(f"[red]Failed to resolve budgets: {e}", stderr=True)
        raise typer.Exit(1) from e


def register_budget_commands(budgets_app: typer.Typer) -> None:
    budgets_app.command("show")(budgets_show)
