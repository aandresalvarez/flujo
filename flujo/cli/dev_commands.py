from __future__ import annotations

import typer

from .dev_commands_budgets import register_budget_commands
from .dev_commands_dev import register_dev_commands
from .dev_commands_experimental import register_experimental_commands
from .dev_commands_health import register_health_commands

__all__ = ["register_commands"]


def register_commands(
    dev_app: typer.Typer, experimental_app: typer.Typer, budgets_app: typer.Typer
) -> None:
    register_health_commands(dev_app)
    register_dev_commands(dev_app)
    register_experimental_commands(experimental_app)
    register_budget_commands(budgets_app)
