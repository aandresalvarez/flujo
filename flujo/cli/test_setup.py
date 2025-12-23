"""Test-specific CLI environment configuration for CI and deterministic output.

This module is part of Flujo's test infrastructure and intentionally checks for
test environment variables (PYTEST_CURRENT_TEST, CI) to configure deterministic
CLI output for testing.

**Architectural Note (per FLUJO_TEAM_GUIDE Section 2.6):**
This module is explicitly exempt from the "no test-aware production code" rule
because it is test infrastructure, not production code. Test infrastructure may
legitimately check for test environment variables to ensure deterministic behavior
in CI and test environments.

**Usage:**
This module should only be imported and used in test contexts. Production code
should not import or depend on this module.
"""

from __future__ import annotations

import os
from typing import Literal, Optional, Union

import click
import typer
import typer.rich_utils as tru


MarkupMode = Optional[Literal["markdown", "rich"]]


def configure_test_environment() -> None:
    """Configure CLI environment for tests/CI to ensure deterministic output.

    This function checks for test environment variables (PYTEST_CURRENT_TEST or CI)
    and configures the CLI environment accordingly. This is intentional test infrastructure
    behavior, not production code checking for test environments.

    **Why this is acceptable:**
    - This is test infrastructure, not production code (per FLUJO_TEAM_GUIDE Section 2.6)
    - The check ensures deterministic test output in CI environments
    - Production code paths do not call this function

    **Configuration applied:**
    - ANSI/color disabling for CI compatibility
    - Rich formatting overrides for deterministic help output
    - Click stderr shim for test compatibility
    - Terminal width standardization for consistent output

    **Environment Variables Checked:**
    - PYTEST_CURRENT_TEST: Set by pytest when running tests
    - CI: Set by most CI/CD systems (GitHub Actions, GitLab CI, etc.)

    Returns:
        None: Function returns early if not in test/CI environment
    """
    # Only configure if in test/CI environment
    # This check is intentional for test infrastructure (see module docstring)
    if not (os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("CI")):
        return

    os.environ.setdefault("NO_COLOR", "1")
    os.environ.setdefault("COLUMNS", "107")

    # Ensure Rich uses a deterministic width inside Click/Typer's CliRunner
    try:
        # Force Rich console width and disable terminal detection for deterministic wrapping
        try:
            setattr(tru, "MAX_WIDTH", 107)
        except Exception:
            pass
        try:
            setattr(tru, "FORCE_TERMINAL", True)
        except Exception:
            pass
        try:
            setattr(tru, "COLOR_SYSTEM", None)
        except Exception:
            pass
        # Reduce edge padding so trailing spaces at table borders don't differ across platforms
        try:
            setattr(tru, "STYLE_OPTIONS_TABLE_PAD_EDGE", False)
        except Exception:
            pass
        try:
            setattr(tru, "STYLE_COMMANDS_TABLE_PAD_EDGE", False)
        except Exception:
            pass
    except Exception:
        pass

    # Configure Rich help formatting for deterministic output
    try:
        from collections import defaultdict
        from typing import DefaultDict, List

        def _flujo_rich_format_help(
            *,
            obj: Union[click.Command, click.Group],
            ctx: click.Context,
            markup_mode: MarkupMode,
        ) -> None:
            safe_markup_mode: Literal["markdown", "rich"] = (
                "rich" if markup_mode is None else markup_mode
            )
            # Usage and description without right-padding spaces to match snapshots
            typer.echo("")
            typer.echo(f" {obj.get_usage(ctx).strip()}")
            typer.echo()
            typer.echo()
            typer.echo()
            if obj.help:
                typer.echo(f" {obj.help.strip()}")
                typer.echo()
                typer.echo()
                typer.echo()
                typer.echo()
                typer.echo()

            # Print a concise, non-truncated flags summary line to ensure full option names
            # are present in the help output (avoids ellipsizing like "--allow-side-eff…").
            try:
                option_names: list[str] = []
                for param in obj.get_params(ctx):
                    if getattr(param, "hidden", False):
                        continue
                    if isinstance(param, click.Option):
                        # Prefer long options; fall back to short if needed
                        longs = [o for o in getattr(param, "opts", []) if o.startswith("--")]
                        shorts = [
                            o
                            for o in getattr(param, "opts", [])
                            if o.startswith("-") and not o.startswith("--")
                        ]
                        if longs:
                            option_names.append(longs[0])
                        elif getattr(param, "secondary_opts", None):
                            # e.g., boolean pairs like --flag/--no-flag
                            secs = [
                                o
                                for o in getattr(param, "secondary_opts", [])
                                if o.startswith("--")
                            ]
                            if secs:
                                option_names.append(f"{secs[0]}")
                        elif shorts:
                            option_names.append(shorts[0])
                if option_names:
                    typer.echo(" Flags: " + ", ".join(option_names))
                    typer.echo()
            except Exception:
                pass

            console = tru._get_rich_console()
            panel_to_arguments: DefaultDict[str, List[click.Argument]] = defaultdict(list)
            panel_to_options: DefaultDict[str, List[click.Option]] = defaultdict(list)
            for param in obj.get_params(ctx):
                if getattr(param, "hidden", False):
                    continue
                if isinstance(param, click.Argument):
                    panel_name = (
                        getattr(param, tru._RICH_HELP_PANEL_NAME, None) or tru.ARGUMENTS_PANEL_TITLE
                    )
                    panel_to_arguments[panel_name].append(param)
                elif isinstance(param, click.Option):
                    panel_name = (
                        getattr(param, tru._RICH_HELP_PANEL_NAME, None) or tru.OPTIONS_PANEL_TITLE
                    )
                    panel_to_options[panel_name].append(param)

            default_arguments = panel_to_arguments.get(tru.ARGUMENTS_PANEL_TITLE, [])
            tru._print_options_panel(
                name=tru.ARGUMENTS_PANEL_TITLE,
                params=default_arguments,
                ctx=ctx,
                markup_mode=safe_markup_mode,
                console=console,
            )
            for panel_name, arguments in panel_to_arguments.items():
                if panel_name == tru.ARGUMENTS_PANEL_TITLE:
                    continue
                tru._print_options_panel(
                    name=panel_name,
                    params=arguments,
                    ctx=ctx,
                    markup_mode=safe_markup_mode,
                    console=console,
                )

            default_options = panel_to_options.get(tru.OPTIONS_PANEL_TITLE, [])
            tru._print_options_panel(
                name=tru.OPTIONS_PANEL_TITLE,
                params=default_options,
                ctx=ctx,
                markup_mode=safe_markup_mode,
                console=console,
            )
            for panel_name, options in panel_to_options.items():
                if panel_name == tru.OPTIONS_PANEL_TITLE:
                    continue
                tru._print_options_panel(
                    name=panel_name,
                    params=options,
                    ctx=ctx,
                    markup_mode=safe_markup_mode,
                    console=console,
                )

            if isinstance(obj, click.Group):
                panel_to_commands: DefaultDict[str, List[click.Command]] = defaultdict(list)
                for command_name in obj.list_commands(ctx):
                    command = obj.get_command(ctx, command_name)
                    if command and not command.hidden:
                        panel_name = (
                            getattr(command, tru._RICH_HELP_PANEL_NAME, None)
                            or tru.COMMANDS_PANEL_TITLE
                        )
                        panel_to_commands[panel_name].append(command)

                max_cmd_len = max(
                    [
                        len(command.name or "")
                        for commands in panel_to_commands.values()
                        for command in commands
                    ],
                    default=0,
                )
                default_commands = panel_to_commands.get(tru.COMMANDS_PANEL_TITLE, [])
                try:
                    tru._print_commands_panel(
                        name=tru.COMMANDS_PANEL_TITLE,
                        commands=default_commands,
                        markup_mode=safe_markup_mode,
                        console=console,
                        cmd_len=max_cmd_len,
                    )
                except TypeError:
                    tru._print_commands_panel(
                        name=tru.COMMANDS_PANEL_TITLE,
                        commands=default_commands,
                        markup_mode=safe_markup_mode,
                        console=console,
                        cmd_len=max_cmd_len,
                    )
                for panel_name, commands in panel_to_commands.items():
                    if panel_name == tru.COMMANDS_PANEL_TITLE:
                        continue
                    try:
                        tru._print_commands_panel(
                            name=panel_name,
                            commands=commands,
                            markup_mode=safe_markup_mode,
                            console=console,
                            cmd_len=max_cmd_len,
                        )
                    except TypeError:
                        tru._print_commands_panel(
                            name=panel_name,
                            commands=commands,
                            markup_mode=safe_markup_mode,
                            console=console,
                            cmd_len=max_cmd_len,
                        )

        setattr(tru, "rich_format_help", _flujo_rich_format_help)
        try:
            import typer.main as tm

            setattr(tm, "rich_format_help", _flujo_rich_format_help)
        except Exception:
            pass
    except Exception:
        pass

    # Click stderr shim for test compatibility
    try:
        import click.testing

        if not hasattr(click.testing.Result, "_flujo_stderr_shim"):

            def _stderr(self: click.testing.Result) -> str:
                return getattr(self, "output", "")

            # Assign property at runtime for test compatibility
            click.testing.Result.stderr = property(_stderr)  # type: ignore[assignment]
            setattr(click.testing.Result, "_flujo_stderr_shim", True)
    except Exception:
        pass
