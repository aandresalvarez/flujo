from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

from typer import Exit


def apply_cli_defaults(command: str, **kwargs: Any) -> dict[str, Any]:
    """Apply CLI defaults from configuration file to command arguments."""
    from flujo.cli.main import get_cli_defaults

    try:
        import click as _click
    except Exception:
        _click = None  # type: ignore[assignment]

    cli_defaults = get_cli_defaults(command)
    result = kwargs.copy()

    explicitly_set: set[str] = set()
    if _click is not None:
        try:
            ctx = _click.get_current_context(silent=True)
            if ctx is not None and isinstance(getattr(ctx, "params", None), dict):
                explicitly_set = set(ctx.params.keys())
        except Exception:
            explicitly_set = set()

    for key in kwargs.keys():
        if key in explicitly_set:
            continue
        if key in cli_defaults and result.get(key, None) is None:
            result[key] = cli_defaults[key]

    return result


def _has_project_markers(dir_path: Path) -> bool:
    """Return True if the directory looks like a Flujo project root."""
    return (
        (dir_path / "flujo.toml").exists()
        or (dir_path / "pipeline.yaml").exists()
        or (dir_path / "pipeline.yml").exists()
    )


def resolve_project_root(
    explicit: Optional[Path] = None, allow_missing: bool = True
) -> Optional[Path]:
    """Resolve the project root using explicit path, env var, or marker search."""
    if explicit is not None:
        p = explicit.resolve()
        if not p.exists() or not p.is_dir():
            from typer import secho

            secho(f"Error: --project '{p}' does not exist or is not a directory", fg="red")
            raise Exit(1)
        return p

    env_root = os.environ.get("FLUJO_PROJECT_ROOT")
    if isinstance(env_root, str) and env_root.strip():
        p = Path(env_root).resolve()
        if p.exists() and p.is_dir():
            return p
        if not allow_missing:
            from typer import secho

            secho(
                f"Error: FLUJO_PROJECT_ROOT='{env_root}' does not exist or is not a directory",
                fg="red",
            )
            raise Exit(1)
        return None

    current = Path.cwd().resolve()
    while True:
        if _has_project_markers(current):
            return current
        if current.parent == current:
            if allow_missing:
                return None
            from typer import secho

            secho(
                "Error: Not a Flujo project. Please run 'flujo init' in your desired project directory first.",
                fg="red",
            )
            raise Exit(1)
        current = current.parent


def ensure_project_root_on_sys_path(root: Optional[Path]) -> None:
    """Add the given project root to sys.path if not already present."""
    try:
        if root is None:
            return
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
    except Exception:
        pass


def find_project_root(start: Optional[Path] = None) -> Path:
    """Find the Flujo project root by locating a flujo.toml or pipeline.yaml."""
    if start is not None:
        resolved = resolve_project_root(start, allow_missing=False)
        assert resolved is not None
        return resolved
    resolved = resolve_project_root(None, allow_missing=False)
    assert resolved is not None
    return resolved


def scaffold_project(directory: Path, *, overwrite_existing: bool = False) -> None:
    """Create a new Flujo project scaffold in the given directory."""
    from typer import secho

    directory = directory.resolve()
    flujo_toml = directory / "flujo.toml"
    hidden_dir = directory / ".flujo"
    skills_dir = directory / "skills"

    if (flujo_toml.exists() or hidden_dir.exists()) and not overwrite_existing:
        secho("Error: This directory already looks like a Flujo project.", fg="red")
        raise Exit(1)

    skills_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir.mkdir(parents=True, exist_ok=True)

    from importlib import resources as _res

    created: list[str] = []
    overwritten: list[str] = []

    def _write(path: Path, content: str) -> None:
        target = path
        existed = target.exists()
        target.write_text(content)
        rel = str(target.relative_to(directory)) if target.is_file() else target.name
        if existed:
            overwritten.append(rel)
        else:
            created.append(rel)

    try:
        template_pkg = "flujo.templates.project"
        with _res.files(template_pkg).joinpath("flujo.toml").open("r") as f:
            _write(flujo_toml, f.read())
        with _res.files(template_pkg).joinpath("pipeline.yaml").open("r") as f:
            _write(directory / "pipeline.yaml", f.read())
        try:
            with _res.files(template_pkg).joinpath(".env.example").open("r") as f:
                _write(directory / ".env.example", f.read())
        except Exception:
            pass
        with _res.files(template_pkg).joinpath("skills__init__.py").open("r") as f:
            _write(skills_dir / "__init__.py", f.read())
        with _res.files(template_pkg).joinpath("custom_tools.py").open("r") as f:
            _write(skills_dir / "custom_tools.py", f.read())
        try:
            with _res.files(template_pkg).joinpath("README.md").open("r") as f:
                _write(directory / "README.md", f.read())
        except Exception:
            pass
    except Exception:
        _write(
            flujo_toml,
            """
# Flujo project configuration

# State backend (choose one):
# - SQLite (recommended for local durability):
#     sqlite:///.flujo/state.db        # relative to project root
#     sqlite:////abs/path/to/ops.db    # absolute path
# - In-memory (ephemeral; no persistence across restarts):
#     memory://
# - Env var override (takes precedence over this file):
#     export FLUJO_STATE_URI="sqlite:///.flujo/state.db"

# Default to a local SQLite database for reliable pause/resume and history
state_uri = "sqlite:///.flujo/state.db"

# Load environment variables (API keys, etc.) from this file.
# Copy `.env.example` to `.env`, fill your keys, or change this path.
env_file = ".env"

[settings]
# default_solution_model = "gpt-4o-mini"

# Centralized budgets (optional)
[budgets]
# [budgets.default]
# total_cost_usd_limit = 5.0
# total_tokens_limit = 100000

# Architect defaults
[architect]
# Enable the agentic Architect state machine by default for this project
state_machine_default = true
# To disable by default, set to false or remove this section.
# Per-run overrides:
#   - Force agentic: export FLUJO_ARCHITECT_STATE_MACHINE=1
#   - Force minimal: export FLUJO_ARCHITECT_MINIMAL=1
# - CLI override on the create command:
#     uv run flujo create --agentic --goal "..."
#     uv run flujo create --no-agentic --goal "..."
            """.strip()
            + "\n",
        )
        _write(
            directory / "pipeline.yaml",
            """
version: "0.1"
name: "example"
steps:
  - kind: step
    name: passthrough
            """.strip()
            + "\n",
        )
        _write(skills_dir / "__init__.py", "# Custom project skills\n")
        _write(
            skills_dir / "custom_tools.py",
            """
from __future__ import annotations

# Example custom tool function
async def echo_tool(x: str) -> str:
    return x
            """.strip()
            + "\n",
        )
        _write(
            directory / "README.md",
            """
# Flujo Project

Welcome! This project is scaffolded for use with Flujo.

## Architect Defaults

This project enables the agentic Architect (state machine) by default via `flujo.toml`:

[architect]
state_machine_default = true

- To disable by default, set `state_machine_default = false` or remove the section.
- Per-run overrides:
  - Force agentic: `FLUJO_ARCHITECT_STATE_MACHINE=1`
  - Force minimal: `FLUJO_ARCHITECT_MINIMAL=1`
- CLI override on the create command:
  - `uv run flujo create --agentic --goal "..."`
  - `uv run flujo create --no-agentic --goal "..."`
            """.strip()
            + "\n",
        )

    try:
        (hidden_dir / "logs").mkdir(exist_ok=True)
        (hidden_dir / "cache").mkdir(exist_ok=True)
        try:
            hidden_dir.chmod(0o700)
        except OSError:
            pass

        db_path = hidden_dir / "state.db"
        try:
            flags = os.O_CREAT | os.O_WRONLY
            flags |= getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(db_path, flags, 0o600)
            try:
                try:
                    os.fchmod(fd, 0o600)
                except (AttributeError, OSError):
                    try:
                        os.chmod(db_path, 0o600)
                    except OSError:
                        pass
            finally:
                os.close(fd)
        except OSError:
            pass
    except Exception:
        pass

    if overwrite_existing and overwritten:
        secho(
            "✅ Re-initialized Flujo project templates.",
            fg="green",
        )
        secho(
            "Overwrote: " + ", ".join(sorted(overwritten)),
            fg="yellow",
        )
        if created:
            secho("Created: " + ", ".join(sorted(created)), fg="cyan")
    else:
        secho("✅ Your new Flujo project has been initialized in this directory!", fg="green")
        try:
            if (hidden_dir / "state.db").exists():
                secho("SQLite DB initialized at .flujo/state.db", fg="cyan")
        except Exception:
            pass


def scaffold_demo_project(directory: Path, *, overwrite_existing: bool = False) -> None:
    """Create a new Flujo demo project with a sample research pipeline."""
    from typer import secho

    directory = directory.resolve()
    flujo_toml = directory / "flujo.toml"
    pipeline_yaml = directory / "pipeline.yaml"
    hidden_dir = directory / ".flujo"
    skills_dir = directory / "skills"

    if (
        flujo_toml.exists() or pipeline_yaml.exists() or hidden_dir.exists()
    ) and not overwrite_existing:
        secho(
            "Error: This directory already contains Flujo project files. Use --force to overwrite.",
            fg="red",
        )
        raise Exit(1)

    skills_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir.mkdir(parents=True, exist_ok=True)

    created: list[str] = []
    overwritten: list[str] = []

    def _write(path: Path, content: str) -> None:
        target = path
        existed = target.exists()
        target.write_text(content)
        rel = str(target.relative_to(directory)) if target.is_file() else target.name
        if existed:
            overwritten.append(rel)
        else:
            created.append(rel)

    flujo_toml_content = (
        """
# Flujo project configuration template (demo)

# Use an in-memory state backend for demos so runs never persist
state_uri = "memory://"

[settings]
# default_solution_model = "gpt-4o-mini"
# reflection_enabled = true

# Centralized budgets (optional)
[budgets]
# [budgets.default]
# total_cost_usd_limit = 5.0
# total_tokens_limit = 100000
\n+# Architect defaults
[architect]
# Enable the agentic Architect state machine by default for this project
state_machine_default = true
""".strip()
        + "\n"
    )

    demo_pipeline_content = (
        generate_demo_yaml(demo_name="research_demo", preset="research_demo") + "\n"
    )

    skills_init_content = "# This marks the skills package for your project.\n"

    custom_tools_content = (
        """
from __future__ import annotations


# Example custom tool function
async def echo_tool(x: str) -> str:
    return x
""".strip()
        + "\n"
    )

    _write(flujo_toml, flujo_toml_content)
    _write(pipeline_yaml, demo_pipeline_content)
    try:
        from importlib import resources as _res

        template_pkg = "flujo.templates.project"
        with _res.files(template_pkg).joinpath(".env.example").open("r") as f:
            _write(directory / ".env.example", f.read())
    except Exception:
        pass
    _write(skills_dir / "__init__.py", skills_init_content)
    _write(skills_dir / "custom_tools.py", custom_tools_content)
    try:
        _write(
            directory / "README.md",
            """
# Flujo Demo Project

This demo is scaffolded with the agentic Architect enabled by default via `flujo.toml`:

[architect]
state_machine_default = true

Per-run overrides:
- Force agentic: `FLUJO_ARCHITECT_STATE_MACHINE=1`
- Force minimal: `FLUJO_ARCHITECT_MINIMAL=1`
CLI overrides (create command): `--agentic` / `--no-agentic`.
            """.strip()
            + "\n",
        )
    except Exception:
        pass

    try:
        (hidden_dir / "logs").mkdir(exist_ok=True)
        (hidden_dir / "cache").mkdir(exist_ok=True)
    except Exception:
        pass

    if overwrite_existing and overwritten:
        secho("✅ Re-initialized Flujo project with the demo pipeline.", fg="green")
        if overwritten:
            secho("Overwrote: " + ", ".join(sorted(overwritten)), fg="yellow")
        if created:
            secho("Created: " + ", ".join(sorted(created)), fg="cyan")
    else:
        secho("✅ Your new Flujo demo project is ready!", fg="green")
        secho("To run the demo, execute: [bold]flujo run[/bold]", fg="cyan")


def update_project_budget(flujo_toml_path: Path, pipeline_name: str, cost_limit: float) -> None:
    """Add or update a budget entry under [budgets.pipeline.<name>] in flujo.toml."""
    text = flujo_toml_path.read_text() if flujo_toml_path.exists() else ""

    section_header_quoted = f'[budgets.pipeline."{pipeline_name}"]'
    new_section = f"\n\n{section_header_quoted}\ntotal_cost_usd_limit = {cost_limit}\n"

    if not text.strip():
        flujo_toml_path.write_text("[budgets]\n" + new_section.lstrip("\n"))
        return

    pattern = rf"^\[(budgets\.pipeline\.(?:\"?{re.escape(pipeline_name)}\"?))\][\s\S]*?(?=^\[|\Z)"
    m = re.search(pattern, text, flags=re.MULTILINE)
    if m:
        start, end = m.span()
        updated = text[:start] + new_section.strip() + "\n" + text[end:]
        flujo_toml_path.write_text(updated)
        return

    if "\n[budgets]\n" not in ("\n" + text + "\n") and not re.search(
        r"^\[budgets\]", text, flags=re.MULTILINE
    ):
        text = text.rstrip() + "\n\n[budgets]\n"

    flujo_toml_path.write_text(text.rstrip() + new_section)


def generate_demo_yaml(
    *,
    demo_name: str = "demo",
    preset: str = "conversational_loop",
    conversation: bool = True,
    ai_turn_source: Optional[str] = None,
    user_turn_sources: Optional[list[str]] = None,
    history_strategy: Optional[str] = None,
    history_max_tokens: Optional[int] = None,
    history_max_turns: Optional[int] = None,
    history_summarize_ratio: Optional[float] = None,
) -> str:
    """Generate a ready-to-run demo pipeline YAML."""
    preset = (preset or "conversational_loop").strip().lower()
    uts = user_turn_sources or ["hitl"]

    if preset == "map_hitl":
        return (
            """
version: "0.1"
name: "{demo_name}"

steps:
  - kind: map
    name: AnnotateItems
    iterable_input: items
    body:
      - kind: hitl
        name: AnnotateItem
        message: "Provide a short note for this item"
      - kind: step
        name: Combine
        agent: {{ id: "flujo.builtins.stringify" }}
""".strip()
        ).format(demo_name=demo_name)

    if preset == "research_demo":
        return (
            """
version: "0.1"
name: "{demo_name}"

agents:
  result_formatter:
    model: "openai:gpt-4o-mini"
    system_prompt: |
      You are a research assistant. You will be given a JSON object containing web search results.
      Format these results into a clear, concise, and human-readable summary.
      Present the title, a brief snippet, and the link for each result in a numbered list.
      If the input is empty or contains no results, say "I couldn't find any information on that topic."
    output_schema:
      type: string
steps:
  - kind: step
    name: perform_web_search
    agent: {{ id: "flujo.builtins.web_search" }}
    input:
      query: "{{ initial_prompt }}"
  - kind: step
    name: summarize_results
    agent: {{ id: "agents.result_formatter" }}
    input:
      results: "{{ previous_step }}"  # web search output
    sink_to: final_result
            """.strip()
        ).format(demo_name=demo_name)

    conversation_settings = ""
    if conversation:
        if history_strategy:
            conversation_settings += f"    history_strategy: {history_strategy}\n"
        if history_max_tokens is not None:
            conversation_settings += f"    history_max_tokens: {history_max_tokens}\n"
        if history_max_turns is not None:
            conversation_settings += f"    history_max_turns: {history_max_turns}\n"
        if history_summarize_ratio is not None:
            conversation_settings += f"    history_summarize_ratio: {history_summarize_ratio}\n"

    return (
        """
version: "0.1"
name: "{demo_name}"

agents:
  ai_turn:
    model: "openai:gpt-4o-mini"
    system_prompt: |
      You are a helpful assistant participating in a guided conversation.
      Answer concisely. If context is available, use it to ground your response.
      If you don't know, say so briefly.

  summary_agent:
    model: "openai:gpt-4o-mini"
    system_prompt: |
      You are a summarizer. Combine the conversation history into a concise summary with key points.
      Keep it short and actionable.

steps:
  - kind: loop
    name: ConversationLoop
    max_loops: 3
    initial_input: "{{ initial_prompt }}"
    exit_condition: "{{ context.get('stop', False) }}"
    conversation_settings:
{conversation_settings}""".rstrip("\n").format(
            demo_name=demo_name, conversation_settings=conversation_settings
        )
        + """
    body:
      - kind: hitl
        name: UserTurn
        sink_to: "conversation_history"
        message: "Provide context or a question for this turn"

      - kind: step
        name: AiTurn
        agent:
          id: "agents.ai_turn"
          params:
            user_turn_source: "{{ user_turn_sources }}"
            ai_turn_source: "{{ ai_turn_source }}"
        input:
          user_turn: "{{ previous_step }}"
          conversation: "{{ conversation_history }}"
        context:
          update: true
          merge_strategy: overwrite
          target_path: "conversation_history"

      - kind: conditional
        name: StopCheck
        condition: "{{ 'stop' in context and context.stop }}"
        branches:
          true:
            - kind: step
              name: StopFlag
              input: "{{ context.stop }}"

      - kind: step
        name: Summarize
        agent:
          id: "agents.summary_agent"
        input:
          history: "{{ conversation_history }}"
        context:
          update: true
          target_path: "summary"
        plugins:
          - id: "flujo.plugins.deduplicate_tokens"

      - kind: step
        name: MaybeStop
        agent:
          id: "flujo.builtins.stringify"
        input: "{{ loop.counter < 3 }}"
        context:
          update: true
          target_path: "stop"
"""
        + (
            """

vars:
  user_turn_sources: {user_turn_sources}
  ai_turn_source: {ai_turn_source}
            """.format(
                user_turn_sources=uts,
                ai_turn_source=f"'{ai_turn_source}'" if ai_turn_source else "'hitl'",
            )
            if conversation
            else ""
        )
    )
