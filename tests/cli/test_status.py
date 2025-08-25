from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _clean_provider_env(env: dict[str, str]) -> dict[str, str]:
    """Remove common provider API key env vars to control test state."""
    for key in [
        # OpenAI
        "OPENAI_API_KEY",
        "ORCH_OPENAI_API_KEY",
        # Anthropic
        "ANTHROPIC_API_KEY",
        "ORCH_ANTHROPIC_API_KEY",
        # Google/Gemini
        "GOOGLE_API_KEY",
        "ORCH_GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_GEMINI_API_KEY",
    ]:
        env.pop(key, None)
    return env


def _run_status(
    args: list[str], env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "flujo.cli.main", "status", *args],
        capture_output=True,
        text=True,
        env=env,
    )


def test_status_json_providers_presence_only() -> None:
    env = _clean_provider_env(os.environ.copy())
    res = _run_status(["--format=json", "--no-network"], env=env)
    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout or "{}")
    providers = {p.get("name"): p for p in payload.get("providers", [])}
    assert {"openai", "anthropic", "gemini"}.issubset(set(providers.keys()))
    assert providers["openai"].get("enabled") is False
    assert providers["anthropic"].get("enabled") is False
    assert providers["gemini"].get("enabled") is False


def test_status_json_openai_enabled_with_env() -> None:
    env = _clean_provider_env(os.environ.copy())
    env["OPENAI_API_KEY"] = "sk-test-xyz"
    res = _run_status(["--format=json", "--no-network"], env=env)
    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout or "{}")
    providers = {p.get("name"): p for p in payload.get("providers", [])}
    assert providers.get("openai") is not None
    assert providers["openai"].get("enabled") is True
    assert providers["anthropic"].get("enabled") is False
    assert providers["gemini"].get("enabled") is False


def test_status_sqlite_configured_when_state_uri_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "ops.db"
    env = _clean_provider_env(os.environ.copy())
    env["FLUJO_STATE_URI"] = f"sqlite:///{db_path.as_posix()}"
    res = _run_status(["--format=json", "--no-network"], env=env)
    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout or "{}")
    sqlite_info = payload.get("sqlite", {})
    assert sqlite_info.get("configured") is True
    assert sqlite_info.get("path", "").endswith("/ops.db")


def test_status_skips_db_when_memory_backend() -> None:
    env = _clean_provider_env(os.environ.copy())
    env["FLUJO_STATE_URI"] = "memory://"
    res = _run_status(["--format=json", "--no-network"], env=env)
    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout or "{}")
    sqlite_info = payload.get("sqlite", {})
    assert sqlite_info.get("configured") is False


def test_status_help_lists_command_and_flags() -> None:
    root = subprocess.run(
        [sys.executable, "-m", "flujo.cli.main", "--help"], capture_output=True, text=True
    )
    assert root.returncode == 0
    assert "status" in (root.stdout + root.stderr).lower()

    res = _run_status(["--help"], env=os.environ.copy())
    assert res.returncode == 0
    help_text = (res.stdout + res.stderr).lower()
    assert "--format" in help_text
    assert "--no-network" in help_text
    assert "--timeout" in help_text
