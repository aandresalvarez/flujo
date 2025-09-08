from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from flujo.cli.main import app


runner = CliRunner()


def test_lens_from_file_with_full_export_payload(tmp_path: Path) -> None:
    file = tmp_path / "trace.json"
    payload = {
        "run_id": "r-1",
        "pipeline_name": "p",
        "exported_at": "2024-01-01T00:00:00",
        "trace_tree": {
            "run_id": "r-1",
            "name": "p",
            "status": "completed",
            "start_time": 0.0,
            "end_time": 1.0,
            "children": [],
        },
    }
    file.write_text(json.dumps(payload))

    res = runner.invoke(app, ["lens", "from-file", str(file), "--prompt-preview-len", "10"])
    assert res.exit_code == 0, res.output
    assert "Run ID:" in res.output or "Trace (from file)" in res.output


def test_lens_from_file_with_bare_trace_root(tmp_path: Path) -> None:
    file = tmp_path / "trace_root.json"
    trace = {
        "run_id": "r-2",
        "name": "p2",
        "status": "completed",
        "start_time": 0.0,
        "end_time": 0.5,
        "children": [],
    }
    file.write_text(json.dumps(trace))

    res = runner.invoke(app, ["lens", "from-file", str(file)])
    assert res.exit_code == 0, res.output
    assert "p2" in res.output


def test_lens_from_file_invalid_json(tmp_path: Path) -> None:
    file = tmp_path / "bad.json"
    file.write_text("not-json")
    res = runner.invoke(app, ["lens", "from-file", str(file)])
    assert res.exit_code != 0
    assert "Failed to read file:" in res.output


def test_lens_from_file_invalid_payload_shape(tmp_path: Path) -> None:
    file = tmp_path / "badshape.json"
    file.write_text(json.dumps([1, 2, 3]))
    res = runner.invoke(app, ["lens", "from-file", str(file)])
    assert res.exit_code != 0
    assert "Invalid trace payload" in res.output


def test_lens_replay_with_file_and_json_output(monkeypatch, tmp_path: Path) -> None:
    # Patch helpers used inside lens.replay to avoid importing user files
    # Minimal fake runner with async replay method
    class _FakeRunner:
        def __init__(self) -> None:
            self.state_backend = None

        async def replay_from_trace(self, run_id: str):  # type: ignore[no-untyped-def]
            return {"ok": 1, "run": run_id}

    # Patch helpers that lens.replay imports at call-time
    monkeypatch.setattr(
        "flujo.cli.helpers.load_pipeline_from_file", lambda *a, **k: (object(), "pipeline")
    )
    monkeypatch.setattr("flujo.cli.helpers.create_flujo_runner", lambda *a, **k: _FakeRunner())

    res = runner.invoke(
        app,
        [
            "lens",
            "replay",
            "rid-123",
            "--file",
            str(tmp_path / "pipe.py"),
            "--json",
        ],
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.stdout or "{}")
    assert payload.get("ok") == 1
    assert payload.get("run") == "rid-123"


def test_lens_replay_respects_state_uri_override(monkeypatch, tmp_path: Path) -> None:
    observed: dict[str, str] = {}

    def fake_load_backend():
        import os as _os

        observed["state_uri"] = _os.environ.get("FLUJO_STATE_URI", "")
        return object()

    class _FakeRunner:
        def __init__(self) -> None:
            self.state_backend = None

        async def replay_from_trace(self, run_id: str):  # type: ignore[no-untyped-def]
            return {"ok": 1, "run": run_id}

    monkeypatch.setattr("flujo.cli.lens.load_backend_from_config", fake_load_backend)
    monkeypatch.setattr(
        "flujo.cli.helpers.load_pipeline_from_file", lambda *a, **k: (object(), "pipeline")
    )
    monkeypatch.setattr("flujo.cli.helpers.create_flujo_runner", lambda *a, **k: _FakeRunner())

    state_uri = f"sqlite:///{(tmp_path / 'ops.db').as_posix()}"
    res = runner.invoke(
        app,
        [
            "lens",
            "replay",
            "rid-xyz",
            "--file",
            str(tmp_path / "pipe.py"),
            "--json",
            "--state-uri",
            state_uri,
        ],
    )
    assert res.exit_code == 0, res.output
    assert observed.get("state_uri") == state_uri


def test_lens_replay_infers_pipeline_from_project_yaml(monkeypatch, tmp_path: Path) -> None:
    # Set up a project with pipeline.yaml
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        (tmp_path / "skills").mkdir(parents=True, exist_ok=True)
        (tmp_path / "pipeline.yaml").write_text('version: "0.1"\nsteps: []\n')

        class _FakeRunner:
            def __init__(self) -> None:
                self.state_backend = None

            async def replay_from_trace(self, run_id: str):  # type: ignore[no-untyped-def]
                return {"ok": 1, "run": run_id}

        # Ensure lens finds the current directory as project root
        monkeypatch.setattr("flujo.cli.lens.find_project_root", lambda: tmp_path)
        monkeypatch.setattr("flujo.cli.lens.load_backend_from_config", lambda: object())
        # Helpers are imported inside replay_command at runtime; patch there
        monkeypatch.setattr(
            "flujo.cli.helpers.load_pipeline_from_yaml_file", lambda *_a, **_k: object()
        )
        monkeypatch.setattr("flujo.cli.helpers.create_flujo_runner", lambda *a, **k: _FakeRunner())
        # Avoid heavy printing
        monkeypatch.setattr("flujo.cli.helpers.display_pipeline_results", lambda *a, **k: None)

        res = runner.invoke(app, ["lens", "replay", "rid-abc"])
        assert res.exit_code == 0, res.output


def test_lens_replay_errors_without_file_and_no_registry(monkeypatch) -> None:
    # Provide a fast fake backend that reports missing run
    class _FakeBackend:
        async def get_run_details(self, run_id: str):  # type: ignore[no-untyped-def]
            return None

    monkeypatch.setattr("flujo.cli.lens.load_backend_from_config", lambda: _FakeBackend())

    res = runner.invoke(app, ["lens", "replay", "nonexistent"], env={})
    assert res.exit_code != 0
    assert "Run not found" in (res.stdout + res.stderr)
