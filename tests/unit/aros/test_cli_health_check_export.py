from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from flujo.cli.main import dev_health_check as _dev_health_check


class _FakeBackend:
    def __init__(self) -> None:
        now = datetime.now(timezone.utc)
        self.runs = [
            {
                "run_id": "r1",
                "start_time": (now - timedelta(hours=1)).isoformat(),
            },
            {
                "run_id": "r2",
                "start_time": (now - timedelta(hours=12)).isoformat(),
            },
        ]
        self.spans = {
            "r1": [
                {
                    "name": "stepA",
                    "attributes": {
                        "aros.coercion.total": 6,
                        "aros.coercion.stage.tolerant": 5,
                        "aros.coercion.stage.semantic": 1,
                        "aros.model_id": "openai:gpt-4o",
                    },
                }
            ],
            "r2": [
                {
                    "name": "stepB",
                    "attributes": {
                        "aros.coercion.total": 2,
                        "aros.coercion.stage.semantic": 2,
                        "aros.model_id": "anthropic:claude-3",
                    },
                }
            ],
        }

    async def list_runs(self, pipeline_name=None, limit=50):  # type: ignore[no-untyped-def]
        return self.runs[:limit]

    async def get_spans(self, run_id):  # type: ignore[no-untyped-def]
        return self.spans.get(run_id, [])


@pytest.mark.fast
def test_health_check_json_export_includes_bucket_step_and_model_breakdowns(tmp_path, monkeypatch):
    # Monkeypatch backend loader to use fake backend

    fake = _FakeBackend()
    monkeypatch.setattr("flujo.cli.dev_commands.load_backend_from_config", lambda: fake)

    out_path = tmp_path / "hc.json"
    # Call function directly; it runs anyio.run internally
    _dev_health_check(
        project=None,
        limit=10,
        pipeline=None,
        since_hours=24,
        trend_buckets=2,
        export="json",
        output=str(out_path),
    )

    # Verify JSON exported with expected bucket breakdowns
    payload = json.loads(out_path.read_text())
    assert isinstance(payload, dict)
    trend = payload.get("trend")
    assert isinstance(trend, dict)
    buckets = trend.get("buckets")
    assert isinstance(buckets, list) and len(buckets) == 2
    # Find a bucket with non-empty step/model stages
    found_step = any(isinstance(b.get("step_stages"), dict) and b["step_stages"] for b in buckets)
    found_model = any(
        isinstance(b.get("model_stages"), dict) and b["model_stages"] for b in buckets
    )
    assert found_step and found_model
    # Top-level step/model stage maps exist and contain expected keys
    step_stages = payload.get("step_stages")
    model_stages = payload.get("model_stages")
    assert isinstance(step_stages, dict) and "stepA" in step_stages and "stepB" in step_stages
    assert (
        isinstance(model_stages, dict)
        and "openai:gpt-4o" in model_stages
        and "anthropic:claude-3" in model_stages
    )


@pytest.mark.fast
def test_health_check_recommendations_include_stage_aware(monkeypatch, capsys):
    fake = _FakeBackend()
    monkeypatch.setattr("flujo.cli.dev_commands.load_backend_from_config", lambda: fake)

    _dev_health_check(
        project=None,
        limit=10,
        pipeline=None,
        since_hours=24,
        trend_buckets=None,
        export=None,
        output=None,
    )
    out = capsys.readouterr().out
    assert "High tolerant-decoder activity detected" in out
    # Also includes per-step stage-aware guidance for top step
    assert "Step 'stepA' shows tolerant-decoder activity" in out
    # And per-model stage-aware guidance for top model
    assert "Model 'openai:gpt-4o' often needs tolerant decoders" in out


@pytest.mark.fast
def test_health_check_json_export_without_since_hours_has_no_buckets(tmp_path, monkeypatch):
    fake = _FakeBackend()
    monkeypatch.setattr("flujo.cli.dev_commands.load_backend_from_config", lambda: fake)

    out_path = tmp_path / "hc2.json"
    _dev_health_check(
        project=None,
        limit=10,
        pipeline=None,
        since_hours=None,
        trend_buckets=3,
        export="json",
        output=str(out_path),
    )

    payload = json.loads(out_path.read_text())
    trend = payload.get("trend")
    assert isinstance(trend, dict)
    assert (trend.get("buckets") is None) or (
        isinstance(trend.get("buckets"), list) and len(trend.get("buckets")) == 0
    )


@pytest.mark.fast
def test_health_check_handles_no_runs(monkeypatch, capsys):
    class _EmptyBackend(_FakeBackend):
        def __init__(self) -> None:
            self.runs = []
            self.spans = {}

    empty = _EmptyBackend()
    monkeypatch.setattr("flujo.cli.dev_commands.load_backend_from_config", lambda: empty)

    # Should print a message and not raise
    _dev_health_check(
        project=None,
        limit=10,
        pipeline=None,
        since_hours=24,
        trend_buckets=2,
        export=None,
        output=None,
    )
    out = capsys.readouterr().out
    assert "No runs found" in out
