from __future__ import annotations

import pytest

from datetime import datetime, timedelta, timezone
from flujo.cli.main import dev_health_check as _dev_health_check


class _FakeBackendTrend:
    def __init__(self) -> None:
        now = datetime.now(timezone.utc)
        # Two runs mapped into different buckets by time
        self.runs = [
            {"run_id": "old", "start_time": (now - timedelta(hours=20)).isoformat()},
            {"run_id": "new", "start_time": (now - timedelta(hours=1)).isoformat()},
        ]
        # old bucket: low tolerant for stepX/modelX; new bucket: higher tolerant
        self.spans = {
            "old": [
                {
                    "name": "stepX",
                    "attributes": {
                        "aros.coercion.total": 1,
                        "aros.coercion.stage.tolerant": 1,
                        "aros.model_id": "openai:gpt-4o",
                    },
                }
            ],
            "new": [
                {
                    "name": "stepX",
                    "attributes": {
                        "aros.coercion.total": 4,
                        "aros.coercion.stage.tolerant": 4,
                        "aros.model_id": "openai:gpt-4o",
                    },
                }
            ],
        }

    async def list_runs(self, pipeline_name=None, limit=50):  # type: ignore[no-untyped-def]
        return self.runs[:limit]

    async def get_spans(self, run_id):  # type: ignore[no-untyped-def]
        return self.spans.get(run_id, [])


@pytest.mark.fast
def test_trend_stage_specific_recommendations(monkeypatch, capsys):
    fake = _FakeBackendTrend()
    monkeypatch.setattr("flujo.cli.dev_commands.load_backend_from_config", lambda: fake)

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
    assert "Trend: Step 'stepX' tolerant coercions rising" in out
    assert "Trend: Model 'openai:gpt-4o' tolerant coercions rising" in out
