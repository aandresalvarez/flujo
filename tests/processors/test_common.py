import json
import pytest

from flujo.processors.common import EnforceJsonResponse
from flujo.processors.repair import DeterministicRepairProcessor


@pytest.mark.asyncio
async def test_enforce_json_response_uses_safe_deserialize(monkeypatch):
    calls = {}

    def fake(data):
        calls["called"] = True
        return json.loads(json.dumps(data))

    monkeypatch.setattr("flujo.processors.common.safe_deserialize", fake)
    proc = EnforceJsonResponse()
    result = await proc.process('{"x":1}')
    assert calls.get("called")
    assert result == {"x": 1}


@pytest.mark.asyncio
async def test_repair_canonical_uses_safe_deserialize(monkeypatch):
    calls = {}
    import flujo.processors.repair as repair_mod

    orig_sd = repair_mod.safe_deserialize

    def spy(data):
        calls["called"] = True
        return orig_sd(data)

    monkeypatch.setattr(repair_mod, "safe_deserialize", spy)
    proc = DeterministicRepairProcessor()
    result = await proc.process('{"a":1}')
    assert result == '{"a":1}'
    assert calls.get("called")
