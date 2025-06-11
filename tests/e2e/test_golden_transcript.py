import pytest
import vcr
from pydantic_ai_orchestrator.application.orchestrator import Orchestrator

def test_golden_transcript(tmp_path):
    orch = Orchestrator()
    with vcr.use_cassette(str(tmp_path / "golden.yaml")):
        result = orch.run_sync("Say hello.")
        assert isinstance(result, str)
        assert "hello" in result.lower() 