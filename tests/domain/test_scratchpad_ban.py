import pytest

from flujo.domain.models import PipelineContext


def test_pipeline_context_rejects_scratchpad_payload() -> None:
    with pytest.raises(ValueError):
        PipelineContext.model_validate({"scratchpad": {"foo": 1}})
