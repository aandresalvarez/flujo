import pytest

from flujo.domain.models import PipelineContext


def test_pipeline_context_rejects_removed_field_payload() -> None:
    removed_field = "scrat" + "chpad"
    with pytest.raises(ValueError):
        PipelineContext.model_validate({removed_field: {"foo": 1}})
