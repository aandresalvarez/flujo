from __future__ import annotations

import pytest

from flujo.domain.blueprint.loader import BlueprintError, load_pipeline_blueprint_from_yaml


def test_inline_lambda_condition_raises_actionable_error() -> None:
    yaml_text = (
        'version: "0.1"\n'
        "steps:\n"
        "  - kind: conditional\n"
        "    name: cond\n"
        '    condition: "lambda output, ctx: output"\n'
        "    branches:\n"
        "      'true': []\n"
        "      'false': []\n"
    )
    with pytest.raises(BlueprintError) as ei:
        load_pipeline_blueprint_from_yaml(yaml_text)
    msg = str(ei.value).lower()
    assert ("inline" in msg or "lambda" in msg) and "condition_expression" in msg


def test_bad_import_path_condition_is_rewrapped_with_field_context() -> None:
    yaml_text = (
        'version: "0.1"\n'
        "steps:\n"
        "  - kind: conditional\n"
        "    name: cond\n"
        '    condition: "this_package_does_not_exist.foo:bar"\n'
        "    branches:\n"
        "      general: []\n"
    )
    with pytest.raises(BlueprintError) as ei:
        load_pipeline_blueprint_from_yaml(yaml_text)
    msg = str(ei.value).lower()
    assert "field: condition" in msg
    assert ("provide a python import path" in msg) or ("condition_expression" in msg)
