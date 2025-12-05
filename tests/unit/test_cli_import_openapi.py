from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import importlib

import pytest
import typer


def test_import_openapi_invokes_codegen(monkeypatch: Any, capsys: Any) -> None:
    called_args: dict[str, list[str]] = {}

    def fake_main(args: list[str]) -> None:
        called_args["args"] = args

    monkeypatch.setitem(
        importlib.import_module("sys").modules,
        "datamodel_code_generator",
        SimpleNamespace(main=fake_main),
    )

    import flujo.cli.dev_commands_dev as dev_cmd

    dev_cmd.import_openapi(
        spec="spec.yaml",
        output="out_dir",
        target_python_version="3.11",
        base_class="pydantic.BaseModel",
        disable_timestamp=True,
    )

    assert called_args["args"][0:6] == [
        "--input",
        "spec.yaml",
        "--input-file-type",
        "openapi",
        "--output",
        "out_dir",
    ]
    assert "--disable-timestamp" in called_args["args"]


def test_import_openapi_missing_dependency(monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.setitem(importlib.import_module("sys").modules, "datamodel_code_generator", None)

    import flujo.cli.dev_commands_dev as dev_cmd

    with pytest.raises(typer.Exit):
        dev_cmd.import_openapi(
            spec="spec.yaml",
            output="out_dir",
            target_python_version="3.11",
            base_class="pydantic.BaseModel",
            disable_timestamp=True,
        )
