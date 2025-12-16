from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import importlib
from pathlib import Path

import pytest
import typer


def test_import_openapi_invokes_codegen(monkeypatch: Any, capsys: Any, tmp_path: Any) -> None:
    called_args: dict[str, list[str]] = {}

    def fake_main(args: list[str]) -> None:
        called_args["args"] = args

    monkeypatch.setitem(
        importlib.import_module("sys").modules,
        "datamodel_code_generator",
        SimpleNamespace(main=fake_main),
    )

    import flujo.cli.dev_commands_dev as dev_cmd

    out_dir = tmp_path / "out_dir"
    dev_cmd.import_openapi(
        spec="spec.yaml",
        output=str(out_dir),
        target_python_version="3.13",
        base_class="pydantic.BaseModel",
        disable_timestamp=True,
        generate_agents=False,
    )

    assert called_args["args"][0:6] == [
        "--input",
        "spec.yaml",
        "--input-file-type",
        "openapi",
        "--output",
        str(out_dir / "generated_models.py"),
    ]
    assert "--disable-timestamp" in called_args["args"]


def test_import_openapi_missing_dependency(monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.setitem(importlib.import_module("sys").modules, "datamodel_code_generator", None)

    import flujo.cli.dev_commands_dev as dev_cmd

    with pytest.raises(typer.Exit):
        dev_cmd.import_openapi(
            spec="spec.yaml",
            output="out_dir",
            target_python_version="3.13",
            base_class="pydantic.BaseModel",
            disable_timestamp=True,
            generate_agents=False,
        )


def test_import_openapi_rejects_non_package_output(monkeypatch: Any) -> None:
    monkeypatch.setitem(
        importlib.import_module("sys").modules,
        "datamodel_code_generator",
        SimpleNamespace(main=lambda _: None),
    )

    import flujo.cli.dev_commands_dev as dev_cmd

    with pytest.raises(typer.Exit):
        dev_cmd.import_openapi(spec="spec.yaml", output="bad-name", generate_agents=False)


def test_import_openapi_writes_init_exports(monkeypatch: Any, tmp_path: Any) -> None:
    def fake_main(args: list[str]) -> None:
        out = args[args.index("--output") + 1]
        Path(out).write_text(
            "from __future__ import annotations\n"
            "from pydantic import BaseModel\n"
            "class Pet(BaseModel):\n"
            "    name: str\n",
            encoding="utf-8",
        )

    monkeypatch.setitem(
        importlib.import_module("sys").modules,
        "datamodel_code_generator",
        SimpleNamespace(main=fake_main),
    )

    def fake_load(_: str) -> dict[str, Any]:
        return {"openapi": "3.0.0", "paths": {"/pets": {"get": {"operationId": "getPet"}}}}

    def fake_generate(
        *,
        spec: dict[str, Any],
        models_package: str,
        output_dir: Any,
        agents_filename: str = "openapi_agents.py",
        models_module: str = "generated_models",
    ) -> Any:
        path = Path(output_dir) / agents_filename
        path.write_text(
            "from __future__ import annotations\n"
            "OPERATION_FUNCS = {}\n"
            "RESPONSE_MODEL_NAMES = {}\n"
            "def make_openapi_agent(*, base_url: str, model: str = 'x', system_prompt: str | None = None):\n"
            "    return object()\n"
            "def make_openapi_operation_agent(*, base_url: str, operation: str, model: str = 'x', system_prompt: str | None = None):\n"
            "    return object()\n",
            encoding="utf-8",
        )
        return path

    monkeypatch.setattr("flujo.cli.dev_commands_dev.load_openapi_spec", fake_load)
    monkeypatch.setattr("flujo.cli.dev_commands_dev.generate_openapi_agents", fake_generate)

    import flujo.cli.dev_commands_dev as dev_cmd

    pkg_dir = tmp_path / "generated_tools"
    spec_file = tmp_path / "spec.json"
    spec_file.write_text('{"openapi":"3.0.0","paths":{}}', encoding="utf-8")

    dev_cmd.import_openapi(
        spec=str(spec_file),
        output=str(pkg_dir),
        generate_agents=True,
    )

    init_text = (pkg_dir / "__init__.py").read_text(encoding="utf-8")
    assert "make_openapi_agent" in init_text
    assert "make_openapi_operation_agent" in init_text

    monkeypatch.syspath_prepend(str(tmp_path))
    pkg = __import__("generated_tools", fromlist=["make_openapi_agent", "Pet"])
    assert hasattr(pkg, "make_openapi_agent")
