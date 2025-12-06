from pathlib import Path

import pytest

from flujo.cli.generators.openapi_agent_generator import (
    generate_openapi_agents,
    load_openapi_spec,
)


def test_generate_openapi_agents_creates_wrapper(tmp_path: Path) -> None:
    spec = {
        "openapi": "3.0.0",
        "paths": {
            "/users": {
                "post": {
                    "operationId": "createUser",
                }
            }
        },
    }
    out_dir = tmp_path / "generated_tools"
    agents_path = generate_openapi_agents(
        spec=spec,
        models_package="generated_tools",
        output_dir=out_dir,
    )

    content = agents_path.read_text(encoding="utf-8")
    assert "async def createuser" in content  # sanitized operationId
    assert "make_openapi_agent" in content
    assert "__init__.py" in {p.name for p in out_dir.iterdir()}


def test_load_openapi_spec_json(tmp_path: Path) -> None:
    data = {"openapi": "3.0.0", "paths": {}}
    spec_path = tmp_path / "spec.json"
    spec_path.write_text('{"openapi":"3.0.0","paths":{}}', encoding="utf-8")

    loaded = load_openapi_spec(str(spec_path))
    assert loaded == data


def test_generate_openapi_agents_errors_on_empty_spec(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_openapi_agents(
            spec={"openapi": "3.0.0", "paths": {}},
            models_package="generated_tools",
            output_dir=tmp_path / "generated_tools",
        )
