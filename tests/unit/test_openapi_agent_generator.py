from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from flujo.cli.generators.openapi_agent_generator import generate_openapi_agents, load_openapi_spec


def test_load_openapi_spec_local_json(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text('{"openapi":"3.0.0","paths":{}}', encoding="utf-8")
    spec = load_openapi_spec(str(spec_path))
    assert spec["openapi"] == "3.0.0"
    assert spec["paths"] == {}


def test_load_openapi_spec_local_yaml(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text("openapi: 3.0.0\npaths: {}\n", encoding="utf-8")
    spec = load_openapi_spec(str(spec_path))
    assert spec["openapi"] == "3.0.0"
    assert spec["paths"] == {}


def test_load_openapi_spec_url_json() -> None:
    payload = b'{"openapi":"3.0.0","paths":{}}'

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib name
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format: str, *_args: Any) -> None:  # silence
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    host, port = server.server_address
    host_str = host.decode("utf-8") if isinstance(host, bytes) else host
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        spec = load_openapi_spec(f"http://{host_str}:{port}/spec.json")
        assert spec["openapi"] == "3.0.0"
        assert spec["paths"] == {}
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.asyncio
async def test_generated_agent_resolves_output_type(tmp_path: Path, monkeypatch: Any) -> None:
    pkg = tmp_path / "gen"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "generated_models.py").write_text(
        "from __future__ import annotations\n"
        "from pydantic import BaseModel\n"
        "class Pet(BaseModel):\n"
        "    name: str\n",
        encoding="utf-8",
    )

    spec = {
        "openapi": "3.0.0",
        "paths": {
            "/pets/{id}": {
                "get": {
                    "operationId": "getPet",
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {"schema": {"$ref": "#/components/schemas/Pet"}}
                            }
                        }
                    },
                }
            }
        },
    }

    agents_path = generate_openapi_agents(
        spec=spec,
        models_package="gen",
        output_dir=pkg,
        agents_filename="openapi_agents.py",
        models_module="generated_models",
    )
    assert agents_path.exists()

    # Import the generated module as a package under sys.path.
    monkeypatch.syspath_prepend(str(tmp_path))
    mod = __import__("gen.openapi_agents", fromlist=["make_openapi_operation_agent", "Pet"])

    # Creating the agent should not call any network; we only validate the resolved output_type.
    agent = mod.make_openapi_operation_agent(base_url="http://example", operation="getpet")
    assert getattr(agent, "target_output_type", None) is mod.Pet


@pytest.mark.asyncio
async def test_generator_handles_array_response_schema(tmp_path: Path, monkeypatch: Any) -> None:
    pkg = tmp_path / "gen_arr"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "generated_models.py").write_text(
        "from __future__ import annotations\n"
        "from pydantic import BaseModel\n"
        "class Pet(BaseModel):\n"
        "    name: str\n",
        encoding="utf-8",
    )

    spec = {
        "openapi": "3.0.0",
        "paths": {
            "/pets": {
                "get": {
                    "operationId": "listPets",
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {"$ref": "#/components/schemas/Pet"},
                                    }
                                }
                            }
                        }
                    },
                }
            }
        },
    }

    agents_path = generate_openapi_agents(
        spec=spec,
        models_package="gen_arr",
        output_dir=pkg,
        agents_filename="openapi_agents.py",
        models_module="generated_models",
    )
    assert agents_path.exists()

    monkeypatch.syspath_prepend(str(tmp_path))
    mod = __import__("gen_arr.openapi_agents", fromlist=["RESPONSE_MODEL_NAMES"])
    assert mod.RESPONSE_MODEL_NAMES["listpets"] == "list[Pet]"

    agent = mod.make_openapi_operation_agent(base_url="http://example", operation="listpets")
    assert getattr(agent, "target_output_type", None) == list[mod.Pet]


@pytest.mark.asyncio
async def test_generator_non_json_response_falls_back_to_dict(
    tmp_path: Path, monkeypatch: Any
) -> None:
    pkg = tmp_path / "gen_text"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "generated_models.py").write_text(
        "from __future__ import annotations\n"
        "from pydantic import BaseModel\n"
        "class Ok(BaseModel):\n"
        "    ok: bool\n",
        encoding="utf-8",
    )

    spec = {
        "openapi": "3.0.0",
        "paths": {
            "/health": {
                "get": {
                    "operationId": "health",
                    "responses": {
                        "200": {"content": {"text/plain": {"schema": {"type": "string"}}}}
                    },
                }
            }
        },
    }

    agents_path = generate_openapi_agents(
        spec=spec,
        models_package="gen_text",
        output_dir=pkg,
        agents_filename="openapi_agents.py",
        models_module="generated_models",
    )
    assert agents_path.exists()

    monkeypatch.syspath_prepend(str(tmp_path))
    mod = __import__("gen_text.openapi_agents", fromlist=["RESPONSE_MODEL_NAMES"])
    assert mod.RESPONSE_MODEL_NAMES["health"] == "dict"

    agent = mod.make_openapi_operation_agent(base_url="http://example", operation="health")
    assert getattr(agent, "target_output_type", None) is dict


@pytest.mark.asyncio
async def test_generator_nested_ref_maps_to_top_level_model(
    tmp_path: Path, monkeypatch: Any
) -> None:
    pkg = tmp_path / "gen_nested"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "generated_models.py").write_text(
        "from __future__ import annotations\n"
        "from pydantic import BaseModel\n"
        "class Pet(BaseModel):\n"
        "    name: str\n"
        "class PetsResponse(BaseModel):\n"
        "    pets: list[Pet]\n",
        encoding="utf-8",
    )

    spec = {
        "openapi": "3.0.0",
        "paths": {
            "/pets": {
                "get": {
                    "operationId": "listPets",
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/PetsResponse"}
                                }
                            }
                        }
                    },
                }
            }
        },
    }

    agents_path = generate_openapi_agents(
        spec=spec,
        models_package="gen_nested",
        output_dir=pkg,
        agents_filename="openapi_agents.py",
        models_module="generated_models",
    )
    assert agents_path.exists()

    monkeypatch.syspath_prepend(str(tmp_path))
    mod = __import__("gen_nested.openapi_agents", fromlist=["PetsResponse"])
    agent = mod.make_openapi_operation_agent(base_url="http://example", operation="listpets")
    assert getattr(agent, "target_output_type", None) is mod.PetsResponse
