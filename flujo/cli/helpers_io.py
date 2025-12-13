from __future__ import annotations

import hashlib
import json
import os
import runpy
import sys
from typing import Any, List, Optional, Type, Union

import yaml
from typer import Exit

from flujo.domain.dsl import Pipeline, Step
from flujo.domain.models import PipelineContext
from flujo.infra.skills_catalog import load_skills_catalog, load_skills_entry_points
from flujo.type_definitions.common import JSONObject
from .exit_codes import EX_IMPORT_ERROR, EX_RUNTIME_ERROR


def print_rich_or_typer(msg: str, *, style: Optional[str] = None, stderr: bool = False) -> None:
    """Print a message using Rich when available, else fall back to typer.echo."""
    try:
        from rich.console import Console as _C
        import sys as _sys

        console = _C(file=_sys.stderr) if stderr else _C()
        console.print(msg, style=style)
        return
    except ModuleNotFoundError:
        pass

    import typer as _ty
    import re as _re

    plain = _re.sub(r"\[(?:/?)[a-zA-Z0-9_:\- ]+\]", "", msg)
    _ty.echo(plain, err=stderr)


def load_pipeline_from_file(
    pipeline_file: str,
    pipeline_name: str = "pipeline",
    *,
    lenient_dsl: bool = False,
) -> tuple["Pipeline[Any, Any]", str]:
    """Load a pipeline from a Python file."""
    prev_strict: str | None = None
    try:
        if lenient_dsl:
            prev_strict = os.environ.get("FLUJO_STRICT_DSL")
            os.environ["FLUJO_STRICT_DSL"] = "0"
        try:
            ns: JSONObject = runpy.run_path(pipeline_file)
        finally:
            if lenient_dsl:
                if prev_strict is None:
                    os.environ.pop("FLUJO_STRICT_DSL", None)
                else:
                    os.environ["FLUJO_STRICT_DSL"] = prev_strict
    except ModuleNotFoundError as e:
        mod = getattr(e, "name", None) or str(e)
        try:
            from typer import secho
            import os as _os
            import traceback as _tb

            secho(
                f"Import error: module '{mod}' not found. Try setting PYTHONPATH=. or pass --project/FLUJO_PROJECT_ROOT",
                fg="red",
                err=True,
            )
            if _os.getenv("FLUJO_CLI_VERBOSE") == "1" or _os.getenv("FLUJO_CLI_TRACE") == "1":
                secho("\nTraceback:", fg="yellow", err=True)
                secho("".join(_tb.format_exception(e)), err=True)
        except Exception:
            pass
        raise Exit(EX_IMPORT_ERROR)
    except Exception as e:
        try:
            from typer import secho
            import os as _os
            import traceback as _tb

            msg = f"Failed to load pipeline file: {type(e).__name__}: {e}"
            try:
                if isinstance(e, ValueError):
                    txt = str(e)
                    if (
                        "Type mismatch between steps" in txt
                        or "accepts a generic input type" in txt
                    ):
                        msg = f"Pipeline validation failed before run: {txt}"
            except Exception:
                pass
            secho(msg, fg="red", err=True)
            if _os.getenv("FLUJO_CLI_VERBOSE") == "1" or _os.getenv("FLUJO_CLI_TRACE") == "1":
                secho("\nTraceback:", fg="yellow", err=True)
                secho("".join(_tb.format_exception(e)), err=True)
        except Exception:
            pass
        raise Exit(EX_RUNTIME_ERROR)

    pipeline_obj = ns.get(pipeline_name)
    try:
        with open("output/last_pipeline_debug.txt", "w") as f:
            f.write(f"has_pipeline_var={pipeline_obj is not None}\n")
    except Exception:
        pass

    if pipeline_obj is None:
        pipeline_candidates = [
            (name, val)
            for name, val in ns.items()
            if isinstance(val, Pipeline) or isinstance(val, Step)
        ]
        if pipeline_candidates:
            selected = None
            for name, val in pipeline_candidates:
                if isinstance(val, Pipeline) and hasattr(val, "steps") and len(val.steps) > 1:
                    selected = (name, val)
                    break
            if selected:
                pipeline_name, pipeline_obj = selected
            else:
                pipeline_name, pipeline_obj = pipeline_candidates[0]
        else:
            try:
                from typer import secho

                secho(f"No Pipeline instance found in {pipeline_file}", fg="red")
            except Exception:
                pass
            raise Exit(1)

    if not isinstance(pipeline_obj, Pipeline):
        if isinstance(pipeline_obj, Step):
            pipeline_obj = Pipeline.from_step(pipeline_obj)
        else:
            try:
                from typer import secho

                secho(f"Object '{pipeline_name}' is not a Pipeline instance", fg="red")
            except Exception:
                pass
            raise Exit(1)

    return pipeline_obj, pipeline_name


def load_pipeline_from_yaml_file(yaml_path: str) -> "Pipeline[Any, Any]":
    """Load a pipeline from a YAML blueprint file (progressive v0)."""
    from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml

    try:
        dirname = os.path.dirname(os.path.abspath(yaml_path))
        load_skills_catalog(dirname)
        load_skills_entry_points()
        with open(yaml_path, "r") as f:
            text = f.read()
        os.environ["FLUJO_YAML_SPEC_SHA256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return load_pipeline_blueprint_from_yaml(text, base_dir=dirname)
    except Exception as e:
        try:
            from typer import secho
            import os as _os
            import traceback as _tb

            msg = f"Failed to load YAML pipeline: {type(e).__name__}: {e}"
            if isinstance(e, ModuleNotFoundError) or "No module named" in str(e):
                msg = f"Import error while loading YAML: {e}. Try setting PYTHONPATH=. or pass --project/FLUJO_PROJECT_ROOT"
            secho(msg, fg="red", err=True)
            if _os.getenv("FLUJO_CLI_VERBOSE") == "1" or _os.getenv("FLUJO_CLI_TRACE") == "1":
                secho("\nTraceback:", fg="yellow", err=True)
                secho("".join(_tb.format_exception(e)), err=True)
        except Exception:
            pass
        raise Exit(EX_IMPORT_ERROR if isinstance(e, ModuleNotFoundError) else EX_RUNTIME_ERROR)


def load_dataset_from_file(dataset_path: str) -> Any:
    """Load a dataset from a Python file."""
    try:
        dataset_ns: JSONObject = runpy.run_path(dataset_path)
    except Exception:
        raise Exit(1)

    dataset = dataset_ns.get("dataset") or dataset_ns.get("DATASET")
    if dataset is None:
        raise Exit(1)

    return dataset


def parse_context_data(
    context_data: Optional[str], context_file: Optional[str]
) -> Optional[JSONObject]:
    """Parse context data from string or file."""
    if context_data:
        try:
            from flujo.cli.main import safe_deserialize

            raw = safe_deserialize(json.loads(context_data))
            if raw is None:
                return None
            if isinstance(raw, dict):
                return raw
            raise Exit(1)
        except json.JSONDecodeError:
            raise Exit(1)

    if context_file:
        try:
            with open(context_file, "r") as f:
                if context_file.endswith((".yaml", ".yml")):
                    raw = yaml.safe_load(f)
                else:
                    from flujo.cli.main import safe_deserialize

                    raw = safe_deserialize(json.load(f))
                if raw is None:
                    return None
                if isinstance(raw, dict):
                    return raw
                raise Exit(1)
        except Exception:
            raise Exit(1)

    return None


def resolve_initial_input(input_data: Optional[str]) -> str:
    """Resolve the initial input to feed the pipeline."""
    from ..infra import telemetry

    logfire = telemetry.logfire

    if input_data is not None:
        if input_data.strip() == "-":
            try:
                content = sys.stdin.read()
                logfire.debug(f"[INPUT] Read from stdin via --input -: {len(content)} chars")
                return content.strip()
            except Exception as e:
                logfire.warning(f"[INPUT] Failed to read stdin via --input -: {e}")
                return ""
        logfire.debug(f"[INPUT] Using explicit --input value: {len(input_data)} chars")
        return input_data

    try:
        env_val = os.environ.get("FLUJO_INPUT")
        if isinstance(env_val, str) and env_val != "":
            logfire.debug(f"[INPUT] Using FLUJO_INPUT env var: {len(env_val)} chars")
            return env_val
    except Exception:
        pass

    try:
        is_tty_fn = getattr(sys.stdin, "isatty", None)
        if is_tty_fn is None:
            logfire.debug("[INPUT] isatty() unavailable, attempting stdin read")
            try:
                content = sys.stdin.read()
                if content.strip():
                    logfire.debug(f"[INPUT] Read from stdin (no isatty): {len(content)} chars")
                    return content.strip()
            except Exception:
                pass
        elif not is_tty_fn():
            logfire.debug("[INPUT] stdin is non-TTY, reading piped input")
            try:
                content = sys.stdin.read()
                logfire.debug(f"[INPUT] Read piped input: {len(content)} chars")
                return content.strip()
            except Exception as e:
                logfire.warning(f"[INPUT] Failed to read piped stdin: {e}")
                return ""
    except Exception as e:
        logfire.warning(f"[INPUT] Error checking stdin: {e}")

    logfire.debug("[INPUT] No input source found, using empty string")
    return ""


def validate_context_model(
    context_model: str, pipeline_file: str, ns: JSONObject
) -> Optional[Type[PipelineContext]]:
    """Validate and return a context model class."""
    try:
        context_model_class = ns.get(context_model)
        if context_model_class is None:
            from typer import secho

            secho(f"Context model '{context_model}' not found in {pipeline_file}", fg="red")
            raise Exit(1)

        if not isinstance(context_model_class, type):
            from typer import secho

            secho(f"'{context_model}' is not a class", fg="red")
            raise Exit(1)

        if not issubclass(context_model_class, PipelineContext):
            from typer import secho

            secho(f"'{context_model}' must inherit from PipelineContext", fg="red")
            raise Exit(1)

        return context_model_class
    except Exit:
        raise
    except Exception as e:
        from typer import secho

        secho(f"Error loading context model '{context_model}': {e}", fg="red")
        raise Exit(1)


def load_weights_file(weights_path: str) -> List[dict[str, Union[str, float]]]:
    """Load weights from a JSON or YAML file."""
    if not os.path.isfile(weights_path):
        try:
            from typer import secho

            secho(f"Weights file not found: {weights_path}", err=True)
        except Exception:
            pass
        raise Exit(1)

    try:
        with open(weights_path, "r") as f:
            if weights_path.endswith((".yaml", ".yml")):
                raw_weights = yaml.safe_load(f)
            else:
                from flujo.cli.main import safe_deserialize

                raw_weights = safe_deserialize(json.load(f))

        if not isinstance(raw_weights, list):
            raw_weights = None

        weights: List[dict[str, Union[str, float]]] = []
        if raw_weights is not None:
            for raw in raw_weights:
                if not isinstance(raw, dict):
                    raw_weights = None
                    break
                item = raw.get("item")
                weight = raw.get("weight")
                if not isinstance(item, str) or not isinstance(weight, (int, float)):
                    raw_weights = None
                    break
                weights.append({"item": item, "weight": float(weight)})

        if raw_weights is None:
            try:
                from typer import secho

                secho("Weights file must be a list of objects with 'item' and 'weight'", err=True)
            except Exception:
                pass
            raise Exit(1)

        return weights
    except Exception:
        try:
            from typer import secho

            secho("Error loading weights file", err=True)
        except Exception:
            pass
        raise Exit(1)
