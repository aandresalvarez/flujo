from __future__ import annotations

import asyncio
import gc
import importlib.metadata as importlib_metadata
import json
import os
import sys
from typing import Any, Dict, List, Optional, Type, cast

from typer import Exit

from flujo.domain.models import PipelineContext
from flujo.infra import telemetry as _telemetry
from flujo.utils.serialization import safe_serialize

from .helpers_io import (
    load_dataset_from_file,
    load_pipeline_from_file,
    parse_context_data,
    print_rich_or_typer,
    resolve_initial_input,
    validate_context_model,
)
from .helpers_solve import (
    create_improvement_report_table,
    create_pipeline_results_table,
    format_improvement_suggestion,
    setup_json_output_mode,
)


def setup_run_command_environment(
    pipeline_file: str,
    pipeline_name: str,
    json_output: bool,
    input_data: Optional[str],
    context_model: Optional[str],
    context_data: Optional[str],
    context_file: Optional[str],
) -> tuple[Any, str, Any, Optional[Dict[str, Any]], Optional[Type[PipelineContext]]]:
    """Set up the environment for the run command."""
    import runpy

    setup_json_output_mode(json_output)
    pipeline_obj, pipeline_name = load_pipeline_from_file(pipeline_file, pipeline_name)
    input_data = resolve_initial_input(input_data)

    context_model_class = None
    if context_model:
        ns = runpy.run_path(pipeline_file)
        context_model_class = validate_context_model(context_model, pipeline_file, ns)

    initial_context_data = parse_context_data(context_data, context_file)

    if context_model_class is not None:
        if initial_context_data is None:
            initial_context_data = {}
        if "initial_prompt" not in initial_context_data:
            initial_context_data["initial_prompt"] = input_data

    return pipeline_obj, pipeline_name, input_data, initial_context_data, context_model_class


def create_flujo_runner(
    pipeline: Any,
    context_model_class: Optional[Type[PipelineContext]],
    initial_context_data: Optional[Dict[str, Any]],
    state_backend: Optional[Any] = None,
    *,
    debug: bool = False,
    live: bool = False,
) -> Any:
    """Create a Flujo runner instance with the given configuration."""
    from flujo.cli.main import Flujo
    from flujo.domain.models import PipelineContext

    try:
        inferred_name = getattr(pipeline, "name", None)
        if not isinstance(inferred_name, str) or not inferred_name.strip():
            inferred_name = None
    except Exception:
        inferred_name = None

    usage_limits_arg = None
    try:
        from flujo.utils.config import get_settings as _get_proc_settings

        if not _get_proc_settings().test_mode:
            from flujo.infra.budget_resolver import resolve_limits_for_pipeline as _resolve
            from flujo.infra.config_manager import ConfigManager

            cfg = ConfigManager().load_config()
            pname = inferred_name or ""
            usage_limits_arg, _src = _resolve(getattr(cfg, "budgets", None), pname)
    except Exception:
        usage_limits_arg = None

    local_tracer_arg: Any
    if debug:
        local_tracer_arg = "default"
    elif live:
        try:
            from flujo.infra.console_tracer import ConsoleTracer as _ConsoleTracer

            local_tracer_arg = _ConsoleTracer(level="info", log_inputs=False, log_outputs=False)
        except Exception:
            local_tracer_arg = "default"
    else:
        local_tracer_arg = None

    enable_tracing_arg = True
    try:
        from flujo.utils.config import get_settings as _get_proc_settings

        if _get_proc_settings().test_mode:
            enable_tracing_arg = False
    except Exception:
        pass

    if context_model_class is not None:
        runner = Flujo[Any, Any, PipelineContext](
            pipeline=pipeline,
            context_model=context_model_class,
            initial_context_data=initial_context_data,
            state_backend=state_backend,
            pipeline_name=inferred_name,
            usage_limits=usage_limits_arg,
            local_tracer=local_tracer_arg,
            enable_tracing=enable_tracing_arg,
        )
    else:
        runner = Flujo[Any, Any, PipelineContext](
            pipeline=pipeline,
            context_model=None,
            initial_context_data=initial_context_data,
            state_backend=state_backend,
            pipeline_name=inferred_name,
            usage_limits=usage_limits_arg,
            local_tracer=local_tracer_arg,
            enable_tracing=enable_tracing_arg,
        )

    return runner


def execute_pipeline_with_output_handling(
    runner: Any,
    input_data: str,
    run_id: Optional[str],
    json_output: bool,
) -> Any:
    """Execute the pipeline and handle output formatting."""
    import asyncio as _asyncio
    import io as _io
    import sys as _sys

    _disable_span = False
    try:
        _disable_span = str(os.getenv("FLUJO_DISABLE_TRACING", "")).strip().lower() in {
            "1",
            "true",
            "on",
            "yes",
        }
        if not _disable_span:
            from flujo.utils.config import get_settings as _get_proc_settings

            _disable_span = bool(_get_proc_settings().test_mode)
    except Exception:
        pass

    def _run_pipeline() -> Any:
        if json_output:
            buf = _io.StringIO()
            old_stdout = _sys.stdout
            _sys.stdout = buf
            try:
                if run_id is not None:
                    result = runner.run(input_data, run_id=run_id)
                else:
                    result = runner.run(input_data)
            finally:
                _sys.stdout = old_stdout

            from flujo.utils.serialization import serialize_to_json_robust

            return serialize_to_json_robust(result, indent=2)
        if run_id is not None:
            result = runner.run(input_data, run_id=run_id)
        else:
            result = runner.run(input_data)
        return result

    if _disable_span:
        try:
            result = _run_pipeline()
            return result
        finally:
            try:
                gc.collect()
            except Exception:
                pass
    else:
        with _telemetry.logfire.span("pipeline_run"):
            try:
                result = _run_pipeline()
                return result
            finally:
                try:
                    gc.collect()
                except Exception:
                    pass

                try:
                    import ctypes

                    if sys.platform.startswith("linux"):
                        try:
                            libc = ctypes.CDLL("libc.so.6")
                            libc.malloc_trim(0)
                        except Exception:
                            pass
                    elif sys.platform == "darwin":
                        try:
                            libsys = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
                            libsys.malloc_default_zone.restype = ctypes.c_void_p
                            zone = libsys.malloc_default_zone()
                            libsys.malloc_zone_pressure_relief.argtypes = [
                                ctypes.c_void_p,
                                ctypes.c_size_t,
                            ]
                            libsys.malloc_zone_pressure_relief.restype = ctypes.c_size_t
                            libsys.malloc_zone_pressure_relief(
                                ctypes.c_void_p(zone), ctypes.c_size_t(0)
                            )
                        except Exception:
                            pass
                except Exception:
                    pass

                try:
                    loop = _asyncio.get_running_loop()
                    for task in _asyncio.all_tasks(loop):
                        if not task.done():
                            task.cancel()
                except RuntimeError:
                    pass

                try:
                    import httpx

                    if hasattr(httpx, "_default_limits"):
                        httpx._default_limits = None
                except Exception:
                    pass

                try:
                    pass
                except Exception:
                    pass

                try:
                    sb = getattr(runner, "state_backend", None)
                    if sb is not None and hasattr(sb, "shutdown"):

                        async def _do_shutdown() -> None:
                            try:
                                await sb.shutdown()
                            except Exception:
                                pass

                        try:
                            _asyncio.run(_do_shutdown())
                        except RuntimeError:
                            try:
                                loop = _asyncio.get_running_loop()
                                t = loop.create_task(_do_shutdown())
                                loop.run_until_complete(t)
                            except Exception:
                                pass
                except Exception:
                    pass

                try:
                    from flujo.infra.skill_registry import get_skill_registry_provider

                    reg = get_skill_registry_provider().get_registry()
                    entries = getattr(reg, "_entries", None)
                    if isinstance(entries, dict):
                        preserved: dict[str, Any] = {
                            k: v
                            for k, v in list(entries.items())
                            if isinstance(k, str)
                            and (
                                k.startswith("flujo.builtins.") or k.startswith("flujo.architect.")
                            )
                        }
                        entries.clear()
                        entries.update(preserved)
                except Exception:
                    pass

                try:
                    gc.collect()
                except Exception:
                    pass


def display_pipeline_results(
    result: Any,
    run_id: Optional[str],
    json_output: bool,
    *,
    show_steps: bool = True,
    show_context: bool = True,
    show_output_column: bool = True,
    output_preview_len: Optional[int] = None,
    final_output_format: str = "auto",
    pager: bool = False,
    only_steps: Optional[List[str]] = None,
) -> None:
    """Display pipeline execution results in the appropriate format."""
    if json_output:
        return

    try:
        from rich.console import Console as _Console

        _rich_available = True
        console = _Console()
    except ModuleNotFoundError:
        _rich_available = False
        console = None
    paused = False
    hitl_message = None
    try:
        ctx = getattr(result, "final_pipeline_context", None)
        scratch = getattr(ctx, "scratchpad", None) if ctx is not None else None
        if isinstance(scratch, dict) and scratch.get("status") == "paused":
            paused = True
            hitl_message = scratch.get("pause_message") or scratch.get("hitl_message")
    except Exception:
        paused = False

    def _render_body() -> None:
        if paused:
            if _rich_available and console is not None:
                console.print("[bold yellow]Pipeline execution paused.[/bold yellow]")
            else:
                print_rich_or_typer("[bold yellow]Pipeline execution paused.[/bold yellow]")
            if hitl_message:
                if _rich_available and console is not None:
                    console.print(hitl_message)
                else:
                    print_rich_or_typer(str(hitl_message))
            return
        else:
            is_success = bool(getattr(result, "success", False))
            if _rich_available and console is not None:
                if is_success:
                    console.print(
                        "[bold green]Pipeline execution completed successfully![/bold green]"
                    )
                else:
                    console.print("[bold red]Pipeline execution failed.[/bold red]")
            else:
                print_rich_or_typer(
                    "[bold green]Pipeline execution completed successfully![/bold green]"
                    if is_success
                    else "[bold red]Pipeline execution failed.[/bold red]"
                )

        final_output = result.step_history[-1].output if result.step_history else None
        try:
            if hasattr(final_output, "value") and not isinstance(final_output, (str, bytes)):
                final_output = getattr(final_output, "value")
            elif (
                isinstance(final_output, dict)
                and "value" in final_output
                and len(final_output) == 1
            ):
                final_output = final_output.get("value")

            if _rich_available and console is not None:
                console.print("[bold]Final output:[/bold]")
            else:
                print_rich_or_typer("[bold]Final output:[/bold]")
            fmt = (final_output_format or "auto").lower()
            if fmt == "raw":
                if _rich_available and console is not None:
                    console.print(str(final_output))
                else:
                    print_rich_or_typer(str(final_output))
            elif fmt == "json":
                try:
                    to_dump: Any
                    if isinstance(final_output, str):
                        to_dump = json.loads(final_output)
                    else:
                        to_dump = safe_serialize(final_output)
                    if _rich_available and console is not None:
                        console.print(json.dumps(to_dump, indent=2, ensure_ascii=False))
                    else:
                        print_rich_or_typer(json.dumps(to_dump, indent=2, ensure_ascii=False))
                except Exception:
                    if _rich_available and console is not None:
                        console.print(str(final_output))
                    else:
                        print_rich_or_typer(str(final_output))
            elif fmt == "yaml":
                try:
                    import yaml as _yaml

                    to_dump = safe_serialize(final_output)
                    dumped = _yaml.safe_dump(
                        to_dump, sort_keys=False, allow_unicode=True, default_flow_style=False
                    )
                    if _rich_available and console is not None:
                        console.print(dumped)
                    else:
                        print_rich_or_typer(dumped)
                except Exception:
                    if _rich_available and console is not None:
                        console.print(str(final_output))
                    else:
                        print_rich_or_typer(str(final_output))
            elif fmt == "md":
                try:
                    from rich.markdown import Markdown as _Markdown

                    if _rich_available and console is not None:
                        console.print(_Markdown(str(final_output)))
                    else:
                        print_rich_or_typer(str(final_output))
                except ModuleNotFoundError:
                    print_rich_or_typer(str(final_output))
            else:
                if isinstance(final_output, str):
                    try:
                        from rich.markdown import Markdown as _Markdown

                        if _rich_available and console is not None:
                            console.print(_Markdown(final_output))
                        else:
                            print_rich_or_typer(final_output)
                    except ModuleNotFoundError:
                        print_rich_or_typer(final_output)
                elif isinstance(final_output, bytes):
                    try:
                        decoded_output = final_output.decode("utf-8")
                        if _rich_available and console is not None:
                            console.print(decoded_output)
                        else:
                            print_rich_or_typer(decoded_output)
                    except UnicodeDecodeError:
                        if _rich_available and console is not None:
                            console.print(repr(final_output))
                        else:
                            print_rich_or_typer(repr(final_output))
                else:
                    if _rich_available and console is not None:
                        console.print(str(final_output))
                    else:
                        print_rich_or_typer(str(final_output))
        except Exception:
            if isinstance(final_output, bytes):
                try:
                    decoded_output = final_output.decode("utf-8")
                    if _rich_available and console is not None:
                        console.print(f"[bold]Final output:[/bold] {decoded_output}")
                    else:
                        print_rich_or_typer(f"[bold]Final output:[/bold] {decoded_output}")
                except UnicodeDecodeError:
                    if _rich_available and console is not None:
                        console.print(f"[bold]Final output:[/bold] {final_output!r}")
                    else:
                        print_rich_or_typer(f"[bold]Final output:[/bold] {final_output!r}")
            else:
                if _rich_available and console is not None:
                    console.print(f"[bold]Final output:[/bold] {final_output}")
                else:
                    print_rich_or_typer(f"[bold]Final output:[/bold] {final_output}")

        line_total_cost = f"[bold]Total cost:[/bold] ${result.total_cost_usd:.4f}"
        if _rich_available and console is not None:
            console.print(line_total_cost)
        else:
            print_rich_or_typer(line_total_cost)

        try:
            total_tokens = int(getattr(result, "total_tokens", 0))
        except Exception:
            total_tokens = sum(getattr(s, "token_counts", 0) for s in result.step_history)
        line_total_tokens = f"[bold]Total tokens:[/bold] {total_tokens}"
        line_steps_executed = f"[bold]Steps executed:[/bold] {len(result.step_history)}"
        if _rich_available and console is not None:
            console.print(line_total_tokens)
            console.print(line_steps_executed)
        else:
            print_rich_or_typer(line_total_tokens)
            print_rich_or_typer(line_steps_executed)

        if show_steps and result.step_history:
            header = "\n[bold]Step Results:[/bold]"
            if _rich_available and console is not None:
                console.print(header)
                table = create_pipeline_results_table(
                    result.step_history,
                    show_output_column=show_output_column,
                    output_preview_len=output_preview_len,
                    include_steps=only_steps,
                )
                console.print(table)
            else:
                print_rich_or_typer(header)
                try:
                    include_set = set(only_steps) if only_steps else None
                except Exception:
                    include_set = None
                rows: list[str] = []
                headers = ["Step", "Status"]
                if show_output_column:
                    headers.append("Output")
                headers.extend(["Cost", "Tokens"])
                rows.append(" | ".join(headers))
                preview_len = (
                    100 if output_preview_len is None else max(-1, int(output_preview_len))
                )

                def add_plain_rows(step_res: Any, prefix: str = "") -> None:
                    name_attr = getattr(step_res, "step_name", None) or getattr(
                        step_res, "name", "<unknown>"
                    )
                    raw_name = name_attr
                    if include_set is not None and str(raw_name) not in include_set:
                        pass
                    else:
                        step_name = f"{prefix}{raw_name}" if prefix else raw_name
                        status = "✅" if getattr(step_res, "success", False) else "❌"
                        out_str = str(getattr(step_res, "output", ""))
                        if preview_len >= 0 and len(out_str) > preview_len:
                            out_str = out_str[:preview_len] + "..."
                        cost = (
                            f"${getattr(step_res, 'cost_usd'):.4f}"
                            if hasattr(step_res, "cost_usd")
                            else "N/A"
                        )
                        tokens = (
                            str(getattr(step_res, "token_counts"))
                            if hasattr(step_res, "token_counts")
                            else "N/A"
                        )
                        fields = [str(step_name), str(status)]
                        if show_output_column:
                            fields.append(str(out_str))
                        fields.extend([str(cost), str(tokens)])
                        rows.append(" | ".join(fields))

                    try:
                        nested_history = getattr(step_res, "step_history", None)
                        if isinstance(nested_history, list) and nested_history:
                            for child in nested_history:
                                add_plain_rows(child, prefix + "  ")
                    except Exception:
                        pass
                    try:
                        out = getattr(step_res, "output", None)
                        if (
                            out is not None
                            and hasattr(out, "step_history")
                            and isinstance(getattr(out, "step_history", None), list)
                            and getattr(out, "step_history")
                        ):
                            for child in getattr(out, "step_history"):
                                add_plain_rows(child, prefix + "  ")
                    except Exception:
                        pass

                for s in result.step_history:
                    add_plain_rows(s)
                print_rich_or_typer("\n".join(rows))

        if show_context and result.final_pipeline_context:
            header_ctx = "\n[bold]Final Context:[/bold]"
            payload = json.dumps(
                safe_serialize(result.final_pipeline_context.model_dump()),
                indent=2,
            )
            if _rich_available and console is not None:
                console.print(header_ctx)
                console.print(payload)
            else:
                print_rich_or_typer(header_ctx)
                print_rich_or_typer(payload)

        ctx = getattr(result, "final_pipeline_context", None)
        ctx_run_id = getattr(ctx, "run_id", None) if ctx is not None else None
        display_run_id = run_id or ctx_run_id or "N/A"
        line_run_id = f"[bold]Run ID:[/bold] {display_run_id}"
        if _rich_available and console is not None:
            console.print(line_run_id)
        else:
            print_rich_or_typer(line_run_id)

    if pager and _rich_available and console is not None:
        with console.pager(styles=True):
            _render_body()
    else:
        _render_body()


def get_version_string() -> str:
    """Return the installed flujo version or 'unknown' if not found."""
    try:
        return importlib_metadata.version("flujo")
    except (importlib_metadata.PackageNotFoundError, Exception):
        return "unknown"


def get_masked_settings_dict() -> Dict[str, Any]:
    """Return settings as a dict with sensitive keys masked/removed."""
    import flujo.cli.main as cli_main

    settings = cli_main.load_settings()
    return cast(Dict[str, Any], settings.model_dump(exclude={"openai_api_key", "logfire_api_key"}))


def execute_improve(
    pipeline_path: str,
    dataset_path: str,
    improvement_agent_model: Optional[str],
    json_output: bool,
) -> Optional[str]:
    """Run evaluation and improvement, printing output or returning JSON string."""
    import functools

    from rich.console import Console

    try:
        try:
            os.makedirs("output", exist_ok=True)
            with open("output/trace_improve.txt", "a") as f:
                f.write("stage:load_pipeline\n")
        except Exception:
            pass
        pipeline = load_pipeline_from_file(pipeline_path)[0]
        try:
            with open("output/trace_improve.txt", "a") as f:
                f.write("stage:load_dataset\n")
            dataset = load_dataset_from_file(dataset_path)
        except Exception:
            dataset = object()

        with open("output/trace_improve.txt", "a") as f:
            f.write("stage:build_runner\n")
        from flujo.application.runner import Flujo

        runner: Any = Flujo(pipeline)
        with open("output/trace_improve.txt", "a") as f:
            f.write("stage:import_run_pipeline_async\n")
        from flujo.cli.main import run_pipeline_async

        task_fn = functools.partial(run_pipeline_async, runner=runner)

        with open("output/trace_improve.txt", "a") as f:
            f.write("stage:import_eval_and_agent\n")
        from flujo.cli.main import (
            SelfImprovementAgent,
            evaluate_and_improve,
            make_self_improvement_agent,
        )
        from flujo.exceptions import ConfigurationError

        agent: Any
        try:
            _agent = make_self_improvement_agent(model=improvement_agent_model)
            agent = SelfImprovementAgent(_agent)
        except ConfigurationError:

            class _Dummy:
                pass

            agent = _Dummy()
        with open("output/trace_improve.txt", "a") as f:
            f.write("stage:run_eval\n")
        report = asyncio.run(
            evaluate_and_improve(task_fn, dataset, agent, pipeline_definition=pipeline)
        )

        if json_output:
            return json.dumps(safe_serialize(report.model_dump()), indent=2)

        with open("output/trace_improve.txt", "a") as f:
            f.write("stage:print\n")
        console = Console()
        console.print("[bold]IMPROVEMENT REPORT[/bold]")
        groups, table = create_improvement_report_table(report.suggestions)
        for step, suggestions in groups.items():
            console.print(f"\n[bold cyan]Suggestions for {step}[/bold cyan]")
            for s in suggestions:
                table.add_row(
                    s.failure_pattern_summary,
                    format_improvement_suggestion(s),
                    s.estimated_impact or "",
                    s.estimated_effort_to_implement or "",
                )
            console.print(table)
        return None
    except Exception as e:
        try:
            os.makedirs("output", exist_ok=True)
            with open("output/last_improve_error.txt", "w") as f:
                f.write(repr(e))
        except Exception:
            pass
        raise Exit(1)
