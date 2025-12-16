from __future__ import annotations

import re
from typing import Any, ClassVar, Iterable, Optional

from ..domain.pipeline_validation import ValidationFinding
from .linters_base import BaseLinter, _override_severity


class ExceptionLinter(BaseLinter):
    """Control flow exception handling validation: V-EX1."""

    CONTROL_FLOW_EXCEPTIONS: ClassVar[set[str]] = {
        "PausedException",
        "PipelineAbortSignal",
        "InfiniteRedirectError",
    }

    def analyze(self, pipeline: Any) -> Iterable[ValidationFinding]:
        out: list[ValidationFinding] = []
        steps = getattr(pipeline, "steps", []) or []

        _HITLStep: Optional[type[Any]] = None
        try:
            from ..domain.dsl.step import HumanInTheLoopStep as _HITLStep
        except Exception:
            _HITLStep = None

        for idx, step in enumerate(steps):
            try:
                meta = getattr(step, "meta", None)
                if not isinstance(meta, dict):
                    continue

                yloc = meta.get("_yaml_loc") if isinstance(meta, dict) else None
                loc_path = (yloc or {}).get("path") or f"steps[{idx}]"
                fpath = (yloc or {}).get("file")
                line = (yloc or {}).get("line")
                col = (yloc or {}).get("column")

                step_name = getattr(step, "name", None)
                agent_spec = getattr(step, "agent", None)

                uses_custom_skill = False
                skill_ref = None

                if "uses" in meta and isinstance(meta["uses"], str):
                    uses_val = meta["uses"]
                    if ":" in uses_val and not uses_val.startswith(
                        ("agents.", "imports.", "flujo.builtins.")
                    ):
                        uses_custom_skill = True
                        skill_ref = uses_val

                if not uses_custom_skill and agent_spec:
                    if hasattr(agent_spec, "_step_callable"):
                        wrapped_func = getattr(agent_spec, "_step_callable", None)
                        if wrapped_func:
                            module_name = getattr(wrapped_func, "__module__", None)
                            func_name = getattr(wrapped_func, "__name__", None)

                            if module_name and not module_name.startswith("flujo."):
                                uses_custom_skill = True
                                if func_name:
                                    skill_ref = f"{module_name}:{func_name}"
                                else:
                                    skill_ref = module_name
                    elif callable(agent_spec):
                        module_name = getattr(agent_spec, "__module__", None)
                        func_name = getattr(agent_spec, "__name__", None)

                        if module_name and not module_name.startswith("flujo."):
                            uses_custom_skill = True
                            if func_name:
                                skill_ref = f"{module_name}:{func_name}"
                            else:
                                skill_ref = module_name

                if uses_custom_skill and skill_ref:
                    sev = _override_severity("V-EX1", "warning")
                    if sev is not None:
                        out.append(
                            ValidationFinding(
                                rule_id="V-EX1",
                                severity=sev,
                                message=(
                                    f"Step '{step_name}' uses custom skill '{skill_ref}'. "
                                    f"Ensure it does not catch control flow exceptions "
                                    f"(PausedException, PipelineAbortSignal, InfiniteRedirectError) "
                                    f"without re-raising them."
                                ),
                                step_name=step_name,
                                location_path=loc_path,
                                file=fpath,
                                line=line,
                                column=col,
                                suggestion=(
                                    "In your custom skill, always re-raise control flow exceptions:\n"
                                    "try:\n"
                                    "    # your logic\n"
                                    "except (PausedException, PipelineAbortSignal, InfiniteRedirectError):\n"
                                    "    raise  # CRITICAL: must re-raise\n"
                                    "except Exception as e:\n"
                                    "    # handle other exceptions\n"
                                    "\n"
                                    "See: FLUJO_TEAM_GUIDE.md Section 2 'The Fatal Anti-Pattern'"
                                ),
                            )
                        )

            except Exception:
                continue

        return out


class LoopScopingLinter(BaseLinter):
    """Loop step scoping validation: LOOP-001."""

    def analyze(self, pipeline: Any) -> Iterable[ValidationFinding]:
        out: list[ValidationFinding] = []
        steps = getattr(pipeline, "steps", []) or []

        def _check_loop_body_steps(
            body_steps: list[Any], loop_name: str, _loop_meta: dict[str, Any]
        ) -> None:
            for _idx, step in enumerate(body_steps):
                try:
                    meta = getattr(step, "meta", {})
                    for field_name in ["condition_expression", "exit_expression"]:
                        expr = None

                        if hasattr(step, field_name):
                            expr = getattr(step, field_name, None)
                        elif isinstance(meta, dict) and field_name in meta:
                            expr = meta.get(field_name)

                        if isinstance(expr, str) and (
                            "steps[" in expr
                            or "steps.get(" in expr
                            or re.search(r"\bsteps\.", expr)
                        ):
                            yloc = meta.get("_yaml_loc") if isinstance(meta, dict) else None
                            loc_path = (yloc or {}).get(
                                "path"
                            ) or f"loop '{loop_name}' body steps[{_idx}].{field_name}"
                            fpath = (yloc or {}).get("file")
                            line = (yloc or {}).get("line")
                            col = (yloc or {}).get("column")

                            sev = _override_severity("LOOP-001", "warning")
                            if sev is not None:
                                out.append(
                                    ValidationFinding(
                                        rule_id="LOOP-001",
                                        severity=sev,
                                        message=(
                                            f"Step reference detected in {field_name} inside loop body '{loop_name}'. "
                                            f"Loop body steps are scoped to the current iteration and may not be "
                                            f"accessible via steps['name']."
                                        ),
                                        step_name=getattr(step, "name", None),
                                        location_path=loc_path,
                                        file=fpath,
                                        line=line,
                                        column=col,
                                        suggestion=(
                                            "Use 'previous_step' to reference the immediate previous step in the loop body.\n"
                                            "\n"
                                            "Example:\n"
                                            "  ❌ condition_expression: \"steps['process'].output.status == 'done'\"\n"
                                            "  ✅ condition_expression: \"previous_step.status == 'done'\"\n"
                                            "\n"
                                            "To access steps from outside the loop, use context to carry data."
                                        ),
                                    )
                                )

                    templ = meta.get("templated_input")
                    if isinstance(templ, str) and (
                        "steps[" in templ or "steps.get(" in templ or re.search(r"\bsteps\.", templ)
                    ):
                        yloc = meta.get("_yaml_loc") if isinstance(meta, dict) else None
                        loc_path = (yloc or {}).get(
                            "path"
                        ) or f"loop '{loop_name}' body steps[{_idx}].input"
                        fpath = (yloc or {}).get("file")
                        line = (yloc or {}).get("line")
                        col = (yloc or {}).get("column")

                        sev = _override_severity("LOOP-001", "warning")
                        if sev is not None:
                            out.append(
                                ValidationFinding(
                                    rule_id="LOOP-001",
                                    severity=sev,
                                    message=(
                                        f"Step reference detected in template inside loop body '{loop_name}'. "
                                        f"Loop body steps are scoped to the current iteration."
                                    ),
                                    step_name=getattr(step, "name", None),
                                    location_path=loc_path,
                                    file=fpath,
                                    line=line,
                                    column=col,
                                    suggestion=(
                                        "Use '{{ previous_step }}' to reference the immediate previous step.\n"
                                        "\n"
                                        "Example:\n"
                                        '  ❌ input: "{{ steps.process.output }}"\n'
                                        '  ✅ input: "{{ previous_step }}"\n'
                                    ),
                                )
                            )

                    if hasattr(step, "loop_body_pipeline"):
                        nested_pipeline = getattr(step, "loop_body_pipeline", None)
                        if nested_pipeline and hasattr(nested_pipeline, "steps"):
                            nested_steps = getattr(nested_pipeline, "steps", [])
                            nested_meta = getattr(step, "meta", {})
                            _check_loop_body_steps(
                                nested_steps, getattr(step, "name", "nested_loop"), nested_meta
                            )

                except Exception as e:
                    import logging

                    logging.getLogger(__name__).debug(f"Failed to validate loop body step: {e}")
                    continue

        for step in steps:
            try:
                if hasattr(step, "loop_body_pipeline"):
                    pipeline_body = getattr(step, "loop_body_pipeline", None)
                    if pipeline_body and hasattr(pipeline_body, "steps"):
                        body_steps = getattr(pipeline_body, "steps", [])
                        meta = getattr(step, "meta", {})
                        _check_loop_body_steps(body_steps, getattr(step, "name", "loop"), meta)

                if hasattr(step, "pipeline_to_run"):
                    pipeline_body = getattr(step, "pipeline_to_run", None)
                    if pipeline_body and hasattr(pipeline_body, "steps"):
                        body_steps = getattr(pipeline_body, "steps", [])
                        meta = getattr(step, "meta", {})
                        _check_loop_body_steps(body_steps, getattr(step, "name", "map"), meta)

            except Exception as e:
                import logging

                logging.getLogger(__name__).debug(f"Failed to validate step: {e}")
                continue

        return out


class TemplateControlStructureLinter(BaseLinter):
    """Template Jinja2 control structure validation: TEMPLATE-001."""

    def analyze(self, pipeline: Any) -> Iterable[ValidationFinding]:
        out: list[ValidationFinding] = []
        steps = getattr(pipeline, "steps", []) or []

        def _check_template_field(template_str: str, field_name: str, step: Any, idx: int) -> None:
            if not isinstance(template_str, str):
                return

            if "{%" in template_str or "%}" in template_str:
                meta = getattr(step, "meta", {})
                yloc = meta.get("_yaml_loc") if isinstance(meta, dict) else None
                loc_path = (yloc or {}).get("path") or f"steps[{idx}].{field_name}"
                fpath = (yloc or {}).get("file")
                line = (yloc or {}).get("line")
                col = (yloc or {}).get("column")

                control_match = re.search(r"\{%\s*([a-z]+)", template_str)
                control_name = control_match.group(1) if control_match else "unknown"

                sev = _override_severity("TEMPLATE-001", "error")
                if sev is not None:
                    out.append(
                        ValidationFinding(
                            rule_id="TEMPLATE-001",
                            severity=sev,
                            message=(
                                f"Unsupported Jinja2 control structure '{{%{control_name}%}}' detected in {field_name}. "
                                f"Flujo templates support expressions {{{{ }}}} and filters |, but NOT control structures {{%{control_name}%}}."
                            ),
                            step_name=getattr(step, "name", None),
                            location_path=loc_path,
                            file=fpath,
                            line=line,
                            column=col,
                            suggestion=(
                                "Alternatives:\n"
                                "  1. Use template filters: {{ context.items | join('\\n') }}\n"
                                '  2. Use custom skill: uses: "skills:format_data"\n'
                                "  3. Use conditional steps for if/else logic\n"
                                "  4. Pre-format data in a previous step\n"
                                "\n"
                                "Supported:\n"
                                "  ✅ {{ variable }}\n"
                                "  ✅ {{ value | filter }}\n"
                                "  ✅ {{ context.nested.field }}\n"
                                "\n"
                                "NOT Supported:\n"
                                "  ❌ {% for %}, {% if %}, {% set %}\n"
                                "  ❌ {% macro %}, {% include %}\n"
                                "\n"
                                "Documentation: https://flujo.dev/docs/templates"
                            ),
                        )
                    )

        for idx, step in enumerate(steps):
            try:
                meta = getattr(step, "meta", {})
                templ = meta.get("templated_input")
                if templ:
                    _check_template_field(templ, "input", step, idx)

                if hasattr(step, "message"):
                    message = getattr(step, "message", None)
                    if message:
                        _check_template_field(message, "message", step, idx)

                for field in ["description", "prompt", "system_prompt"]:
                    if hasattr(step, field):
                        value = getattr(step, field, None)
                        if value:
                            _check_template_field(value, field, step, idx)

            except Exception as e:
                import logging

                logging.getLogger(__name__).debug(f"Failed to validate template field: {e}")
                continue

        return out


class HitlNestedContextLinter(BaseLinter):
    """HITL nested context validation: HITL-NESTED-001 (CRITICAL)."""

    def analyze(self, pipeline: Any) -> Iterable[ValidationFinding]:
        out: list[ValidationFinding] = []

        try:
            from ..domain.dsl.step import HumanInTheLoopStep as _HITLStep
        except Exception as e:
            import logging

            logging.getLogger(__name__).debug(f"Failed to import HumanInTheLoopStep: {e}")
            _HITLStep = None  # type: ignore[misc,assignment]

        if _HITLStep is None:
            return out

        def _check_for_hitl_in_steps(
            steps: list[Any], context_chain: list[str], _parent_step_name: Optional[str] = None
        ) -> None:
            for _idx, step in enumerate(steps):
                try:
                    is_hitl = isinstance(step, _HITLStep)

                    if not is_hitl:
                        meta = getattr(step, "meta", {})
                        if isinstance(meta, dict):
                            kind = meta.get("kind", "")
                            is_hitl = kind == "hitl"

                    if is_hitl and len(context_chain) > 0:
                        loop_indices = [
                            i
                            for i, ctx in enumerate(context_chain)
                            if ctx.startswith(("loop:", "map:"))
                        ]
                        if not loop_indices:
                            continue

                        innermost_loop_idx = max(loop_indices)
                        conditional_inside_loop = any(
                            idx > innermost_loop_idx
                            and context_chain[idx].startswith("conditional:")
                            for idx in range(innermost_loop_idx + 1, len(context_chain))
                        )

                        if not conditional_inside_loop:
                            continue
                        meta = getattr(step, "meta", {})
                        yloc = meta.get("_yaml_loc") if isinstance(meta, dict) else None
                        loc_path = (yloc or {}).get(
                            "path"
                        ) or f"step '{getattr(step, 'name', 'unnamed')}'"
                        fpath = (yloc or {}).get("file")
                        line = (yloc or {}).get("line")
                        col = (yloc or {}).get("column")

                        context_desc = " > ".join(context_chain)

                        sev = _override_severity("HITL-NESTED-001", "error")
                        if sev is not None:
                            out.append(
                                ValidationFinding(
                                    rule_id="HITL-NESTED-001",
                                    severity=sev,
                                    message=(
                                        f"HITL step '{getattr(step, 'name', 'unnamed')}' will be SILENTLY SKIPPED at runtime. "
                                        f"Context: {context_desc}. "
                                        f"This is a known limitation: HITL steps in conditional branches inside loops do NOT execute. "
                                        f"The step will be filtered out silently with no error message, causing data loss."
                                    ),
                                    step_name=getattr(step, "name", None),
                                    location_path=loc_path,
                                    file=fpath,
                                    line=line,
                                    column=col,
                                    suggestion=(
                                        "CRITICAL: This pipeline will fail at runtime. Apply one of these workarounds:\n"
                                        "\n"
                                        "  1. Move HITL step outside the loop (RECOMMENDED)\n"
                                        "     - Collect user input before entering the loop\n"
                                        "     - Store result in context for use inside loop\n"
                                        "\n"
                                        "  2. Remove the conditional wrapper\n"
                                        "     - If the HITL must be in a loop, remove the conditional\n"
                                        "     - HITL directly in loop body should work\n"
                                        "\n"
                                        "  3. Use flujo.builtins.ask_user skill instead\n"
                                        "     - Built-in skills may have better nested context support\n"
                                        "\n"
                                        "Example fix:\n"
                                        "  # ❌ WILL FAIL - HITL in conditional in loop\n"
                                        "  - kind: loop\n"
                                        "    body:\n"
                                        "      - kind: conditional\n"
                                        "        branches:\n"
                                        "          true:\n"
                                        "            - kind: hitl  # <- SILENTLY SKIPPED!\n"
                                        "              message: 'Question?'\n"
                                        "\n"
                                        "  # ✅ WORKS - HITL at top-level\n"
                                        "  - kind: hitl\n"
                                        "    name: get_input\n"
                                        "    message: 'Question?'\n"
                                        "    sink_to: 'user_answer'\n"
                                        "  - kind: loop\n"
                                        "    body:\n"
                                        "      - kind: step\n"
                                        "        input: '{{ context.user_answer }}'\n"
                                        "\n"
                                        "Documentation: https://flujo.dev/docs/known-issues/hitl-nested\n"
                                        "Report hours lost debugging this? https://github.com/aandresalvarez/flujo/issues"
                                    ),
                                )
                            )

                    if hasattr(step, "loop_body_pipeline"):
                        loop_pipeline = getattr(step, "loop_body_pipeline", None)
                        if loop_pipeline and hasattr(loop_pipeline, "steps"):
                            nested_steps = getattr(loop_pipeline, "steps", [])
                            step_name = getattr(step, "name", "loop")
                            new_chain = [*context_chain, f"loop:{step_name}"]
                            _check_for_hitl_in_steps(nested_steps, new_chain, step_name)

                    if hasattr(step, "pipeline_to_run"):
                        map_pipeline = getattr(step, "pipeline_to_run", None)
                        if map_pipeline and hasattr(map_pipeline, "steps"):
                            nested_steps = getattr(map_pipeline, "steps", [])
                            step_name = getattr(step, "name", "map")
                            new_chain = [*context_chain, f"map:{step_name}"]
                            _check_for_hitl_in_steps(nested_steps, new_chain, step_name)

                    if hasattr(step, "branches"):
                        branches = getattr(step, "branches", {})
                        if isinstance(branches, dict):
                            step_name = getattr(step, "name", "conditional")
                            for branch_key, branch_pipeline in branches.items():
                                if branch_pipeline and hasattr(branch_pipeline, "steps"):
                                    branch_steps = getattr(branch_pipeline, "steps", [])
                                    new_chain = [
                                        *context_chain,
                                        f"conditional:{step_name}",
                                        f"branch:{branch_key}",
                                    ]
                                    _check_for_hitl_in_steps(branch_steps, new_chain, step_name)

                    if hasattr(step, "parallel_branches"):
                        parallel_branches = getattr(step, "parallel_branches", {})
                        if isinstance(parallel_branches, dict):
                            step_name = getattr(step, "name", "parallel")
                            for branch_name, branch_pipeline in parallel_branches.items():
                                if branch_pipeline and hasattr(branch_pipeline, "steps"):
                                    branch_steps = getattr(branch_pipeline, "steps", [])
                                    new_chain = [
                                        *context_chain,
                                        f"parallel:{step_name}",
                                        f"branch:{branch_name}",
                                    ]
                                    _check_for_hitl_in_steps(branch_steps, new_chain, step_name)

                except Exception as e:
                    import logging

                    logging.getLogger(__name__).debug(
                        f"Failed to validate HITL nested context: {e}"
                    )
                    continue

        steps = getattr(pipeline, "steps", []) or []
        _check_for_hitl_in_steps(steps, [], None)

        return out
