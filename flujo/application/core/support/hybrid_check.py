from __future__ import annotations

import inspect

from flujo.domain.plugins import PluginOutcome
from flujo.domain.base_model import BaseModel as DomainBaseModel
from flujo.domain.validation import Validator


async def run_hybrid_check(
    data: object,
    plugins: list[tuple[object, int]],
    validators: list[Validator],
    context: object | None = None,
    resources: object | None = None,
) -> tuple[object, str | None]:
    """
    Run plugins then validators in sequence and return a tuple:
      (possibly-transformed data, aggregated failure feedback or None).

    PluginOutcome failures are collected into feedback; ValidationResult failures
    are collected; combined with "; " between them.
    Plugin exceptions raise PluginError.
    """
    # 1. Plugins
    output = data
    plugin_feedbacks: list[str] = []
    for plugin, priority in sorted(plugins, key=lambda x: x[1], reverse=True):
        plugin_input: dict[str, object]
        if isinstance(output, dict):
            plugin_input = dict(output)
        else:
            plugin_input = {"output": output}
        plugin_kwargs: dict[str, object] = {}
        try:
            validate_fn = getattr(plugin, "validate", None)
        except Exception:
            validate_fn = None
        if not callable(validate_fn):
            from ..execution.executor_helpers import PluginError

            raise PluginError("Plugin missing validate()")
        try:
            sig = inspect.signature(validate_fn)
            if "context" in sig.parameters and context is not None:
                plugin_kwargs["context"] = context
            if "resources" in sig.parameters and resources is not None:
                plugin_kwargs["resources"] = resources
        except Exception:
            plugin_kwargs = plugin_kwargs
        try:
            res = validate_fn(plugin_input, **plugin_kwargs)
            result = await res if inspect.isawaitable(res) else res
        except Exception as e:
            from ..execution.executor_helpers import PluginError

            raise PluginError(str(e)) from e
        if isinstance(result, PluginOutcome):
            if not result.success:
                if result.feedback is not None:
                    plugin_feedbacks.append(result.feedback)
            else:
                if result.new_solution is not None:
                    output = result.new_solution
        else:
            output = result
    # 2. Validators
    failed_msgs: list[str] = []
    if validators:
        from flujo.domain.validation import ValidationResult

        validator_context: DomainBaseModel | None = (
            context if isinstance(context, DomainBaseModel) else None
        )
        for validator in validators:
            try:
                vr: ValidationResult = await validator.validate(output, context=validator_context)
            except Exception as e:
                failed_msgs.append(f"{validator.name}: {e}")
                continue
            if not vr.is_valid:
                failed_msgs.append(f"{vr.validator_name}: {vr.feedback}")
    all_msgs = plugin_feedbacks + failed_msgs
    if all_msgs:
        return output, "; ".join(all_msgs)
    return output, None
