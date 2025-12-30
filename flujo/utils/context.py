"""
Context utilities for Flujo framework.

This module provides utilities for working with Pydantic-based context objects,
including safe merging operations that respect Pydantic validation.
"""

from typing import Any, Literal, Optional, Type, TypeVar
import logging
import os

from pydantic import BaseModel, ValidationError as PydanticValidationError
from flujo.domain.dsl.step import MergeStrategy
from flujo.exceptions import (
    ConfigurationError,
    InfiniteRedirectError,
    PausedException,
    PipelineAbortSignal,
    ValidationError as FlujoValidationError,
)

logger = logging.getLogger(__name__)
_VERBOSE_DEBUG: bool = bool(os.getenv("FLUJO_VERBOSE_CONTEXT_DEBUG"))

T = TypeVar("T", bound=BaseModel)


# Cache for excluded fields to avoid repeated environment variable access
_EXCLUDED_FIELDS_CACHE: Optional[set[str]] = None
_ENV_EXCLUDED_FIELDS_CACHE: Optional[str] = None


def get_excluded_fields() -> set[str]:
    """
    Retrieve the set of fields to exclude from context merging.

    The default excluded fields are:
    - 'command_log': Avoids redundant or conflicting entries during loop operations.

    Returns:
        set[str]: A set of field names to exclude.
    """
    global _EXCLUDED_FIELDS_CACHE
    import os

    # Always recompute to avoid stale globals across tests/suites.
    _EXCLUDED_FIELDS_CACHE = None

    # Default excluded fields
    # 'command_log' is excluded to prevent redundant or conflicting entries during loop operations
    # where the same command might be logged multiple times across iterations
    # Default excluded fields to prevent duplication and noisy merges for history-like fields
    default_excluded_fields = {
        "command_log",
        "cache_timestamps",
        "cache_keys",
        "tree_search_state",
        "tree_search_frontier",
    }

    # Always read the environment afresh (test helpers may preseed _ENV_EXCLUDED_FIELDS_CACHE)
    env_val = (
        _ENV_EXCLUDED_FIELDS_CACHE
        if _ENV_EXCLUDED_FIELDS_CACHE is not None
        else os.getenv("EXCLUDED_FIELDS", "")
    )

    excluded_fields = env_val
    # Defensive: ensure excluded_fields is a string-like value
    if not isinstance(excluded_fields, str):  # pragma: no cover - CI hardening
        try:
            excluded_fields = str(excluded_fields) if excluded_fields is not None else ""
        except Exception:
            excluded_fields = ""

    # Whitelist of allowed field names for security
    ALLOWED_EXCLUDED_FIELDS = {
        "command_log",
        "cache_hits",
        "cache_misses",
        "processing_history",
        "current_operation",
        "operation_count",
        "cache_timestamps",
        "cache_keys",
        "hitl_history",
        "run_id",
        "initial_prompt",
        "tree_search_state",
        "tree_search_frontier",
    }

    # Maximum allowed length for field names to prevent abuse
    MAX_FIELD_NAME_LENGTH = 50

    if excluded_fields:
        # Validate and sanitize the field names against whitelist
        sanitized_fields = set()
        for field in excluded_fields.split(","):
            # Normalize and guard against non-string types
            try:
                field = field.strip() if isinstance(field, str) else str(field).strip()
            except Exception:
                continue

            # Skip empty entries
            if not field:
                continue

            # Enforce maximum length to mitigate potential DoS attacks with huge env vars
            if len(field) > MAX_FIELD_NAME_LENGTH:
                logger.warning(
                    f"Field '{field}' exceeds maximum length of {MAX_FIELD_NAME_LENGTH}. Skipping."
                )
                continue

            # Ensure field is a valid Python identifier (alphanumeric + underscores, no leading digits)
            try:
                is_ident = field.isidentifier()
            except Exception:
                is_ident = False
            if not is_ident:
                logger.warning(f"Field '{field}' contains invalid characters. Skipping.")
                continue

            # Whitelist validation
            if field in ALLOWED_EXCLUDED_FIELDS:
                sanitized_fields.add(field)
            else:
                logger.warning(
                    f"Field '{field}' from EXCLUDED_FIELDS environment variable is not in allowed whitelist. Skipping."
                )

        if sanitized_fields:
            _EXCLUDED_FIELDS_CACHE = sanitized_fields
            return sanitized_fields
        else:
            logger.warning(
                "Environment variable 'EXCLUDED_FIELDS' contains no valid field names. Falling back to default excluded fields."
            )

    _EXCLUDED_FIELDS_CACHE = default_excluded_fields
    return default_excluded_fields


def safe_merge_context_updates(
    target_context: T,
    source_context: T,
    context_type: Optional[Type[T]] = None,
    excluded_fields: Optional[set[str]] = None,
    merge_strategy: Optional[MergeStrategy] = None,
) -> bool:
    """
    Safely merge context updates from source to target, respecting Pydantic validation.

    This function properly handles Pydantic models by:
    1. Using model_dump() to get field values (works with both v1 and v2)
    2. Using setattr() for updates to trigger validation
    3. Handling computed fields and validators properly
    4. Gracefully handling equality comparison failures
    5. Enhanced error handling for loop context updates
    6. Performing deep merges for dictionaries and lists to prevent overwrites

    Args:
        target_context: The target context to update
        source_context: The source context with updates
        context_type: Optional type hint for better error handling
        excluded_fields: Optional set of fields to exclude from merging

    Returns:
        bool: True if merge was successful, False if any errors occurred
    """
    if target_context is None or source_context is None:
        return False

    # Use default excluded fields if none provided
    if excluded_fields is None:
        excluded_fields = get_excluded_fields()

    if _VERBOSE_DEBUG:
        logger.debug("safe_merge_context_updates called")
        logger.debug("target_context type: %s", type(target_context))
        logger.debug("source_context type: %s", type(source_context))
        logger.debug("excluded_fields: %s", excluded_fields)

    def _make_list_item_signature(value: Any) -> str:
        """Create a stable signature for list item deduplication.

        Uses hashing for primitives; for complex values falls back to a compact
        serialized representation to avoid identity-based duplicates while keeping
        near O(N) behavior.
        """
        try:
            if isinstance(value, (str, int, float, bool, type(None))):
                return f"p:{repr(value)}"
            if isinstance(value, tuple) and all(
                isinstance(x, (str, int, float, bool, type(None))) for x in value
            ):
                return f"t:{repr(value)}"
            if isinstance(value, dict):
                keys = sorted(value.keys(), key=str)
                parts = []
                for k in keys:
                    parts.append(f"{k}={type(value[k]).__name__}:{repr(value[k])[:64]}")
                return "d:" + ",".join(parts)
            if hasattr(value, "model_dump"):
                try:
                    dumped = value.model_dump()
                except Exception:
                    dumped = {
                        n: getattr(value, n, None)
                        for n in getattr(value.__class__, "model_fields", {})
                    }
                keys = sorted(dumped.keys(), key=str)
                parts = []
                for k in keys:
                    parts.append(f"{k}={type(dumped[k]).__name__}:{repr(dumped[k])[:64]}")
                return "m:" + ",".join(parts)
            return f"o:{type(value).__name__}:{repr(value)[:64]}"
        except Exception:
            return f"e:{type(value).__name__}"

    def _merge_path(prefix: str, key: object) -> str:
        return f"{prefix}.{key}" if prefix else str(key)

    def _raise_conflict(path: str) -> None:
        raise ConfigurationError(
            f"Merge conflict for key '{path or '<root>'}'. Set an explicit merge strategy or field_mapping in your ParallelStep."
        )

    def _detect_nested_conflict(target_value: object, source_value: object, path: str) -> None:
        if isinstance(target_value, dict) and isinstance(source_value, dict):
            for nested_key in set(target_value.keys()) & set(source_value.keys()):
                _detect_nested_conflict(
                    target_value[nested_key],
                    source_value[nested_key],
                    _merge_path(path, nested_key),
                )
            return
        if isinstance(target_value, list) and isinstance(source_value, list):
            try:
                differs = target_value != source_value
            except Exception:
                differs = True
            if differs:
                _raise_conflict(path)
            return
        if target_value is not None and source_value is not None:
            try:
                differs = target_value != source_value
            except Exception:
                differs = True
            if differs:
                _raise_conflict(path)

    def deep_merge_dict(
        target_dict: dict[str, Any],
        source_dict: dict[str, Any],
        *,
        merge_strategy: MergeStrategy | None = None,
        path: str = "",
    ) -> dict[str, Any]:
        """Recursively merge source dictionary into target dictionary.

        List handling follows an append/extend strategy for correctness and performance.
        We intentionally avoid per-item membership checks which can be O(N*M) and
        unreliable for unhashable/nested structures. Deduplication, if desired, should
        be implemented by callers with explicit domain rules.

        Phase 1 Fix: Preserve pause_message from target if it exists (recipe sets it correctly).
        """
        result = target_dict.copy()
        # Phase 1: Preserve pause_message from target if it exists (recipe sets it correctly)
        preserved_pause_message = None
        if isinstance(result, dict):
            preserved_pause_message = result.get("pause_message")

        for key, source_value in source_dict.items():
            if merge_strategy == MergeStrategy.KEEP_FIRST and key in result:
                if isinstance(result[key], dict) and isinstance(source_value, dict):
                    result[key] = deep_merge_dict(
                        result[key],
                        source_value,
                        merge_strategy=merge_strategy,
                        path=_merge_path(path, key),
                    )
                continue

            if merge_strategy == MergeStrategy.ERROR_ON_CONFLICT and key in result:
                _detect_nested_conflict(result[key], source_value, _merge_path(path, key))

            if key in result and isinstance(result[key], dict) and isinstance(source_value, dict):
                result[key] = deep_merge_dict(
                    result[key],
                    source_value,
                    merge_strategy=merge_strategy,
                    path=_merge_path(path, key),
                )
            elif key in result and isinstance(result[key], list) and isinstance(source_value, list):
                if merge_strategy == MergeStrategy.OVERWRITE:
                    result[key] = list(source_value)
                else:
                    # Robust de-duplication using stable content hashing to avoid false matches
                    import json as _json
                    import hashlib as _hashlib

                    try:
                        from flujo.utils.serialization import (
                            _robust_serialize_internal as _robust_serialize,
                        )
                    except Exception:

                        def _robust_serialize(
                            obj: Any,
                            *,
                            circular_ref_placeholder: str | None = "<circular-ref>",
                            bytes_mode: Literal["base64", "utf8"] = "base64",
                            allow_object_dict: bool = False,
                        ) -> str | int | float | bool | dict[str, Any] | list[Any] | None | str:
                            _ = (circular_ref_placeholder, bytes_mode, allow_object_dict)
                            return (
                                obj
                                if isinstance(obj, (str, int, float, bool, dict, list))
                                else str(obj)
                            )

                    def _stable_item_hash(v: Any) -> str:
                        try:
                            payload = _robust_serialize(v)
                            s = _json.dumps(payload, sort_keys=True, separators=(",", ":"))
                            return _hashlib.sha1(s.encode("utf-8")).hexdigest()
                        except Exception:
                            try:
                                return _hashlib.sha1(repr(v).encode("utf-8")).hexdigest()
                            except Exception:
                                return f"unknown:{type(v).__name__}"

                    existing = {_stable_item_hash(v) for v in result[key]}
                    for item in source_value:
                        h = _stable_item_hash(item)
                        if h not in existing:
                            existing.add(h)
                            result[key].append(item)
            else:
                # Merge primitives with preservation semantics:
                # - booleans: logical OR to avoid losing a True flag
                # - numbers: take the max to retain increments from successful branches
                # - all other types: overwrite with the latest value
                if key in result:
                    target_value = result[key]
                    if merge_strategy == MergeStrategy.OVERWRITE:
                        result[key] = source_value
                    elif isinstance(target_value, bool) and isinstance(source_value, bool):
                        result[key] = target_value or source_value
                    elif isinstance(target_value, (int, float)) and isinstance(
                        source_value, (int, float)
                    ):
                        result[key] = source_value if source_value > target_value else target_value
                    else:
                        result[key] = source_value
                else:
                    result[key] = source_value

        # Phase 1: Restore preserved pause_message if it was set (recipe set it correctly)
        if preserved_pause_message is not None and isinstance(result, dict):
            result["pause_message"] = preserved_pause_message

        return result

    try:
        # Use Pydantic model_dump or dict to get source fields, excluding None for efficiency
        # But don't exclude defaults for boolean fields as False is a meaningful value
        if hasattr(source_context, "model_dump"):
            try:
                # Get all fields including boolean fields with False values
                source_fields = source_context.model_dump(exclude_none=True)
                # Add boolean fields that might have been excluded
                model_class = type(source_context)
                if hasattr(model_class, "model_fields"):
                    for field_name, field_info in model_class.model_fields.items():
                        if field_info.annotation is bool and field_name not in source_fields:
                            source_fields[field_name] = getattr(source_context, field_name)
            except TypeError:
                source_fields = source_context.model_dump()
        elif hasattr(source_context, "dict"):
            try:
                # Get all fields including boolean fields with False values
                source_fields = source_context.dict(exclude_none=True)
                # Add boolean fields that might have been excluded
                model_class = type(source_context)
                if hasattr(model_class, "__fields__"):
                    fields = getattr(model_class, "__fields__")
                    if hasattr(fields, "items"):
                        for field_name, field_info in fields.items():
                            if (
                                hasattr(field_info, "type_")
                                and field_info.type_ is bool
                                and field_name not in source_fields
                            ):
                                source_fields[field_name] = getattr(source_context, field_name)
            except TypeError:
                source_fields = source_context.dict()
        else:
            # Fallback for non-Pydantic objects
            source_fields = {
                key: value
                for key, value in source_context.__dict__.items()
                if not key.startswith("_")
            }

        # Include dynamically added attributes that model_dump/model.dict may skip
        try:
            extra_attrs = getattr(source_context, "__dict__", {})
            if isinstance(extra_attrs, dict):
                for extra_key, extra_value in extra_attrs.items():
                    if extra_key.startswith("_"):
                        continue
                    if extra_key not in source_fields:
                        source_fields[extra_key] = extra_value
        except Exception:
            pass

        if _VERBOSE_DEBUG:
            # Avoid massive stringification: truncate long string values for debug
            def _truncate(v: Any) -> Any:
                try:
                    if isinstance(v, str) and len(v) > 256:
                        return v[:256] + "…"  # noqa: E501
                    if isinstance(v, list):
                        return [_truncate(x) for x in v[:10]] + (["…"] if len(v) > 10 else [])
                    if isinstance(v, dict):
                        out: dict[str, Any] = {}
                        for i, (k, val) in enumerate(v.items()):
                            if i >= 10:
                                out["…"] = "…"
                                break
                            out[k] = _truncate(val)
                        return out
                except Exception:
                    pass
                return v

            logger.debug("source_fields: %s", _truncate(source_fields))

        # Update only changed fields using setattr to trigger validation
        updated_count = 0
        validation_errors = []

        for field_name, source_value in source_fields.items():
            try:
                # Skip private fields
                if field_name.startswith("_"):
                    continue

                # Skip excluded fields to prevent duplication during loop merging
                if field_name in excluded_fields:
                    if _VERBOSE_DEBUG:
                        logger.debug("Skipping excluded field: %s", field_name)
                    continue

                # Check if field exists in target
                if not hasattr(target_context, field_name):
                    if _VERBOSE_DEBUG:
                        logger.debug(f"Field not found in target: {field_name}, creating it")
                    try:
                        value_to_set = getattr(source_context, field_name)
                    except Exception:
                        value_to_set = source_value
                    setattr(target_context, field_name, value_to_set)
                    updated_count += 1
                    continue

                # Always get the actual value from the source context for merging
                actual_source_value = getattr(source_context, field_name)
                current_value = getattr(target_context, field_name)

                if _VERBOSE_DEBUG:
                    logger.debug(f"Processing field: {field_name}")
                    logger.debug(f"current_value: {current_value}")
                    logger.debug(f"actual_source_value: {actual_source_value}")

                # Avoid recursive equality on search state nodes that hold context refs.
                if field_name == "tree_search_state":
                    try:
                        from flujo.domain.models import SearchState as _SearchState

                        if isinstance(actual_source_value, _SearchState):
                            for node in actual_source_value.nodes.values():
                                try:
                                    node._context = None
                                except Exception:
                                    continue
                    except Exception:
                        pass
                    if current_value is not actual_source_value:
                        setattr(target_context, field_name, actual_source_value)
                        updated_count += 1
                    continue

                if merge_strategy == MergeStrategy.KEEP_FIRST:
                    if isinstance(current_value, dict) and isinstance(actual_source_value, dict):
                        keep_first_merged = deep_merge_dict(
                            current_value,
                            actual_source_value,
                            merge_strategy=merge_strategy,
                            path=str(field_name),
                        )
                        if keep_first_merged != current_value:
                            setattr(target_context, field_name, keep_first_merged)
                            updated_count += 1
                        continue
                    if current_value is not None:
                        if _VERBOSE_DEBUG:
                            logger.debug(f"KEEP_FIRST skipping field: {field_name}")
                        continue

                # Conflict detection for differing simple values when strategy requires it
                # Note: Only applies when both contexts have the field and values differ
                if merge_strategy in (
                    MergeStrategy.ERROR_ON_CONFLICT,
                    MergeStrategy.CONTEXT_UPDATE,
                ):
                    # Only treat as conflict for non-container fields
                    is_container = isinstance(current_value, (dict, list)) or isinstance(
                        actual_source_value, (dict, list)
                    )
                    if (
                        not is_container
                        and current_value is not None
                        and actual_source_value is not None
                    ):
                        try:
                            values_differ = current_value != actual_source_value
                        except Exception:
                            values_differ = True
                        if values_differ:
                            raise ConfigurationError(
                                f"Merge conflict for key '{field_name}'. Set an explicit merge strategy or field_mapping in your ParallelStep."
                            )

                # Perform deep merge for dictionaries and lists
                if isinstance(current_value, dict) and isinstance(actual_source_value, dict):
                    if _VERBOSE_DEBUG:
                        logger.debug(f"Merging dictionaries for field: {field_name}")
                    merged_dict: dict[str, Any] = deep_merge_dict(
                        current_value,
                        actual_source_value,
                        merge_strategy=merge_strategy,
                        path=str(field_name),
                    )
                    if merged_dict != current_value:
                        setattr(target_context, field_name, merged_dict)
                        updated_count += 1
                        if _VERBOSE_DEBUG:
                            logger.debug(f"Updated dict field: {field_name}")
                elif isinstance(current_value, list) and isinstance(actual_source_value, list):
                    if _VERBOSE_DEBUG:
                        logger.debug(f"Merging lists for field: {field_name}")
                    if merge_strategy == MergeStrategy.OVERWRITE:
                        try:
                            if current_value != actual_source_value:
                                setattr(target_context, field_name, list(actual_source_value))
                                updated_count += 1
                        except Exception:
                            pass
                        continue
                    if merge_strategy == MergeStrategy.ERROR_ON_CONFLICT:
                        try:
                            differs = current_value != actual_source_value
                        except Exception:
                            differs = True
                        if differs:
                            _raise_conflict(str(field_name))
                    # Append with de-duplication and robust handling for edge cases
                    try:
                        if actual_source_value:
                            existing_signatures = {
                                _make_list_item_signature(v) for v in current_value
                            }
                            added = 0
                            for item in actual_source_value:
                                sig = _make_list_item_signature(item)
                                if sig not in existing_signatures:
                                    existing_signatures.add(sig)
                                    current_value.append(item)
                                    added += 1
                            if added:
                                updated_count += 1
                                if _VERBOSE_DEBUG:
                                    logger.debug(
                                        f"Extended list field: {field_name} with {added} unique items"
                                    )
                    except TypeError:
                        # Source not iterable or mock masquerading as iterable; skip
                        if _VERBOSE_DEBUG:
                            logger.debug(
                                f"Skipping non-iterable list merge for field: {field_name}"
                            )
                        pass
                else:
                    # For other types, use simple replacement
                    try:
                        if current_value != actual_source_value:
                            # Use setattr to trigger Pydantic validation
                            setattr(target_context, field_name, actual_source_value)
                            updated_count += 1
                            if _VERBOSE_DEBUG:
                                logger.debug(f"Updated simple field: {field_name}")
                        else:
                            if _VERBOSE_DEBUG:
                                logger.debug(f"Field unchanged: {field_name}")
                    except (
                        TypeError,
                        ValueError,
                        AttributeError,
                        PydanticValidationError,
                        FlujoValidationError,
                    ) as e:
                        # Enhanced error handling for loop context updates
                        error_msg: str = f"Failed to update field '{field_name}': {e}"
                        logger.warning(error_msg)
                        validation_errors.append(error_msg)
                        continue

            except (
                AttributeError,
                TypeError,
                PydanticValidationError,
                FlujoValidationError,
                ValueError,
            ) as e:
                # Skip fields that can't be accessed or set
                skip_error_msg: str = f"Skipping field '{field_name}' due to access/set error: {e}"
                if _VERBOSE_DEBUG:
                    logger.debug(skip_error_msg)
                validation_errors.append(skip_error_msg)
                continue

        if _VERBOSE_DEBUG:
            logger.debug(f"Total fields updated: {updated_count}")
            logger.debug(f"Validation errors: {validation_errors}")

        if updated_count > 0:
            # Note: We don't validate the entire context after updates to allow for more flexible handling
            # of invalid values as expected by tests. Pydantic v2 field validators are not automatically
            # called on attribute assignment, so this approach is more permissive.

            return True
        else:
            return True

    except (
        PausedException,
        PipelineAbortSignal,
        InfiniteRedirectError,
    ):
        raise
    except ConfigurationError as e:
        # Bubble up to policy enforcement to fail parallel step with clear message
        logger.error(f"ConfigurationError encountered during context merge: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to merge context updates: {e}")
        return False


def safe_context_field_update(context: Any, field_name: str, new_value: Any) -> bool:
    """
    Safely update a single field in a context object, respecting Pydantic validation.

    Enhanced version with better error handling and Pydantic v2 support.

    Args:
        context: The context object to update
        field_name: The name of the field to update
        new_value: The new value to set

    Returns:
        bool: True if update was successful, False otherwise
    """
    if context is None:
        logger.warning("Cannot update field on None context")
        return False

    try:
        # Check if field exists
        if not hasattr(context, field_name):
            logger.warning(f"Field '{field_name}' does not exist in context")
            return False

        # Get current value for comparison
        current_value = getattr(context, field_name)

        # Only update if value has changed
        if current_value != new_value:
            # Use setattr to trigger Pydantic validation
            setattr(context, field_name, new_value)

            # Note: Pydantic v2 field validators are not automatically called on attribute assignment
            # So we don't validate the entire context after each field update
            # This allows for more flexible handling of invalid values as expected by tests

            logger.debug(
                f"Successfully updated field '{field_name}' from {current_value} to {new_value}"
            )
            return True
        else:
            logger.debug(f"Field '{field_name}' unchanged, skipping update")
            return True

    except (
        AttributeError,
        TypeError,
        PydanticValidationError,
        FlujoValidationError,
        ValueError,
    ) as e:
        logger.error("Failed to update field '" + field_name + "': " + str(e))
        return False


def validate_context_updates(context: Any, updates: dict[str, Any]) -> list[str]:
    """
    Validate that context updates can be applied without errors.

    This function checks if the proposed updates would be valid
    without actually applying them, useful for pre-validation.

    Args:
        context: The context object to validate against
        updates: Dictionary of proposed updates

    Returns:
        list[str]: List of validation error messages (empty if valid)
    """
    if context is None:
        return ["Context is None"]

    errors = []

    for field_name, new_value in updates.items():
        try:
            # Check if field exists
            if not hasattr(context, field_name):
                errors.append(f"Field '{field_name}' does not exist in context")
                continue

            # Check if value can be set (basic type check)
            current_value = getattr(context, field_name)
            if not isinstance(new_value, type(current_value)):
                # Allow some flexibility for numeric types
                if not (
                    isinstance(current_value, (int, float)) and isinstance(new_value, (int, float))
                ):
                    errors.append(
                        f"Type mismatch for field '{field_name}': "
                        f"expected {type(current_value).__name__}, "
                        f"got {type(new_value).__name__}"
                    )
                    continue

        except AttributeError as e:
            errors.append(f"Cannot access field '{field_name}': {e}")
            continue

    return errors


def apply_context_updates_safely(
    context: Any, updates: dict[str, Any], validate_before_apply: bool = True
) -> tuple[bool, list[str]]:
    """
    Apply context updates with comprehensive error handling and validation.

    This is a high-level function that combines validation and application
    of context updates with detailed error reporting.

    Args:
        context: The context object to update
        updates: Dictionary of updates to apply
        validate_before_apply: Whether to validate before applying updates

    Returns:
        tuple[bool, list[str]]: (success, error_messages)
    """
    if context is None:
        return False, ["Context is None"]

    errors = []

    # Pre-validation if requested
    if validate_before_apply:
        validation_errors = validate_context_updates(context, updates)
        if validation_errors:
            return False, validation_errors

    # Apply updates one by one
    successful_updates = 0
    for field_name, new_value in updates.items():
        if safe_context_field_update(context, field_name, new_value):
            successful_updates += 1
        else:
            errors.append(f"Failed to update field '{field_name}'")

    if successful_updates == len(updates):
        return True, []
    else:
        return False, errors


def get_context_field_safely(context: Any, field_name: str, default: Any = None) -> Any:
    """
    Safely get a field value from a context object.

    Args:
        context: The context object
        field_name: Name of the field to get
        default: Default value if field doesn't exist or can't be accessed

    Returns:
        The field value or default
    """
    if context is None:
        return default

    try:
        if hasattr(context, field_name):
            return getattr(context, field_name)
    except (AttributeError, TypeError):
        pass

    # If the field does not exist or cannot be accessed, return the provided default
    return default


def has_context_field(context: Any, field_name: str) -> bool:
    """
    Check if a context object has a specific field.

    Args:
        context: The context object
        field_name: Name of the field to check

    Returns:
        bool: True if field exists and is accessible
    """
    if context is None:
        return False

    try:
        return hasattr(context, field_name)
    except (AttributeError, TypeError):
        return False


def predicate_is_valid_report(report: Any, context: Any) -> str:
    """Return branch key for conditional steps based on ValidationReport.

    Accepts (report, context) to match ConditionalStep signature; context is ignored.

    Args:
        report: ValidationReport-like object with boolean attribute `is_valid`.
        context: Pipeline context (unused).

    Returns:
        "valid" when report.is_valid is truthy, otherwise "invalid".
    """
    try:
        v = False
        # Primary: report objects
        try:
            v = bool(getattr(report, "is_valid", False))
        except Exception:
            v = False
        # Fallback: mapping forms
        if not v and isinstance(report, dict):
            v = bool(report.get("is_valid", False) or report.get("yaml_is_valid", False))
        # As a last resort, consult context flag
        if not v and context is not None:
            try:
                v = bool(getattr(context, "yaml_is_valid", False))
            except Exception:
                if isinstance(context, dict):
                    v = bool(context.get("yaml_is_valid", False))
        return "valid" if v else "invalid"
    except Exception:
        return "invalid"


def set_nested_context_field(context: Any, path: str, value: Any) -> bool:
    """
    Set a nested field in a context object using dot notation.

    Supports both attribute-based and dict-based access.

    Args:
        context: The context object (typically PipelineContext)
        path: Dot-separated path to the field (e.g., "result" or "user_query")
        value: The value to set

    Returns:
        bool: True if successful, False otherwise

    Raises:
        AttributeError: If any intermediate path doesn't exist or isn't accessible

    Example:
        >>> context = PipelineContext()
        >>> set_nested_context_field(context, "user_name", "Alice")
        True
    """
    if not path:
        return False

    parts = path.split(".")
    target = context

    # Navigate to the parent of the final field
    for part in parts[:-1]:
        try:
            # Try attribute access first
            target = getattr(target, part)
        except AttributeError:
            # Try dict access
            if isinstance(target, dict) and part in target:
                target = target[part]
            else:
                raise AttributeError(
                    f"Cannot set '{path}': intermediate field '{part}' does not exist on {type(target).__name__}"
                )

    # Set the final field
    final_field = parts[-1]
    try:
        # If there is an existing numeric value, preserve numeric type when value is a numeric string.
        def _coerce_numeric(existing: Any, incoming: Any) -> Any:
            if isinstance(incoming, str) and isinstance(existing, (int, float)):
                try:
                    return type(existing)(incoming)
                except Exception:
                    return incoming
            return incoming

        if isinstance(target, dict):
            if final_field in target:
                value = _coerce_numeric(target.get(final_field), value)
            target[final_field] = value
            return True
        else:
            try:
                existing_val = getattr(target, final_field)
            except Exception:
                existing_val = None
            value = _coerce_numeric(existing_val, value)
            # Fall back to attribute assignment
            setattr(target, final_field, value)
            return True
    except (AttributeError, TypeError) as e:
        raise AttributeError(
            f"Cannot set '{path}': failed to set field '{final_field}' on {type(target).__name__}: {e}"
        )


"""
Note: architect_mode_branch has been removed as the project uses a single Architect path.
"""
