from __future__ import annotations


def format_cost(value: float | int) -> str:
    """
    Format a cost value for display, following production-grade standards.

    This function handles decimal formatting in a way that:
    - Removes unnecessary trailing zeros (e.g., 10.0 -> "10", 10.5 -> "10.5")
    - Preserves meaningful decimal places (e.g., 0.15 -> "0.15", 1.2 -> "1.2")
    - Handles edge cases like exact integers and zero values
    - Is consistent across all numeric inputs
    - Follows standard financial formatting practices

    Args:
        value: The numeric value to format (int, float, or other types)

    Returns:
        A string representation of the cost value with appropriate decimal formatting
    """
    if not isinstance(value, (int, float)):
        return str(value)

    # Handle zero values
    if value == 0:
        return "0"

    # Convert to float to handle both int and float inputs consistently
    float_value = float(value)

    # Check if the value is exactly representable as an integer
    if float_value.is_integer():
        return str(int(float_value))

    # For non-integer values, use a clean format
    # Use a high precision format and then clean up trailing zeros
    formatted = f"{float_value:.10f}"

    # Remove trailing zeros and decimal point if not needed
    cleaned = formatted.rstrip("0").rstrip(".")

    # Handle the case where we end up with an empty string (shouldn't happen, but safety first)
    if not cleaned:
        return "0"

    return cleaned
