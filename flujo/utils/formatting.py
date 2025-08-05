def format_cost(value) -> str:
    """Format a cost value for display, removing unnecessary trailing zeros."""
    if isinstance(value, (int, float)):
        return f"{value:g}"
    return str(value)