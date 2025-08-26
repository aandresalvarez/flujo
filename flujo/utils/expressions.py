from __future__ import annotations

import ast
from typing import Any, Optional, Callable


class _SafeEvaluator(ast.NodeVisitor):
    """Evaluate a restricted Python expression AST safely.

    Allowed constructs:
    - Literals: Constant (str, int, float, bool, None)
    - Names: previous_step, output, context, steps
    - Attribute access (e.g., obj.attr)
    - Subscript access with string key (e.g., obj['key'])
    - Bool operations: and/or
    - Unary not
    - Comparisons: ==, !=, <, <=, >, >=, in, not in
    """

    def __init__(self, names: dict[str, Any]) -> None:
        self.names = names

    def visit_Expression(self, node: ast.Expression) -> Any:  # pragma: no cover - delegated
        return self.visit(node.body)

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id not in {"previous_step", "output", "context", "steps"}:
            raise ValueError(f"Unknown name: {node.id}")
        return self.names.get(node.id)

    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        obj = self.visit(node.value)
        try:
            if isinstance(obj, dict):
                return obj.get(node.attr)
            return getattr(obj, node.attr)
        except Exception:
            return None

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        # Only allow string keys
        key_node = node.slice
        if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
            key = key_node.value
        else:
            raise ValueError("Only string literal keys are allowed in subscripts")
        obj = self.visit(node.value)
        try:
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)
        except Exception:
            return None

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        if isinstance(node.op, ast.Not):
            return not bool(self.visit(node.operand))
        raise ValueError("Unsupported unary operator")

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        if isinstance(node.op, ast.And):
            result = True
            for v in node.values:
                result = bool(self.visit(v)) and result
                if not result:
                    break
            return result
        if isinstance(node.op, ast.Or):
            for v in node.values:
                if bool(self.visit(v)):
                    return True
            return False
        raise ValueError("Unsupported boolean operator")

    def visit_Compare(self, node: ast.Compare) -> Any:
        left = self.visit(node.left)
        result = True
        for op, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            if isinstance(op, ast.In):
                ok = left in right if right is not None else False
            elif isinstance(op, ast.NotIn):
                ok = left not in right if right is not None else True
            elif isinstance(op, ast.Eq):
                ok = left == right
            elif isinstance(op, ast.NotEq):
                ok = left != right
            elif isinstance(op, ast.Lt):
                ok = left < right
            elif isinstance(op, ast.LtE):
                ok = left <= right
            elif isinstance(op, ast.Gt):
                ok = left > right
            elif isinstance(op, ast.GtE):
                ok = left >= right
            else:
                raise ValueError("Unsupported comparison operator")
            if not ok:
                result = False
                break
            left = right
        return result

    # Disallow everything else
    def generic_visit(self, node: ast.AST) -> Any:  # pragma: no cover - safety
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def compile_expression_to_callable(expression: str) -> Callable[[Any, Optional[Any]], Any]:
    """Compile a restricted expression string into a callable.

    The returned function accepts (output, context) and returns the expression value
    (truthiness used by loops; raw value used by conditionals).
    """

    expr = expression.strip()
    parsed = ast.parse(expr, mode="eval")

    def _call(output: Any, context: Optional[Any]) -> Any:
        # Late import to avoid heavy deps if unused
        ctx_proxy: Any
        try:
            from .template_vars import TemplateContextProxy, get_steps_map_from_context

            steps_map = get_steps_map_from_context(context)
            ctx_proxy = TemplateContextProxy(context, steps=steps_map)
        except Exception:  # pragma: no cover - defensive fallback
            steps_map = {}
            ctx_proxy = context
        names = {
            "previous_step": output,
            "output": output,
            "context": ctx_proxy,
            "steps": steps_map,
        }
        evaluator = _SafeEvaluator(names)
        return evaluator.visit(parsed)

    return _call


__all__ = ["compile_expression_to_callable"]
