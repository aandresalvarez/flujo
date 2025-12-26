from __future__ import annotations

import ast
from collections.abc import Callable


class _SafeEvaluator(ast.NodeVisitor):
    """Evaluate a restricted Python expression AST safely.

    Allowed constructs:
    - Literals: Constant (str, int, float, bool, None)
    - Names: previous_step, output, context, steps, resume_input
    - Attribute access (e.g., obj.attr)
    - Subscript access with string key (e.g., obj['key'])
    - Bool operations: and/or
    - Unary not
    - Comparisons: ==, !=, <, <=, >, >=, in, not in
    """

    def __init__(self, names: dict[str, object]) -> None:
        self.names = names

    def visit_Expression(self, node: ast.Expression) -> object:  # pragma: no cover - delegated
        return self.visit(node.body)

    def visit_Name(self, node: ast.Name) -> object:
        if node.id in {"true", "True"}:
            return True
        if node.id in {"false", "False"}:
            return False
        if node.id not in {"previous_step", "output", "context", "steps", "resume_input"}:
            raise ValueError(f"Unknown name: {node.id}")
        return self.names.get(node.id)

    def visit_Constant(self, node: ast.Constant) -> object:
        return node.value

    def visit_Tuple(self, node: ast.Tuple) -> object:
        return tuple(self.visit(elt) for elt in node.elts)

    def visit_List(self, node: ast.List) -> object:
        return [self.visit(elt) for elt in node.elts]

    def visit_Attribute(self, node: ast.Attribute) -> object:
        obj = self.visit(node.value)
        try:
            if isinstance(obj, dict):
                return obj.get(node.attr)
            return getattr(obj, node.attr)
        except Exception:
            return None

    def visit_Subscript(self, node: ast.Subscript) -> object:
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

    def visit_Call(self, node: ast.Call) -> object:
        """Allow-list a tiny set of safe, read-only method calls.

        Supported:
        - dict.get(key[, default])
        - str.lower()
        - str.upper()
        - str.strip()       (no args)
        - str.startswith(x) (single string arg)
        - str.endswith(x)   (single string arg)
        """
        # Only allow method calls (obj.method(...)); never bare function calls
        if not isinstance(node.func, ast.Attribute):
            raise ValueError("Unsupported expression element: Call")

        # Disallow keyword arguments entirely for simplicity/safety
        if node.keywords:
            raise ValueError("Unsupported expression element: Call")

        obj = self.visit(node.func.value)
        method = node.func.attr

        if obj is None:
            return None

        # dict.get(key[, default])
        if isinstance(obj, dict) and method == "get":
            if len(node.args) not in (1, 2):
                raise ValueError("Unsupported expression element: Call")
            key = self.visit(node.args[0])
            if not isinstance(key, str):
                # Restrict to string keys to stay consistent with other access rules
                raise ValueError("Unsupported expression element: Call")
            default = self.visit(node.args[1]) if len(node.args) == 2 else None
            try:
                return obj.get(key, default)
            except Exception:
                return None

        # String methods
        if isinstance(obj, str):
            if method == "lower" and len(node.args) == 0:
                return obj.lower()
            if method == "upper" and len(node.args) == 0:
                return obj.upper()
            if method == "strip" and len(node.args) == 0:
                return obj.strip()
            if method in ("startswith", "endswith") and len(node.args) == 1:
                arg = self.visit(node.args[0])
                if not isinstance(arg, str):
                    raise ValueError("Unsupported expression element: Call")
                return getattr(obj, method)(arg)

        # Everything else is disallowed
        raise ValueError("Unsupported expression element: Call")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> object:
        if isinstance(node.op, ast.Not):
            return not bool(self.visit(node.operand))
        raise ValueError("Unsupported unary operator")

    def visit_BoolOp(self, node: ast.BoolOp) -> object:
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

    def visit_Compare(self, node: ast.Compare) -> object:
        left = self.visit(node.left)
        result = True
        for op, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            try:

                def _coerce_number(val: object) -> object:
                    if isinstance(val, str):
                        try:
                            return int(val)
                        except Exception:
                            try:
                                return float(val)
                            except Exception:
                                return val
                    return val

                left_cmp = _coerce_number(left)
                right_cmp = _coerce_number(right)
                if isinstance(op, ast.In):
                    if right_cmp is None:
                        ok = False
                    elif isinstance(right_cmp, dict):
                        ok = left_cmp in right_cmp
                    elif isinstance(right_cmp, (list, tuple, set, str)):
                        ok = left_cmp in right_cmp
                    else:
                        ok = False
                elif isinstance(op, ast.NotIn):
                    if right_cmp is None:
                        ok = True
                    elif isinstance(right_cmp, dict):
                        ok = left_cmp not in right_cmp
                    elif isinstance(right_cmp, (list, tuple, set, str)):
                        ok = left_cmp not in right_cmp
                    else:
                        ok = True
                elif isinstance(op, ast.Eq):
                    ok = left_cmp == right_cmp
                elif isinstance(op, ast.NotEq):
                    ok = left_cmp != right_cmp
                elif isinstance(op, ast.Lt):
                    if isinstance(left_cmp, (int, float)) and isinstance(right_cmp, (int, float)):
                        ok = left_cmp < right_cmp
                    elif isinstance(left_cmp, str) and isinstance(right_cmp, str):
                        ok = left_cmp < right_cmp
                    else:
                        ok = False
                elif isinstance(op, ast.LtE):
                    if isinstance(left_cmp, (int, float)) and isinstance(right_cmp, (int, float)):
                        ok = left_cmp <= right_cmp
                    elif isinstance(left_cmp, str) and isinstance(right_cmp, str):
                        ok = left_cmp <= right_cmp
                    else:
                        ok = False
                elif isinstance(op, ast.Gt):
                    if isinstance(left_cmp, (int, float)) and isinstance(right_cmp, (int, float)):
                        ok = left_cmp > right_cmp
                    elif isinstance(left_cmp, str) and isinstance(right_cmp, str):
                        ok = left_cmp > right_cmp
                    else:
                        ok = False
                elif isinstance(op, ast.GtE):
                    if isinstance(left_cmp, (int, float)) and isinstance(right_cmp, (int, float)):
                        ok = left_cmp >= right_cmp
                    elif isinstance(left_cmp, str) and isinstance(right_cmp, str):
                        ok = left_cmp >= right_cmp
                    else:
                        ok = False
                else:
                    raise ValueError("Unsupported comparison operator")
            except Exception:
                ok = False
            if not ok:
                result = False
                break
            left = right
        return result

    # Disallow everything else
    def generic_visit(self, node: ast.AST) -> object:  # pragma: no cover - safety
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")


ExpressionCallable = Callable[[object, object | None], object]


def compile_expression_to_callable(expression: str) -> ExpressionCallable:
    """Compile a restricted expression string into a callable.

    The returned function accepts (output, context) and returns the expression value
    (truthiness used by loops; raw value used by conditionals).
    """

    expr = expression.strip()
    parsed = ast.parse(expr, mode="eval")

    def _call(output: object, context: object | None) -> object:
        # Late import to avoid heavy deps if unused
        ctx_proxy: object
        steps_map: dict[str, object] = {}
        try:
            from .template_vars import TemplateContextProxy, get_steps_map_from_context

            steps_map = dict(get_steps_map_from_context(context))
            ctx_proxy = TemplateContextProxy(context, steps=steps_map)
        except Exception:  # pragma: no cover - defensive fallback
            ctx_proxy = context
        names = {
            "previous_step": output,
            "output": output,
            "context": ctx_proxy,
            "steps": steps_map,
        }

        # Add resume_input if HITL history exists
        try:
            if context is not None and hasattr(context, "hitl_history"):
                hitl_history = getattr(context, "hitl_history", None)
                if isinstance(hitl_history, list) and hitl_history:
                    last = hitl_history[-1]
                    names["resume_input"] = getattr(last, "human_response", None)
        except Exception:
            pass  # resume_input will be undefined if no HITL history

        evaluator = _SafeEvaluator(names)
        return evaluator.visit(parsed)

    return _call


__all__ = ["compile_expression_to_callable"]
