"""Evaluate saved pyoperon expressions without fitting or changing constants."""

from __future__ import annotations

import ast
import numpy as np
import sympy as sp


class ExpressionError(ValueError):
    """An expression is outside the supported exported-equation grammar."""


class EvaluationInputError(ValueError):
    """Feature or target arrays do not meet the evaluation contract."""


def evaluate(expression: str, x: np.ndarray) -> np.ndarray:
    """Evaluate a real scalar expression, preserving historical SymPy simplification.

    Allow only numbers, X1..Xn, arithmetic, powers, and exp/log/abs/Abs.
    Nonfinite predictions are returned for the scoring layer to classify.
    """
    x = np.asarray(x)
    if x.ndim != 2 or not all(x.shape) or not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x):
        raise EvaluationInputError("Features must be a nonempty real numeric matrix")
    if not np.isfinite(x).all():
        raise EvaluationInputError("Features must be finite")
    symbols = [sp.Symbol(f"X{i}") for i in range(1, x.shape[1] + 1)]
    names = {symbol.name: symbol for symbol in symbols}
    names.update({"exp": sp.exp, "log": sp.log, "abs": sp.Abs, "Abs": sp.Abs})
    if not isinstance(expression, str) or not expression.strip():
        raise ExpressionError("Expected a nonempty expression string")
    source = expression.replace("^", "**")
    try:
        syntax = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError("Invalid expression syntax") from exc

    def validate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return
        if isinstance(node, ast.Name) and node.id in {s.name for s in symbols}:
            return
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            validate(node.left)
            validate(node.right)
            return
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            validate(node.operand)
            return
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in ("exp", "log", "abs", "Abs")
                and len(node.args) == 1 and not node.keywords):
            validate(node.args[0])
            return
        raise ExpressionError("Unsupported function, variable, or expression structure")

    validate(syntax.body)
    tree = sp.sympify(source, locals=names)
    function = sp.lambdify(symbols, tree, "numpy")
    with np.errstate(all="ignore"):
        prediction = function(*(x[:, i] for i in range(x.shape[1])))
    result = np.asarray(prediction)
    if np.iscomplexobj(result) or result.shape not in ((), (len(x),)):
        raise ExpressionError("Expected a real scalar or one prediction per row")
    return np.asarray(result, dtype=float) * np.ones(len(x))
