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
    """Evaluate a symbolic equation for every row of a feature matrix.

    Accept numbers, X1..Xn, arithmetic, powers, and exp/log/abs/Abs.
    Both ** and ^ denote exponentiation. Nonfinite predictions are returned
    for the scoring layer to classify.

    Parameters
    ----------
    expression : str
        Nonempty expression from the symbolic regression search. Variable Xi
        refers to column i - 1 of x. The expression may use a subset of columns.
    x : np.ndarray
        Finite, nonempty real numeric matrix of shape (n_samples, n_features).
        All columns must be finite, including unused features.

    Returns
    -------
    prediction : np.ndarray
        Predictions of shape (n_samples,), one per input row.
        Results may contain NaN or infinity for invalid numerical operations.

    Raises
    ------
    EvaluationInputError
        If x is not a finite, nonempty, two-dimensional real numeric matrix.
    ExpressionError
        If expression is not a nonempty string, has invalid syntax, or contains
        an unsupported function, variable, or expression structure. Also raised
        if evaluation produces a complex result or an unexpected output shape.
    """
    x = np.asarray(x)
    if x.ndim != 2 or not all(x.shape) or not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x):
        raise EvaluationInputError("Features must be a nonempty real numeric matrix")
    if not np.isfinite(x).all():
        raise EvaluationInputError("Features must be finite")
    symbols = [sp.Symbol(f"X{i}") for i in range(1, x.shape[1] + 1)]
    tree = to_sympy(expression, x.shape[1])
    function = sp.lambdify(symbols, tree, "numpy")
    with np.errstate(all="ignore"):
        prediction = function(*(x[:, i] for i in range(x.shape[1])))
    result = np.asarray(prediction)
    if np.iscomplexobj(result) or result.shape not in ((), (len(x),)):
        raise ExpressionError("Expected a real scalar or one prediction per row")
    return np.asarray(result, dtype=float) * np.ones(len(x))

def to_sympy(expression: str, n_vars: int) -> sp.Expr:
    """Parse an exported Operon equation using the evaluator's allowed grammar.

    Parameters
    ----------
    expression : str
        Equation using X1..Xn, numbers, arithmetic, powers, and exp/log/abs/Abs.
    n_vars : int
        Number of available feature columns; Xi denotes column i - 1.

    Returns
    -------
    sympy.Expr
        Symbolic expression with the exported constants preserved.

    Raises
    ------
    ExpressionError
        If the expression has invalid syntax or unsupported operations or variables.
    """
    symbols = [sp.Symbol(f"X{i}") for i in range(1, n_vars + 1)]
    names = {symbol.name: symbol for symbol in symbols}
    names.update({"exp": sp.exp, "log": sp.log, "abs": sp.Abs, "Abs": sp.Abs})
    if not isinstance(expression, str) or not expression.strip():
        raise ExpressionError("Expected a nonempty expression string")
    source = expression.replace("^", "**")
    try:
        syntax = ast.parse(source, mode="eval")  # Parse one expression; validate its allowed operations below.
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
    return sp.sympify(source, locals=names)


def first_order_latent(equation: str, n_features: int) -> tuple[sp.Expr, tuple[sp.Symbol, ...]]:
    """Expand jointly to first order about zero latents, keeping redshift exact.

    Parameters
    ----------
    equation : str
        Saved symbolic equation with inputs X1 through Xn.
    n_features : int
        Number of inputs. The final input is redshift; all others are latents.

    Returns
    -------
    expression : sp.Expr
        Constant and linear latent terms, with decimal coefficients parsed
        exactly as rationals. Redshift dependence is not expanded.
    symbols : tuple[sp.Symbol, ...]
        Real-valued input symbols in column order, for coefficient extraction
        or numerical evaluation of the returned expression.
    """
    symbols = sp.symbols(f"X1:{n_features + 1}", real=True)
    names = {symbol.name: symbol for symbol in symbols}
    names.update({"exp": sp.exp, "log": sp.log, "abs": sp.Abs, "Abs": sp.Abs})
    tree = sp.sympify(equation.replace("^", "**"), locals=names, rational=True)
    # One scale parameter makes every quadratic/mixed latent term O(epsilon**2).
    epsilon = sp.Symbol("epsilon", real=True)
    scaled = tree.subs({s: epsilon * s for s in symbols[:-1]}, simultaneous=True)
    first_order = sp.series(scaled, epsilon, 0, 2).removeO().subs(epsilon, 1).expand()
    return first_order, symbols


def has_fixed_latent_direction(expr):
    """Test proportional symbolic gradients, including nonlinear single-score forms.

    Constant ratios imply dependence on one fixed linear combination on each
    connected differentiable domain; this does not imply an additive correction.
    """
    # Operon inputs are real. Without real assumptions, differentiating Abs
    # introduces independent re/im derivatives and falsely breaks proportionality.
    expr=expr.xreplace({v:sp.Symbol(v.name,real=True) for v in expr.free_symbols})
    expr=expr.xreplace({v:sp.Rational(str(v)) for v in expr.atoms(sp.Float)})
    grads=[sp.diff(expr,sp.Symbol(f'X{i}',real=True)) for i in range(1,11)]
    active=[g for g in grads if g!=0]
    if not active:return False
    return all(not sp.cancel(g/active[0]).free_symbols for g in active)
