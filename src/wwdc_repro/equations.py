"""Evaluate saved pyoperon expressions without fitting or changing constants."""

from __future__ import annotations

import numpy as np
import sympy as sp


def evaluate(expression: str, x: np.ndarray) -> np.ndarray:
    symbols = [sp.Symbol(f"X{i}") for i in range(1, x.shape[1] + 1)]
    names = {symbol.name: symbol for symbol in symbols}
    names.update({"exp": sp.exp, "log": sp.log, "abs": sp.Abs, "Abs": sp.Abs})
    tree = sp.sympify(expression.replace("^", "**"), locals=names)
    function = sp.lambdify(symbols, tree, "numpy")
    with np.errstate(all="ignore"):
        prediction = function(*(x[:, i] for i in range(x.shape[1])))
    return np.asarray(prediction, dtype=float) * np.ones(len(x))
