"""Re-score the stored Pareto fronts on the unused validation split.

``train_logm.py:156-158`` selects ``best_len`` by minimising ``mse_test``, over ~25
candidates per front, and ``utils.py:26-27`` shows ``fit_sym_fn`` only ever builds
train/test matrices -- the ``val`` split that ``train_logm.py`` loads is never touched.
So every reported R^2 is both scored *and* selected on the reporting set.

This script fixes that without re-running the search. ``pareto_fronts.csv`` stores the
printed expression for every front member, so the fronts can be re-scored by parsing
those strings (sympy) and evaluating them on a val matrix rebuilt exactly as
``fit_sym_fn`` would have.

**The validation gate.** Before any val number is trusted, each equation is also re-
scored on *test* and compared against the recorded ``Test_R2``. If the rebuilt feature
matrix or the parser were wrong, the val numbers would be wrong in exactly the same way
and nothing downstream would reveal it. The run fails loudly if agreement is worse than
``--tolerance``.

Preprocessing details that must match ``utils.fit_sym_fn`` (getting any of these wrong
changes the numbers silently):

- rows: ``dropna(subset=[feature])`` then ``feature != -9999``; upstream,
  ``train_logm.py:96-97`` applies ``mask_ratio <= 0.5`` and ``z_x <= 0.3``.
- the ``StandardScaler`` is fit on the **full** post-filter train split, before the 10k
  subsample is drawn -- so it cannot be reconstructed from the subsample.
- in the ``cond+z`` arm redshift is appended **raw, after scaling**, so the last column
  is physical z rather than a standardised feature.
- the ``z``-only arm is never scaled at all.
- pyoperon prints variables 1-indexed: ``X1`` is column 0.

Usage:
    python val_rescore.py                            # every arm, every seed
    python val_rescore.py --embed-type cond+z --seed 42
    python val_rescore.py --galspec /path/to/galSpecExtra-dr8.fits
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import sympy as sp
from astroML.datasets import fetch_sdss_specgals
from astropy.table import Table
from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

EMBED_TYPES = ["z", "cond", "cond+z", "uncond", "orig"]
SEEDS = [42, 43, 44]
SUBSAMPLE = 10000  # utils.fit_sym_fn's N

# Numeric literals in a printed pyoperon expression. Requires a decimal point, so it
# never matches the digits in a variable name like X11.
CONST_RE = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?")


def to_sympy(equation: str, n_vars: int) -> sp.Expr:
    """Parse a printed pyoperon expression into a sympy expression over X1..Xn."""
    text = equation.replace("^", "**")
    local = {f"X{i}": sp.Symbol(f"X{i}") for i in range(1, n_vars + 1)}
    local.update({"exp": sp.exp, "log": sp.log, "abs": sp.Abs, "Abs": sp.Abs})
    return sp.sympify(text, locals=local)


def evaluate_equation(equation: str, X: np.ndarray) -> np.ndarray:
    """Evaluate a printed expression on a design matrix, 1-indexed columns."""
    n_vars = X.shape[1]
    expr = to_sympy(equation, n_vars)
    symbols = [sp.Symbol(f"X{i}") for i in range(1, n_vars + 1)]
    fn = sp.lambdify(symbols, expr, "numpy")
    with np.errstate(all="ignore"):
        out = fn(*[X[:, i] for i in range(n_vars)])
    return np.asarray(out, dtype=float) * np.ones(len(X))


def parameterise(equation: str) -> tuple[str, list[float]]:
    """Replace every numeric literal with a free parameter c0..ck.

    Per *occurrence*, not per distinct value -- two literals that happen to be equal are
    still two independent parameters, which is what the fitted model actually had.
    """
    constants: list[float] = []

    def swap(match: re.Match) -> str:
        constants.append(float(match.group()))
        return f"c{len(constants) - 1}"

    return CONST_RE.sub(swap, equation), constants


def refit_equation(
    equation: str,
    X: np.ndarray,
    y: np.ndarray,
    check: tuple[np.ndarray, ...] = (),
    max_nfev: int = 2000,
) -> tuple[str, str] | None:
    """Re-fit an expression's constants by least squares, keeping its functional form.

    Repairs entries whose printed constants are unusable -- pyoperon formats in fixed
    point, so a degenerate pair like ``c1 * c2`` with c1~1e21 and c2~1e-22 prints its
    second factor as ``0.000000`` and the expression evaluates to garbage. Only the
    product was ever meaningful, so refitting recovers an equivalent well-conditioned
    parameterisation rather than inventing a new model.

    An unconstrained fit can wander somewhere the expression is undefined *outside* the
    training subsample -- widening ``c`` in ``log(c*X + 1)`` until the argument goes
    negative for extreme rows, say. Matrices passed in ``check`` are tested for
    finiteness, and if the unbounded solution fails them the fit is retried in
    progressively tighter boxes around the printed constants, finally falling back to
    the printed constants themselves. Degenerate pairs still get repaired, since they
    only reach the bounded branches if the unbounded one produced something undefined.

    Returns ``(expression_string, how)`` where ``how`` is "unbounded" or "bounded", or
    None if no usable fit was found.
    """
    template, p0 = parameterise(equation)
    if not p0:
        return None

    n_vars = X.shape[1]
    var_symbols = [sp.Symbol(f"X{i}") for i in range(1, n_vars + 1)]
    par_symbols = [sp.Symbol(f"c{i}") for i in range(len(p0))]
    local = {s.name: s for s in var_symbols + par_symbols}
    local.update({"exp": sp.exp, "log": sp.log, "abs": sp.Abs, "Abs": sp.Abs})

    expr = sp.sympify(template.replace("^", "**"), locals=local)
    fn = sp.lambdify(var_symbols + par_symbols, expr, "numpy")
    columns = [X[:, i] for i in range(n_vars)]

    def residual(p: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            pred = np.asarray(fn(*columns, *p), dtype=float) * np.ones(len(X))
        # least_squares cannot handle non-finite residuals; steer it away instead.
        return np.where(np.isfinite(pred), pred - y, 1e6)

    def finite_everywhere(p: np.ndarray) -> bool:
        for matrix in check:
            cols = [matrix[:, i] for i in range(matrix.shape[1])]
            with np.errstate(all="ignore"):
                pred = np.asarray(fn(*cols, *p), dtype=float) * np.ones(len(matrix))
            if not np.all(np.isfinite(pred)):
                return False
        return True

    # A printed 0.0 is the degenerate half of a pair, and starting exactly there gives a
    # zero step in that direction; nudge it off zero. x_scale="jac" handles the ~1e21
    # spread between partners.
    start = np.array([p if p != 0.0 else 1e-8 for p in p0], dtype=float)

    def solve(bounds) -> np.ndarray | None:
        try:
            fit = least_squares(
                residual,
                start,
                method="trf",
                x_scale="jac",
                max_nfev=max_nfev,
                bounds=bounds,
            )
        except (ValueError, TypeError, ZeroDivisionError, OverflowError):
            return None
        return fit.x

    def render(params: np.ndarray, how: str) -> tuple[str, str]:
        fitted = expr.subs(dict(zip(par_symbols, params, strict=True)))
        return str(sp.N(fitted, 8)), how

    # Unbounded first (the only branch that can repair a degenerate 1e21/1e-22 pair),
    # then progressively tighter boxes around the printed constants. A wide box is not
    # enough on its own: doubling the c in log(c*X + 1) is already enough to send the
    # argument negative on the tails of a standardised column.
    params = solve((-np.inf, np.inf))
    if params is not None and finite_everywhere(params):
        return render(params, "unbounded")

    for scale in (1.0, 0.1, 0.01):
        lo = np.array(
            [min(p * (1 - scale), p * (1 + scale)) if p else -scale for p in start]
        )
        hi = np.array(
            [max(p * (1 - scale), p * (1 + scale)) if p else scale for p in start]
        )
        params = solve((lo, hi))
        if params is not None and finite_everywhere(params):
            return render(params, f"bounded±{scale:g}")

    # Last resort: the printed constants themselves. For an equation that already
    # reproduced these are known finite, so every such entry still gets a row -- just
    # flagged as unchanged rather than silently dropped.
    if finite_everywhere(start):
        return render(start, "unchanged")
    return None


def load_embeddings(root: str) -> dict[str, pd.DataFrame]:
    """Embedding parquet per split.

    ``train_logm.py`` reads these in ``natsorted`` order, but nothing here depends
    on row order: the catalogue joins are on ``specObjID``, the scaler's statistics
    are order-invariant, and X/y are built from the same frame so they stay aligned.
    Plain ``sorted`` reproduces the same numbers without pulling in ``natsort``.
    """
    frames = {}
    for split in ("train", "val", "test"):
        files = sorted(glob.glob(f"{root}/{split}/*.parquet"))
        if not files:
            message = f"no parquet for split {split!r} under {root!r}"
            raise SystemExit(message)
        frames[split] = pd.concat([pd.read_parquet(f) for f in files]).reset_index(
            drop=True
        )
    return frames


def build_merged(
    root: str, galspec_path: str, specgals_home: str
) -> dict[str, pd.DataFrame]:
    """Replicate train_logm.py's merge chain: embeddings + astroML + galSpecExtra."""
    embeddings = load_embeddings(root)

    df_sdss = pd.DataFrame(fetch_sdss_specgals(data_home=specgals_home))
    df_sdss["merge_id"] = df_sdss["specObjID"].astype("int64")

    galspec = Table.read(galspec_path, format="fits")
    galspec_df = galspec.to_pandas().dropna(subset=["SPECOBJID"]).copy()
    galspec_df["specObjID"] = (
        galspec_df["SPECOBJID"].astype(str).str.extract(r"(\d+)")[0].astype("int64")
    )

    merged = {}
    for split, frame in embeddings.items():
        df_spender = frame.copy()
        df_spender["merge_id"] = (
            df_spender["id"].astype(str).str.extract(r"(\d+)")[0].astype("int64")
        )
        matched = pd.merge(df_spender, df_sdss, on="merge_id", how="inner").drop(
            columns=["merge_id"]
        )
        if "mask_ratio" in matched.columns:
            matched = matched[(matched["mask_ratio"] <= 0.5) & (matched["z_x"] <= 0.3)]
        matched = matched.reset_index(drop=True)
        merged[split] = pd.merge(
            matched, galspec_df, on=["specObjID"], how="inner", suffixes=["_", ""]
        )
        print(f"  {split}: {len(merged[split])} rows after merge")
    return merged


def build_matrices(
    merged_dfs: dict[str, pd.DataFrame], feature: str, embed_type: str
) -> dict[str, np.ndarray]:
    """Rebuild fit_sym_fn's design matrices for train/val/test.

    The scaler is fit on the full post-filter train split, matching fit_sym_fn,
    which scales before drawing its 10k subsample.
    """
    is_flux = "flux" in feature.lower()

    clean = {}
    for split, df in merged_dfs.items():
        sub = df.dropna(subset=[feature])
        sub = sub[sub[feature] != -9999.0]
        if is_flux:
            sub = sub[sub[feature] > 0]
        clean[split] = sub

    out: dict[str, np.ndarray] = {}

    if embed_type == "z":
        # never scaled
        for split, sub in clean.items():
            out[f"X_{split}"] = sub["z_x"].values.reshape(-1, 1)
    else:
        base = "cond" if embed_type == "cond+z" else embed_type
        raw = {split: np.stack(sub[base].values) for split, sub in clean.items()}
        for split, X in raw.items():
            if X.ndim == 3:
                raw[split] = X.reshape(X.shape[0], -1)

        scaler = StandardScaler()
        out["X_train"] = scaler.fit_transform(raw["train"])
        for split in ("val", "test"):
            out[f"X_{split}"] = scaler.transform(raw[split])

        if embed_type == "cond+z":
            # raw redshift appended AFTER scaling -- last column is physical z
            for split, sub in clean.items():
                z = sub["z_x"].values.reshape(-1, 1)
                out[f"X_{split}"] = np.hstack((out[f"X_{split}"], z))

    for split, sub in clean.items():
        y = sub[feature].values
        if is_flux:
            y = np.log10(y)
        out[f"y_{split}"] = y.reshape(-1, 1).flatten()

    return out


def rescore_front(
    front: pd.DataFrame,
    mats: dict[str, np.ndarray],
    fit_subsample: tuple[np.ndarray, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Re-score every front member on val and test, optionally refitting its constants.

    ``fit_subsample`` is the (X, y) the refit trains on -- pass None to skip refitting.
    """
    rows = []
    for row in front.itertuples():
        record = {
            "Length": row.Length,
            "Test_R2_stored": row.Test_R2,
            "Test_MSE_stored": row.Test_MSE,
            "Equation": row.Equation,
        }
        for split in ("test", "val"):
            X, y = mats[f"X_{split}"], mats[f"y_{split}"]
            try:
                pred = evaluate_equation(row.Equation, X)
                if not np.all(np.isfinite(pred)):
                    message = "non-finite predictions"
                    raise ValueError(message)
                record[f"{split.capitalize()}_R2"] = r2_score(y, pred)
                record[f"{split.capitalize()}_MSE"] = mean_squared_error(y, pred)
            except (
                TypeError,
                ValueError,
                KeyError,
                ZeroDivisionError,
                sp.SympifyError,
            ) as exc:
                print(f"    [warn] length {row.Length} on {split}: {exc}")
                record[f"{split.capitalize()}_R2"] = np.nan
                record[f"{split.capitalize()}_MSE"] = np.nan

        record["Refit_Equation"] = None
        record["Refit_Test_R2"] = np.nan
        record["Refit_Val_R2"] = np.nan
        record["Refit_How"] = None
        if fit_subsample is not None:
            refit = refit_equation(
                row.Equation,
                *fit_subsample,
                check=(mats["X_test"], mats["X_val"]),
            )
            if refit is not None:
                refit_eq, how = refit
                record["Refit_Equation"] = refit_eq
                record["Refit_How"] = how
                for split in ("test", "val"):
                    X, y = mats[f"X_{split}"], mats[f"y_{split}"]
                    try:
                        pred = evaluate_equation(refit_eq, X)
                        if np.all(np.isfinite(pred)):
                            record[f"Refit_{split.capitalize()}_R2"] = r2_score(y, pred)
                    except (
                        TypeError,
                        ValueError,
                        KeyError,
                        ZeroDivisionError,
                        sp.SympifyError,
                    ) as exc:
                        print(f"    [warn] refit length {row.Length} on {split}: {exc}")

        rows.append(record)
    return pd.DataFrame(rows)


# Legacy test-gated CLI intentionally omitted; use validate_conditioning.py.
