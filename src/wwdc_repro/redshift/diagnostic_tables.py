"""
This file contains two distinct tables:
1) The L=24 Taylor-truncation table: compare the full equation’s test R² 
with its first-order latent approximation.
2) The prediction-envelope coverage tables: 
count predictions inside the zero-latent baseline ± one or two unit-score amplitudes A(z), matching the anatomy figure.
"""

from pathlib import Path
from math import erf, sqrt
import json

import numpy as np
import pandas as pd
import sympy as sp

from wwdc_repro.shared.equations import evaluate, first_order_latent
from wwdc_repro.redshift.frontier import SEEDS

def write_table(path: Path, columns: str, caption: str, heading: str, rows: list[str]) -> None:
    """Write a LaTeX deluxetable from preformatted text.
 
    Parameters
    ----------
    path : Path
        Destination .tex file. Its parent directory must already exist.
    columns : str
        LaTeX column alignment specification, such as "lccc".
    caption : str
        LaTeX caption text, optionally including a label.
    heading : str
        LaTeX column headings to place inside the tablehead command.
    rows : list[str]
        Preformatted LaTeX rows, including column separators and row endings.
    
    Returns
    -------
    None
    
    Notes
    -----
    Writes the supplied text without escaping LaTeX characters or formatting
    numeric values. An existing destination file is overwritten.
    """
    lines = [rf'\begin{{deluxetable}}{{{columns}}}', rf'\tablecaption{{{caption}}}',
             rf'\tablehead{{{heading}}}', r'\startdata', *rows, r'\enddata', r'\end{deluxetable}']
    path.write_text('\n'.join(lines)+'\n')

def first_order_expression(equation: str, n_features: int) -> str:
    """Expand in all latent inputs about zero, retaining exact redshift dependence. Wrapper for first_order_latent(...)"""
    expression, _ = first_order_latent(equation, n_features)
    return str(expression)

def truncation_table(
    x: np.ndarray, y: np.ndarray, selected: pd.DataFrame,
    outdir: Path, selected_l24,
) -> None:
    """Write the L24 full-versus-first-order test-score table.

    Parameters
    ----------
    x : np.ndarray
        Test inputs: training-standardised latents followed by raw redshift.
    y : np.ndarray
        Test target values aligned with x.
    selected : pd.DataFrame
        Validation-selected equations.
    outdir : Path
        Existing directory for truncation.csv, truncation_table.tex and
        truncation.json (original and first-order expressions).
    selected_l24 : callable
        Function (selected, seed) returning the selected L24 equation.
    """
    variance = float(np.var(y))
    truncation, records = [], []
    for seed in SEEDS:
        equation = selected_l24(selected, seed)
        first_equation = first_order_expression(equation, x.shape[1])
        full, first = evaluate(equation, x), evaluate(first_equation, x)
        if not np.isfinite(full).all() or not np.isfinite(first).all():
            raise ValueError(f"Seed {seed}: nonfinite Taylor predictions")
        if seed == 42:
            np.testing.assert_allclose(full, first, rtol=1e-12, atol=1e-12)
        r2_full = float(1 - np.mean((y-full)**2)/variance)
        r2_first = float(1 - np.mean((y-first)**2)/variance)
        truncation.append({"seed": seed, "rows": len(y), "r2_full": r2_full,
                           "r2_first": r2_first, "delta_r2": r2_full-r2_first,
                           "rms_gap": float(np.sqrt(np.mean((full-first)**2)))})
        records.append({"seed": seed, "equation": equation,
                        "first_order_equation": first_equation})
    pd.DataFrame(truncation).to_csv(outdir/'truncation.csv', index=False)
    trunc_rows = []
    for row in truncation:
        first = '--' if row['seed'] == 42 else f"${row['r2_first']:.4f}$"
        delta = '--' if row['seed'] == 42 else f"${row['delta_r2']:.5f}$"
        trunc_rows.append(f"{row['seed']} & ${row['r2_full']:.4f}$ & {first} & {delta}" + r" \\")
    write_table(outdir/'truncation_table.tex', 'lccc',
                r'Full and first-order latent Taylor test scores for the selected $L=24$ equations. Seed 42 is already linear.\label{tab:truncation}',
                r'\colhead{Seed} & \colhead{Full $R^2$} & \colhead{First-order $R^2$} & \colhead{$\Delta R^2$}', trunc_rows)
    report = {"source_script": "taylor_truncation.py", "records": records}
    (outdir/'truncation.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


def coverage_table(
    x: np.ndarray, selected: pd.DataFrame, outdir: Path, selected_l24,
) -> None:
    """Write prediction coverage within the anatomy figure's unit-score bands.

    Parameters
    ----------
    x : np.ndarray
        Test inputs: training-standardised latents followed by raw redshift.
    selected : pd.DataFrame
        Validation-selected equations.
    outdir : Path
        Existing directory for coverage.csv, coverage_table.tex and
        coverage.json (equations, per-row baselines and amplitudes).
    selected_l24 : callable
        Function (selected, seed) returning the selected L24 equation.
    """
    coverage, records = [], []
    for seed in SEEDS:
        equation = selected_l24(selected, seed)
        values, curves = coverage_values(equation, x)
        coverage.append({"seed": seed, "rows": len(x), **values})
        records.append({"seed": seed, "equation": equation, "coverage_curves": curves})
    pd.DataFrame(coverage).to_csv(outdir/'coverage.csv', index=False)
    cov_rows = [f"{r['seed']} & ${100*r['pred_within_1A']:.1f}\\%$ & ${100*r['pred_within_2A']:.1f}\\%$" + r" \\" for r in coverage]
    cov_rows += [r'\hline', f"$\\mathcal{{N}}(0,1)$ & ${100*erf(1/sqrt(2)):.1f}\\%$ & ${100*erf(2/sqrt(2)):.1f}\\%$" + r" \\"]
    write_table(outdir/'coverage_table.tex', 'lcc',
                r'Fraction of test spectra whose full predicted masses lie within $a+bF(z) \pm A(z)$ and $\pm 2A(z)$. $A(z)$ is the unit-score amplitude used in the anatomy figure, evaluated at each spectrum redshift. Gaussian fractions are shown for comparison; $A(z)$ is not estimated from the prediction standard deviation.\label{tab:coverage}',
                r'\colhead{Seed} & \colhead{$\pm A$} & \colhead{$\pm 2A$}', cov_rows)
    report = {
        "coverage_method": "full predictions; exact per-row zero-latent baseline; norm of exact first-order latent coefficients matching anatomy unit-score A(z); inclusive bounds; no empirical SD",
        "records": records,
    }
    (outdir/'coverage.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


def coverage_values(equation: str, x: np.ndarray) -> tuple[dict, dict]:
    """Count full predictions within baseline ± A(z) and ± 2A(z).

    Use the anatomy figure's unit-score amplitude: the Euclidean norm of
    the first-order latent coefficient vector at each galaxy's redshift.
    The baseline is the full equation at zero latents. 

    Parameters
    ----------
    equation: str
        Equation to evaluate
    
    x: np.ndarray
        Test inputs of shape (n_samples, n_features), with redshift in the
        final column.
    
    Returns
    -------
    result : dict
      Counts and fractions of predictions within each band, stored as
      count_1A, count_2A, pred_within_1A and pred_within_2A.
    curves : dict
        Per-row redshifts, zero-latent baseline predictions and amplitudes,
        stored as z, baseline and amplitude.

    Raises
    ------
    ValueError
        If an amplitude is nonfinite or nonpositive, or a full prediction
        or baseline value is nonfinite.
    AssertionError
        If the latent coefficient vectors do not share a fixed direction
        across the supplied redshifts.
    """
    z = x[:, -1] # extract redshift
    first, symbols = first_order_latent(equation, x.shape[1]) # get the first-order expression if required
    coefficients = np.column_stack([
        np.broadcast_to(np.asarray(sp.lambdify(symbols[-1], first.coeff(symbol),
                                              "numpy")(z), dtype=float), z.shape)
        for symbol in symbols[:-1]
    ]) # get the coefficients
    width = np.linalg.norm(coefficients, axis=1) # get the norm of the latent coefficients
    zero = x.copy()
    zero[:, :-1] = 0
    baseline = evaluate(equation, zero) # get the zero-latent equation
    full = evaluate(equation, x) # evaluate the full prediction
    if not np.isfinite(width).all() or np.any(width <= 0):
        raise ValueError("Coverage needs a finite positive unit-score amplitude")
    if not np.isfinite(full).all() or not np.isfinite(baseline).all():
        raise ValueError("Nonfinite coverage predictions")
    # Check a fixed unit direction describes the coefficient vector, as in anatomy.
    direction = coefficients[len(x)//2] / width[len(x)//2]
    np.testing.assert_allclose(coefficients, width[:, None]*direction, rtol=1e-9, atol=1e-11) # check that direction stays fixed.
    delta = np.abs(full - baseline)
    counts = {f"count_{k}A": int((delta <= k*width).sum()) for k in (1, 2)}
    result = {**counts, **{f"pred_within_{k}A": counts[f"count_{k}A"]/len(x) for k in (1, 2)}} # count spectra within +/- k.A k={1,2}
    return result, {"z": z.tolist(), "baseline": baseline.tolist(), "amplitude": width.tolist()}
