"""
Export the historical conditional+z fronts for the manuscript appendix.

Rows retain the historical finite archived Test_R2 filter; these are not
validation-selected equations. SymPy parses and canonicalises expressions,
then constants are rounded for display. Scores refer to stored fitted models,
not to the rounded expressions. No search or score evaluation is performed.
"""

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import sympy as sp

from wwdc_repro.redshift.frontier import SEEDS, front_path

Z = sp.Symbol("z")

def to_sympy(equation):
    local = {f"X{i}": sp.Symbol(f"X{i}") for i in range(1, 12)}
    local["X11"] = Z
    local.update({"exp": sp.exp, "log": sp.log, "abs": sp.Abs, "Abs": sp.Abs})
    return sp.sympify(equation.replace("^", "**"), locals=local)


def lost_to_formatting(equation) -> bool:
    """Flag possible coefficient loss in the archived equation string.

    The archived Equation strings use six decimal places, so a coefficient far below
    1e-6 is stored as exactly ``0.000000``. If the search pairs a
    small coefficient with a very large one -- e.g., seed 43's L=5 is
    ``9.700627 + 691408658867591053312.0 * (0.000000 * X11)`` -- the original product may have been nonzero, 
    and simplifying drops the term entirely.

    We want to flag using a heuristic for the equations coefficients rather than inferring 
    a replacement coefficient. 

    Parameters
    ----------
    equation: str
        Archived equation string to inspect for possible precision loss.

    Returns
    -------
    bool
        True if the string contains both a decimal literal with absolute
        value greater than 1e12 and a zero-valued decimal literal.
        This is provides a heuristic warning.

    """
    lits = [float(m) for m in re.findall(r"-?\d+\.\d+e?[+-]?\d*", equation)]
    return any(abs(v) > 1e12 for v in lits) and any(v == 0.0 for v in lits)


def round_floats(expr, sig):
    """Round floating-point constants in a SymPy expression for display.
    
    Parameters
    ----------
    expr: sp.Expr
        Symbolic expression whose floating-point constants will be rounded.

    sig: int 
        Number of signficant figures to retain.

    Returns
    -------
    sp.Expr
        Expression with rounded floating-point constants. Integer constants
        are unchanged. Substitution may trigger symbolic simplification.
    
    Notes
    -----
    This changes the displayed expression only; reported test scores are
    not recalculated using the rounded coefficients.
    """
    mapping = {}
    for node in expr.atoms(sp.Float): # loop over the distinct floating point constants in expr
        value = float(node)
        if value == 0:
            mapping[node] = sp.Float(0)
            continue
        digits = sig - int(np.floor(np.log10(abs(value)))) - 1 # calculate number of signficant figures.
        # For example, suppose value = 0.069944 and sig = 4:
        #1. abs(value) removes the sign.
        #2. log10(0.069944) is approximately −1.155.
        #3. floor(...) gives −2, identifying the position of the first significant digit.
        #4. Therefore, digits = 4 - (-2) - 1 = 5.
        mapping[node] = sp.Float(round(value, digits))
    return expr.xreplace(mapping) # replace the coefficients


def front_tables(fronts: Path, outdir: Path, sigfigs: int = 4) -> None:
    """Write front_condz.tex.

    Parameters
    ----------
    fronts : Path
        Root containing the conditional+z Pareto-front CSVs for the SEEDS used in the symbolic search.
    outdir : Path
        Existing output directory.
    sigfigs : int
        Positive number of significant figures for displayed constants.

    Raises
    ------
    ValueError
        If precision is invalid, a seed has no finite scores, or a CSV
        contains another seed. Parsing failures propagate rather than
        silently omitting equations from the appendix.
    """
    if sigfigs < 1:
        raise ValueError("sigfigs must be positive")
    lines, records, exclusions = [], [], []
    for seed in SEEDS:
        path = front_path(fronts, "cond+z", seed)
        frame = pd.read_csv(path)
        if not frame.Seed.eq(seed).all(): # check this is the corresponding seed
            raise ValueError(f"Unexpected seed in {path}")
        if "Equation_p60" in frame:
            frame["Equation"] = frame["Equation_p60"]
        finite = np.isfinite(frame.Test_R2) # check R2 is finite
        exclusions.extend(
            {"seed": seed, "length": int(row.Length), "reason": "nonfinite archived Test_R2"}
            for row in frame.loc[~finite].itertuples()
        ) # record any equations excluded
        frame = frame.loc[finite].sort_values("Length", kind="stable") # keep finite r2 equations and sort by length
        if frame.empty:
            raise ValueError(f"No finite archived test scores for seed {seed}")
        lines.extend([
            rf"\subsubsection*{{Seed {seed}}}",
            r"\begin{longtable}{@{}r@{\ \ \ }l@{\ \ \ }l@{}}",
            r"\toprule", r"$L$ & Test $R^2$ & expression \\",
            r"\midrule", r"\endhead",
        ])
        flagged = False
        for row in frame.itertuples():
            expr = round_floats(to_sympy(row.Equation), sigfigs)
            # Use the manuscript's notation for latents and natural logarithms.
            body = sp.latex(expr, symbol_names={sp.Symbol(f"X{i}"): f"x_{{{i}}}" for i in range(1, 11)}, ln_notation=True)
            lost = bool(lost_to_formatting(row.Equation))
            flagged |= lost
            mark = r"^{\ddagger}" if lost else "" # add dagger if equation has lost coefficients
            lines.append(
                rf"${int(row.Length)}{mark}$ & ${row.Test_R2:.3f}$ & "
                rf"\adjustbox{{max width=0.80\linewidth}}{{$\displaystyle {body}$}} \\"
            )
            records.append({"seed": seed, "length": int(row.Length),
                            "test_r2": float(row.Test_R2), "equation": row.Equation,
                            "latex": body, "stored_precision_warning": lost})
        lines.extend([r"\bottomrule", r"\end{longtable}", r"\addtocounter{table}{-1}"])
        if flagged:
            lines.append(r"$\ddagger$: Stored coefficient precision loses a term; the reported $R^2$ refers to the original fitted expression.")
        if seed != SEEDS[-1]:
            lines.append(r"\newpage")
    (outdir / "front_condz.tex").write_text("\n".join(lines) + "\n")
    report = {"population": "archived cond+z fronts with finite archived Test_R2",
              "sigfigs": sigfigs, "rows": records, "exclusions": exclusions}
    (outdir / "front_condz.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
