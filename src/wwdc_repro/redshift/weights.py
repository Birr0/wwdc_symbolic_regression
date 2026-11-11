"""LaTeX table of the latent weights of each seed's L=24 cond+z expression (tab:weights)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sympy as sp

N_LATENT = 10

CAPTION = r"""Latent weights of the $L=24$ expression of each seed in Equation~\ref{eq:seed-forms}:
$w_k$ for seed~42 and $v_k$ for seeds 43 and 44. Each vector is scaled to unit length and
signed so that a larger $s = \sum_k w_k X_k$ means a larger stellar mass; because
$\log\left|3.14\,z\right| < 0$ over the sample, this is the negative of the coefficient as it
appears in the expression. The final row gives the length of the coefficient vector as
found by the search, which restores the coefficients to their original scale. Latents
marked -- do not appear in that seed's expression.\label{tab:weights}"""


def raw_coefficients(equation: str) -> np.ndarray:
    """
    Extract raw coefficients from an equations nested linear latent sum.

    Select the subexpression with the fewest operations that contains the latents.
    This assumes that the intended latent sum appears as a distinct subexpression and is 
    fine for the redshift equations which have been inspected in this work. More general latent
    and latent coefficient extraction may be useful for future work.

    Parameters
    ----------
    equation: str
        The symbolic equation containing latent variables and redshift 
        from which we extract the latent coefficients.
    
    Returns
    -------
    coeff: np.ndarray
        Unnormalised coefficients of the latent variables, in order, with zeros for
        absent latents.
    
    Raises
    ------
    ValueError
      If the selected polynomial does not have total degree one.
    sp.PolynomialError
        If the selected subexpression is not polynomial in the latents.
    TypeError
        If a coefficient cannot be converted to a float.
    """
    latents = sp.symbols(f"X1:{N_LATENT + 1}") # create the symbols for the latent equation
    names = {s.name: s for s in sp.symbols(f"X1:{N_LATENT + 2}")} # name the latent symbols in a dictionary
    names.update({"exp": sp.exp, "log": sp.log, "abs": sp.Abs}) # add the named operations
    tree = sp.sympify(equation.replace("^", "**"), locals=names) # replace power (^) with python **
    used = tree.free_symbols & set(latents) 
    nodes = [n for n in sp.preorder_traversal(tree) if used <= n.free_symbols] # Find the part of the equation containing the latent sum
    inner = min(nodes, key=sp.count_ops) # Select the simplest subexpression containing all used latents
    poly = sp.Poly(sp.expand(inner), *latents) 
    # check if the the extracted expression is linear in the latents
    if poly.total_degree() != 1 or not all(
        sp.Poly(sp.expand(inner), s).degree() <= 1 for s in latents
    ):
        raise ValueError(f"Latent sum is not linear: {inner}") 
    return np.array([float(poly.coeff_monomial(s)) for s in latents])

def weights_table(
    selected, outdir: Path, selected_l24
) -> None:
    '''
    Write the latent-weight table for the selected L=24 cond+z equations.
    
    For seeds 42, 43 and 44, normalise the raw latent coefficients and
    reverse their sign so that increasing the score increases predicted mass.
    This sign convention applies to the inspected L=24 equations over
    0.02 <= z <= 0.3. Save a LaTeX table and supporting values as JSON.

    Parameters
    ----------
    selected: pd.DataFrame
        Dataframe of validation-selected equations for each arm, seed and budget

    outdir: Path 
        path where latent_weights.tex and latent_weights.json are written

    selected_l24: callable
        Function with signature (selected, seed) that returns the selected
        L=24 cond+z equation as a string.

    Returns
    -------
    None 

    Raises
    ------
    ValueError
        If the raw coefficient vector has a nonfinite or zero norm.
    '''
    seeds = (42, 43, 44)
    columns, report = {}, []
    for seed in seeds:
        equation = selected_l24(selected, seed)
        raw = raw_coefficients(equation)
        norm = float(np.linalg.norm(raw))
        if not np.isfinite(norm) or norm == 0:
            raise ValueError(f"Seed {seed}: coefficients need a finite, nonzero norm")
        # For these three equations, mass decreases with the written latent sum
        # over 0.02 <= z <= 0.3; reverse its sign to make the score increase mass.
        direction = -raw / norm
        columns[seed] = (direction, raw != 0, norm)
        report.append({"seed": seed, "equation": equation, "raw_coefficients": raw.tolist(),
                       "raw_norm": norm, "unit_direction": direction.tolist(),
                       "sign_flipped": bool(raw @ direction < 0)})
    sub = lambda k: f"$X_{{{k}}}$" if k > 9 else f"$X_{k}$"
    rows = []
    for k in range(N_LATENT):
        cells = [f"${columns[s][0][k]:+.3f}$" if columns[s][1][k] else "--" for s in seeds]
        rows.append(f"{sub(k + 1):<8} & " + " & ".join(f"{c:<8}" for c in cells).rstrip() + r" \\")
    norms = [f"${columns[s][2]:.3g}$" for s in seeds]
    rows += [r"\hline", "Raw norm & " + " & ".join(f"{c:<8}" for c in norms).rstrip() + r" \\"]
    tex = "\n".join([
        r"% Generated by wwdc-figures --only weights; do not edit by hand.",
        r"\begin{deluxetable}{lccc}",
        rf"\tablecaption{{{CAPTION}}}",
        r"\tablehead{" + " & ".join(
            [r"\colhead{Latent}"] + [rf"\colhead{{Seed {s}}}" for s in seeds]) + "}",
        r"\startdata", *rows, r"\enddata", r"\end{deluxetable}", "",
    ])
    (outdir / "latent_weights.tex").write_text(tex)
    (outdir / "latent_weights.json").write_text(json.dumps(report, indent=2) + "\n")
