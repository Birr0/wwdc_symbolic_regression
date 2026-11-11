"""Audited numerical recipes for the zero-latent, anatomy, and CMD figures.

Plot styling is deliberately separate from the saved numerical products. The
zero-latent panel preserves the historical six-significant-digit reductions.
"""

from pathlib import Path
import json

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
import sympy as sp
import seaborn as sns


from wwdc_repro.shared.cmd import join_photometry, bin_scores, masters10_red_edge, masters10_ridge
from wwdc_repro.shared.data import matrices
from wwdc_repro.shared.equations import evaluate, first_order_latent
from wwdc_repro.shared.figstyle import INK, INK_MUTED, STYLE, SURFACE, tidy
from wwdc_repro.redshift.frontier import SEEDS, front_path
from wwdc_repro.redshift.weights import raw_coefficients


def _anatomy_legend(ax):
    """
    Name the curves in the equation anatomy plot
    
    Parameters
    ----------
    ax: matplotlib.axes.Axes

    Returns
    -------
    None

    """
    handles = [
        Line2D([], [], color=INK, linewidth=1.8, label="$s=0$"),
        Line2D(
            [], [], color=INK, linewidth=1.0, linestyle=(0, (5.5, 2.4)),
            label="$s=\\pm1$",
        ),
        Line2D(
            [], [], color=INK_MUTED, linewidth=0.9, linestyle=(0, (1.4, 1.8)),
            label="$s=\\pm2$",
        ),
    ]
    leg = ax.legend(
        handles=handles, loc="lower right", frameon=True, fontsize=11,
        handlelength=2.2, borderpad=0.5, labelspacing=0.4,
    )
    frame = leg.get_frame()
    frame.set_edgecolor(INK_MUTED)
    frame.set_linewidth(0.6)
    frame.set_facecolor(SURFACE)
    frame.set_alpha(1.0)


@plt.style.context(STYLE)
def anatomy(frames, selected, outdir: Path, equation_for_seed):
    """Plot length-24 equation curves at s=0, +/-1, and +/-2 for each seed.

    Use a first-order Taylor expansion about zero latent coordinates, retaining
    the full redshift dependence. This is exact for the saved seed-42 equation
    and an approximation for seeds 43/44. Normalize the latent coefficient
    vector at the grid midpoint to define s, and orient it so the mean amplitude
    over the grid is positive. Points show catalogue masses coloured by s.
    The curves are fixed-score predictions at a given redshift.

    Parameters
    ----------
    frames : dict[str, pd.DataFrame]
        Cleaned train, val, and test splits used by matrices to standardize the
        conditional latents using training statistics and append raw redshift.
    selected : pd.DataFrame
        Validation-selected equations for each arm, seed, and length budget.
    outdir : Path
        Existing directory for equation_anatomy_by_seed PNG/PDF plots, CSV curve
        values, and JSON diagnostics.
    equation_for_seed : Callable[[pd.DataFrame, int], str]
        Function returning an equation string from selected and a seed. The
        caller supplies selected_l24, which retrieves the cond+z selection at
        budget 24 and verifies that its actual length is 24.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the baseline or amplitude contains nonfinite values.
    AssertionError
        If A(z) times the fixed unit weights does not reconstruct the first-order
        latent coefficients across the redshift grid within numerical tolerance.
        This check does not measure the Taylor truncation error.
    """
    # Prepare standardized test features and zero-latent inputs on a redshift grid.
    data = matrices(frames, "cond+z")
    x, y = data["X_test"], data["y_test"]
    z = x[:, -1]
    grid = np.linspace(z.min(), z.max(), 300)
    zero = np.zeros((len(grid), x.shape[1]))
    zero[:, -1] = grid
    fig, axes = plt.subplots(
        1, 3, figsize=(13.2, 4.3), dpi=600, sharex=True, sharey=True, facecolor=SURFACE
    )
    records = []
    diagnostics = []
    for ax, seed in zip(axes, SEEDS, strict=True):
        equation = equation_for_seed(selected, seed)
        baseline = evaluate(equation, zero)
        # Expand to first order in the latents while keeping redshift unexpanded.
        first_order, symbols = first_order_latent(equation, x.shape[1])
        # Each latent's linear coefficient is a function of redshift.
        latent_coefficients = np.column_stack([
            np.asarray(sp.lambdify(symbols[-1], first_order.coeff(symbol),
                                  "numpy")(grid), dtype=float) * np.ones(len(grid))
            for symbol in symbols[:-1]
        ])
        # Take the latent coefficients at the middle redshift as a reference.
        unit_direction = latent_coefficients[len(grid) // 2].copy()
        # Normalize to unit length, preserving the relative weights that define s.
        unit_direction /= np.linalg.norm(unit_direction)
        # Find the multiplier A(z) that recovers the coefficients at each redshift.
        amplitude = latent_coefficients @ unit_direction
        # Orient s so larger values increase predicted mass on average over redshift.
        # Flipping both s's weights and A(z) leaves their product unchanged.
        if amplitude.mean() < 0:
            unit_direction *= -1
            amplitude *= -1
        # Verify that A(z) times the fixed weights reconstructs every latent coefficient.
        np.testing.assert_allclose(latent_coefficients, amplitude[:, None] * unit_direction,
                                   rtol=1e-9, atol=1e-11)
        if not np.isfinite(amplitude).all() or not np.isfinite(baseline).all():
            raise ValueError(f"Nonfinite canonical curve for seed {seed}") 
        # Compute each galaxy's s from its standardized latent coordinates.
        score = x[:, :-1] @ unit_direction
        diagnostics.append({"seed": seed, "equation": equation,
                            "unit_direction": unit_direction.tolist(),
                            "score_sd": float(score.std()),
                            "max_gradient_residual": float(np.max(np.abs(
                                latent_coefficients - amplitude[:, None] * unit_direction))),
                            "interpretation": "f(z,0) + s * A(z); exact latent gradient at zero"})
        sc = ax.scatter(
            z, y, c=score, s=3.0, alpha=0.40, linewidths=0, cmap="RdBu_r",
            vmin=-2.2, vmax=2.2, rasterized=True, zorder=2, 
        ) # Plot catalogue stellar mass against redshift, coloured by s.

        ax.fill_between(
            grid, baseline - amplitude, baseline + amplitude, color=INK, alpha=0.06,
            linewidth=0, zorder=3,
        ) # Shade the region between s=-1 and s=+1.
        halo = [pe.withStroke(linewidth=3.2, foreground=SURFACE)]
        ax.plot(grid, baseline, color=INK, linewidth=1.8, zorder=4, path_effects=halo) # Draw the s=0 baseline curve.
        for k, colour, width, dash, stroke in ( # Styles for curves at s=+/-k, with k=1 or 2.
            (1, INK, 1.0, (0, (5.5, 2.4)), 2.2),
            (2, INK_MUTED, 0.9, (0, (1.4, 1.8)), 2.0),
        ):
            for edge in (baseline + k * amplitude, baseline - k * amplitude):
                ax.plot(
                    grid, edge, color=colour, linewidth=width, linestyle=dash, zorder=4,
                    path_effects=[pe.withStroke(linewidth=stroke, foreground=SURFACE)],
                )
        ax.set_title(f"seed {seed}", fontsize=15)
        ax.set_xlabel("$z$", fontsize=20)
        ax.set_xlim(0.02, 0.275)
        ax.set_ylim(9.0, 12.0)
        tidy(ax, grid_axis=None, labelsize=12)
        records.extend(
            {
                "seed": seed,
                "z": float(zi),
                "baseline": float(mu),
                "A": float(a),
                **{f"s_{s:+d}": float(mu + s * a) for s in (-2, -1, 0, 1, 2)},
            }
            for zi, mu, a in zip(grid, baseline, amplitude, strict=True)
        )
    _anatomy_legend(axes[0])
    axes[0].set_ylabel(r"$\log_{10}(M_\star/M_\odot)$ [dex]", fontsize=14)
    colorbar = fig.colorbar(sc, ax=axes, fraction=0.020, pad=0.012)
    colorbar.set_label("$s$", fontsize=13)
    colorbar.ax.tick_params(labelsize=12)
    colorbar.solids.set_alpha(1.0)
    # Save plots, curve values, and diagnostics.
    stem = outdir / "equation_anatomy_by_seed"
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), dpi=600, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(records).to_csv(outdir / "equation_anatomy_by_seed.csv", index=False)
    (outdir / "equation_anatomy_by_seed.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )


def zero_latent(frames, fronts: Path, outdir: Path, save):
    """Plot Figure 2: zero-latent equation curves and deviations from observed means.

    Set all ten latent coordinates to zero in saved cond+z and orig+z equations
    across seeds and lengths. Simplify the reductions to six significant digits
    and evaluate them over redshift. The top panel overlays these curves on the
    test-sample mass distribution and its binned mean log mass. The lower panel
    shows predictions minus the interpolated observed mean. Pointwise median
    equation curves are also currently shown for each arm.

    Parameters
    ----------
    frames : dict[str, pd.DataFrame]
        Cleaned data splits. Uses the test split's z_x and lgm_tot_p50 columns
        for the observed distribution and mean log mass in bins with >=20 rows.
    fronts : Path
        Root directory containing saved Pareto-front CSV files for each arm and
        seed. Uses archived equations, not just validation-selected equations.
    outdir : Path
        Existing directory for zero_latent.png/.pdf, zero_latent.csv (median
        curves, observed means, and deviations), and zero_latent_exclusions.json.
    save : Callable
        Function accepting a Matplotlib Figure and an output Path stem. Saves
        PNG/PDF files and closes the figure; supplied by figures.py.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If an arm has no usable zero-latent reductions, or no finite observed
        bin means are available for interpolation.
    FileNotFoundError
        If a required Pareto-front CSV is missing.

    Notes
    -----
    Nonfinite archived test scores, singular/constant reductions, and handled
    parsing or numerical failures exclude individual equations. These failures
    are recorded in the exclusions JSON rather than terminating the whole run.
    """
    frame = frames["test"] # get the test split of embeddings and data
    z, mass = frame.z_x.to_numpy(), frame.lgm_tot_p50.to_numpy() # convert redshift and mass to numpy
    edges = np.linspace(0.02, 0.3, 31)
    centres = (edges[1:] + edges[:-1]) / 2
    bins = np.digitize(z, edges) - 1 # bin the redshift values
    means = np.array(
        [
            mass[bins == i].mean() if (bins == i).sum() >= 20 else np.nan
            for i in range(30)
        ]
    ) # get the mean stellar mass in each bin
    grid = np.linspace(0.02, 0.3, 200) # grid of redshifts.
    good = np.isfinite(means) # check if means are finite
    observed = np.interp(grid, centres[good], means[good]) # interpolate between mean mass in bins on the redshift grid.
    redshift = sp.Symbol("z", positive=True) # define redshift symbol
    names = {f"X{i}": sp.Symbol(f"X{i}") for i in range(1, 11)} # define each latent variable
    names.update(
        {"X11": redshift, "exp": sp.exp, "log": sp.log, "abs": sp.Abs, "Abs": sp.Abs}
    ) # define operations
    zeros = {sp.Symbol(f"X{i}"): 0 for i in range(1, 11)} # Sets the latents variables to zero.
    bundles, all_records, exclusions = {}, [], []
    for arm in ("cond+z", "orig+z"):
        bundle = []
        for seed in SEEDS:
            front = pd.read_csv(front_path(fronts, arm, seed)).drop_duplicates(
                ["Seed", "Length"]
            ) # read the equation for given length and seed.
            for row in front.itertuples():
                try:
                    if not np.isfinite(row.Test_R2):
                        raise ValueError("historical nonfinite parent score")
                    expression = sp.sympify(
                        row.Equation.replace("^", "**"), locals=names
                    )
                    reduced = sp.N(sp.simplify(expression.subs(zeros)), 6) # sub the zeros for latents, simplify the expression 
                    #and then evaluate the coefficients to six significant figures.
                    if (
                        reduced.has(sp.zoo, sp.oo, sp.nan, sp.I)
                        or redshift not in reduced.free_symbols
                    ):
                        raise ValueError("singular or constant reduction") # raise value error if reduction has not worked.
                    with np.errstate(all="ignore"):
                        values = np.asarray(
                            sp.lambdify(redshift, reduced, "numpy")(grid), dtype=float
                        ) # compute the curve for the redshift grid.
                    if not np.isfinite(values).all():
                        raise ValueError("nonfinite reduction")
                except (
                    TypeError,
                    ValueError,
                    KeyError,
                    ZeroDivisionError,
                    sp.SympifyError,
                ) as exc:
                    exclusions.append(
                        {
                            "arm": arm,
                            "seed": seed,
                            "length": row.Length,
                            "reason": str(exc),
                        }
                    )
                    continue
                bundle.append((row.Length, values)) # add equation length and predicted values to the bundle
        if not bundle:
            raise ValueError(f"No finite zero-latent reductions for {arm}")
        bundles[arm] = bundle # save bundle to correct arm
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 6.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.8]},
        layout="constrained",
    )
    _, _, _, density = axes[0].hist2d(
        z, mass, bins=[120, 120], cmap="Greys", cmin=1, alpha=0.55
    ) # plot the density of mass values across redshift.
    fig.colorbar(
        density, ax=axes[0], location="top", label="Spectra per bin",
        fraction=0.06, pad=0.04,
    ) # add colour bar for the density of mass values.
    all_lengths = [length for bundle in bundles.values() for length, _ in bundle]
    norm = Normalize(min(all_lengths), max(all_lengths)) # provide a colour scale for the equation lengths
    medians = []
    ramps = {
        "cond+z": ["#7babe6", "#276fc5", "#17406f"], # blue shades for cond+z
        "orig+z": ["#c3718f", "#901944", "#550e27"], # red shades for orig+z
    }
    for arm in ("cond+z", "orig+z"):
        cmap = LinearSegmentedColormap.from_list(arm, ramps[arm]) # create a continuous colour gradient for each arm
        for length, values in bundles[arm]:
            colour = cmap(norm(length))
            axes[0].plot(grid, values, color=colour, lw=0.8, alpha=0.8) # plot the zero latent prediction
            axes[1].plot(grid, values - observed, color=colour, lw=0.8, alpha=0.8) # plot the residual of the zero latent prediction from the observed mean.
        # TODO: Remove the thick median curves; retain individual equations and the
        # observed mean. Also revise the median-dependent y-limits and CSV output.
        median = np.median([values for _, values in bundles[arm]], axis=0)
        medians.append(median)
        for ax, values in zip(axes, (median, median - observed), strict=True):
            ax.plot(grid, values, color=cmap(0.8), lw=2.4, label=f"{arm} median")
        fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            ax=axes,
            label=f"{arm}: length",
            fraction=0.025,
            pad=0.02,
        )
        all_records.extend(
            {
                "arm": arm,
                "z": zi,
                "median_curve": mu,
                "empirical_mean": obs,
                "deviation": mu - obs,
            }
            for zi, mu, obs in zip(grid, median, observed, strict=True)
        )
    axes[0].plot(centres[good], means[good], "k--", lw=2, label="observed binned mean")
    lo, hi = (
        min(v.min() for v in [*medians, observed]),
        max(v.max() for v in [*medians, observed]),
    )
    axes[0].set(
        ylim=(lo - 0.12 * (hi - lo), hi + 0.12 * (hi - lo)), ylabel=r"$\log_{10}(M_\star/M_\odot)$ [dex]"
    )
    axes[0].legend(fontsize=8)
    axes[1].axhline(0, color="black", ls="--")
    axes[1].set(
        xlim=(0.02, 0.3), xlabel="redshift", ylabel="prediction − observed mean [dex]"
    )
    save(fig, outdir / "zero_latent")
    pd.DataFrame(all_records).to_csv(outdir / "zero_latent.csv", index=False)
    (outdir / "zero_latent_exclusions.json").write_text(
        json.dumps(exclusions, indent=2) + "\n"
    )


@plt.style.context(STYLE)
def cmd(frames, selected, cache: Path, outdir: Path, equation_for_seed):
    """
    Create the colour-magnitude diagram in Figure 4.

    Combine all splits and match spectra to cached NYU-VAGC photometry.
    Plot the galaxy density and mean latent score in the cmd plane.
    Extract the discovered latent sum directly from the seed-42 equation and
    normalize its coefficients to unit length, oriented towards increasing mass.

    Parameters
    ----------
    frames: dict[pd.DataFrame]
        Data splits for embeddings 

    selected: pd.DataFrame
        Validation-selected equations for each arm, seed, and length budget.

    cache: Path 
        Photometry parquet file containing specObjID, ABSMAG_g, and
        ABSMAG_r. Matching requires unique spectrum IDs in both tables.

    outdir: Path
        Existing directory for the PNG/PDF figure, matched-row CSV,
        numerical grid NPZ, and summary JSON.

    equation_for_seed: Callable[[pd.DataFrame, int], str]
        Function returning an equation string from selected and a seed.
        The caller supplies selected_l24; this plot uses seed 42.

    Returns
    -------
    None 
    
    """
    data = matrices(frames, "cond+z") # make the numpy matrices cond+z
    equation = equation_for_seed(selected, 42) 
    # extract the equation for seed 42 using the equation_for_seed function.
    coefficients = raw_coefficients(equation) # extract coefficients from the equation.
    direction = coefficients / np.linalg.norm(coefficients) # normalise coefficients to unit length.
    # The nested latent sum can enter the full equation with a negative factor.
    # Orient the score by the predicted mass response across the plotted z range.
    reference = np.zeros((100, 11))
    reference[:, -1] = np.linspace(0.02, 0.3, 100)
    shifted = reference.copy()
    shifted[:, :10] = 1e-5 * direction
    response = evaluate(equation, shifted) - evaluate(equation, reference)
    if np.mean(response) < 0:
        direction *= -1
        response *= -1
    if not (np.isfinite(response).all() and (response > 0).all()):
        raise ValueError("CMD score must increase predicted mass across redshift")
    pieces = []
    for split in ("train", "val", "test"):
        x = data[f"X_{split}"]
        pieces.append(
            pd.DataFrame(
                {
                    "specObjID": frames[split].specObjID.to_numpy(dtype="int64"),
                    "s": x[:, :10] @ direction,
                    "z": x[:, -1],
                    "split": split,
                }
            )
        )
    cat = pd.concat(pieces, ignore_index=True) # create full dataframe
    # Convert catalogue h=1 magnitudes to H0=70 with the shared Astropy recipe.
    # TODO: Describe this magnitude convention in the manuscript.
    cat = join_photometry(cat, cache).rename(columns={"gr": "g_r", "Mr": "Mr70"})
    xedges, yedges = np.linspace(-24, -18, 61), np.linspace(0.15, 1.20, 57)
    counts, mean = bin_scores(cat.Mr70, cat.g_r, cat.s, xedges, yedges)
    sample = cat.sample(60000, random_state=0) if len(cat) > 60000 else cat
    # Seaborn levels are low-density tail fractions: 0.1 encloses 90%, etc.
    probability_levels = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    fig, axes = plt.subplots(
        1, 2, figsize=(11.0, 4.6), sharex=True, sharey=True,
        gridspec_kw={"wspace": 0.36},  # 0.30 historically; Matplotlib 3.11 needs more room
    )
    sns.kdeplot(
        data=sample, x="Mr70", y="g_r", ax=axes[0],
        fill=True, levels=probability_levels, cmap="Greys",
        gridsize=180, bw_method="scott", bw_adjust=1, cut=3,
        clip=((-24, -18), (0.15, 1.2)),
    )
    # Use the plotted contour thresholds to label enclosed probabilities.
    filled = axes[0].collections[-1]
    filled.set_edgecolor(INK_MUTED)
    filled.set_linewidth(0.7)
    cb0 = fig.colorbar(filled, ax=axes[0], fraction=0.046, pad=0.02,
                      ticks=filled.levels[:-1])
    cb0.ax.set_yticklabels([f"{1-p:.0%}" for p in probability_levels[:-1]])
    #cb0.set_label("enclosed KDE probability within grid", fontsize=12)
    limit = float(np.nanpercentile(np.abs(mean), 98))
    image = axes[1].pcolormesh(
        xedges, yedges, mean.T, cmap="RdBu_r", vmin=-limit, vmax=limit,
        shading="auto", rasterized=True,
    )
    cb1 = fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.02)
    cb1.set_label("$\\langle s \\rangle$ (dimensionless)", fontsize=16)
    grid = np.linspace(-24, -18, 200)
    for ax in axes:
        ax.plot(
            grid, masters10_red_edge(grid), color=INK, linewidth=1.6,
            linestyle=(0, (5.5, 2.4)), zorder=5,
            label="Blue edge of the red sequence (Masters et al. 2010)",
        )
        ax.plot(
            grid, masters10_ridge(grid), color=INK_MUTED, linewidth=1.3,
            linestyle=(0, (1.2, 1.6)), zorder=5,
            label="Red-sequence fit (Masters et al. 2010)",
        )
        ax.set_xlim(-18, -24)
        ax.set_ylim(0.15, 1.0)
        ax.set_xlabel("$M_r$", fontsize=16)
        tidy(ax, grid_axis=None, labelsize=12)
        ax.tick_params(labelleft=True)
        ax.set_ylabel("$g - r$", fontsize=16)
    leg = axes[0].legend(frameon=True, fontsize=9.5, loc="lower left", borderpad=0.5)
    leg.get_frame().set_edgecolor(INK_MUTED)
    leg.get_frame().set_linewidth(0.6)
    leg.get_frame().set_facecolor(SURFACE)
    for cb in (cb0, cb1):
        cb.outline.set_edgecolor(INK_MUTED)
        cb.outline.set_linewidth(0.6)
        cb.ax.tick_params(labelsize=12)
    stem = outdir / "cmd_latent_all_gr_seed42"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    cat.to_csv(outdir / "cmd_rows.csv.gz", index=False)
    np.savez_compressed(
        outdir / "cmd_grid.npz",
        counts=counts,
        mean_score=mean,
        xedges=xedges,
        yedges=yedges,
    )
    red = cat.g_r > masters10_red_edge(cat.Mr70)
    summary = {
        "rows": len(cat),
        "split_counts": cat.split.value_counts().to_dict(),
        "score_mean": float(cat.s.mean()),
        "score_sd": float(cat.s.std()),
        "red_blue_separation_sd": float(
            abs(cat.s[red].mean() - cat.s[~red].mean()) / cat.s.std()
        ),
        "kde": {
            "implementation": f"seaborn {sns.__version__} kdeplot",
            "sample_rows": len(sample), "random_state": 0,
            "gridsize": 180, "bw_method": "scott", "bw_adjust": 1, "cut": 3,
            "clip": [[-24, -18], [0.15, 1.2]],
            "iso_proportions": probability_levels,
            "density_thresholds": filled.levels.tolist(),
            "normalization": "Probability normalized over Seaborn's evaluation grid",
        },
        "direction": direction.tolist(),
        "cache_policy": "frozen historical first match per spectrum; no new quality cuts",
    }
    (outdir / "cmd_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
