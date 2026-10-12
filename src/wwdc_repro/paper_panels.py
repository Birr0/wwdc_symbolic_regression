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
from scipy.stats import gaussian_kde

from .data import matrices
from .equations import evaluate
from .figstyle import INK, INK_MUTED, STYLE, SURFACE, tidy
from .frontier import SEEDS, front_path


def _anatomy_legend(ax):
    """Name the curves, as in the historical add_legend; the colour bar names s."""
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
def anatomy(frames, selected, outdir: Path, direction, equation_for_seed, save):
    """Canonical L24 curves at s=0, +/-1, +/-2; first order for seeds 43/44."""
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
        # Exact derivatives at zero give the canonical first-order latent term.
        # They retain ln(1+c*z); no large-c*z approximation is needed here.
        symbols = sp.symbols(f"X1:{x.shape[1] + 1}", real=True)
        names = {symbol.name: symbol for symbol in symbols}
        names.update({"exp": sp.exp, "log": sp.log, "abs": sp.Abs, "Abs": sp.Abs})
        tree = sp.sympify(equation.replace("^", "**"), locals=names)
        at_zero = dict.fromkeys(symbols[:-1], 0)
        slopes = np.column_stack([
            np.asarray(sp.lambdify(symbols[-1], sp.diff(tree, symbol).subs(at_zero),
                                  "numpy")(grid), dtype=float) * np.ones(len(grid))
            for symbol in symbols[:-1]
        ])
        unit_direction = slopes[len(grid) // 2].copy()
        unit_direction /= np.linalg.norm(unit_direction)
        if np.dot(unit_direction, direction(equation, z, x.shape[1])) < 0:
            unit_direction *= -1
        amplitude = slopes @ unit_direction
        # One fixed score must represent the gradient at every plotted redshift.
        np.testing.assert_allclose(slopes, amplitude[:, None] * unit_direction,
                                   rtol=1e-9, atol=1e-11)
        if not np.isfinite(amplitude).all() or not np.isfinite(baseline).all():
            raise ValueError(f"Nonfinite canonical curve for seed {seed}")
        score = x[:, :-1] @ unit_direction
        diagnostics.append({"seed": seed, "equation": equation,
                            "unit_direction": unit_direction.tolist(),
                            "score_sd": float(score.std()),
                            "max_gradient_residual": float(np.max(np.abs(
                                slopes - amplitude[:, None] * unit_direction))),
                            "interpretation": "f(z,0) + s * A(z); exact latent gradient at zero"})
        sc = ax.scatter(
            z, y, c=score, s=3.0, alpha=0.40, linewidths=0, cmap="RdBu_r",
            vmin=-2.2, vmax=2.2, rasterized=True, zorder=2,
        )
        ax.fill_between(
            grid, baseline - amplitude, baseline + amplitude, color=INK, alpha=0.06,
            linewidth=0, zorder=3,
        )
        halo = [pe.withStroke(linewidth=3.2, foreground=SURFACE)]
        ax.plot(grid, baseline, color=INK, linewidth=1.8, zorder=4, path_effects=halo)
        for k, colour, width, dash, stroke in (
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
    stem = outdir / "equation_anatomy_by_seed"
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), dpi=600, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(records).to_csv(outdir / "equation_anatomy_by_seed.csv", index=False)
    (outdir / "equation_anatomy_by_seed.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )


def zero_latent(frames, fronts: Path, outdir: Path, save):
    """All usable historical front reductions, not a validation-selected subset."""
    frame = frames["test"]
    z, mass = frame.z_x.to_numpy(), frame.lgm_tot_p50.to_numpy()
    edges = np.linspace(0.02, 0.3, 31)
    centres = (edges[1:] + edges[:-1]) / 2
    bins = np.digitize(z, edges) - 1
    means = np.array(
        [
            mass[bins == i].mean() if (bins == i).sum() >= 20 else np.nan
            for i in range(30)
        ]
    )
    grid = np.linspace(0.02, 0.3, 200)
    good = np.isfinite(means)
    observed = np.interp(grid, centres[good], means[good])
    redshift = sp.Symbol("z", positive=True)
    names = {f"X{i}": sp.Symbol(f"X{i}") for i in range(1, 11)}
    names.update(
        {"X11": redshift, "exp": sp.exp, "log": sp.log, "abs": sp.Abs, "Abs": sp.Abs}
    )
    zeros = {sp.Symbol(f"X{i}"): 0 for i in range(1, 11)}
    bundles, all_records, exclusions = {}, [], []
    for arm in ("cond+z", "orig+z"):
        bundle = []
        for seed in SEEDS:
            front = pd.read_csv(front_path(fronts, arm, seed)).drop_duplicates(
                ["Seed", "Length"]
            )
            for row in front.itertuples():
                try:
                    if not np.isfinite(row.Test_R2):
                        raise ValueError("historical nonfinite parent score")
                    expression = sp.sympify(
                        row.Equation.replace("^", "**"), locals=names
                    )
                    reduced = sp.N(sp.simplify(expression.subs(zeros)), 6)
                    if (
                        reduced.has(sp.zoo, sp.oo, sp.nan, sp.I)
                        or redshift not in reduced.free_symbols
                    ):
                        raise ValueError("singular or constant reduction")
                    with np.errstate(all="ignore"):
                        values = np.asarray(
                            sp.lambdify(redshift, reduced, "numpy")(grid), dtype=float
                        )
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
                bundle.append((row.Length, values))
        if not bundle:
            raise ValueError(f"No finite zero-latent reductions for {arm}")
        bundles[arm] = bundle
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
    )
    fig.colorbar(
        density, ax=axes[0], location="top", label="Spectra per bin",
        fraction=0.06, pad=0.04,
    )
    all_lengths = [length for bundle in bundles.values() for length, _ in bundle]
    norm = Normalize(min(all_lengths), max(all_lengths))
    medians = []
    ramps = {
        "cond+z": ["#7babe6", "#276fc5", "#17406f"],
        "orig+z": ["#c3718f", "#901944", "#550e27"],
    }
    for arm in ("cond+z", "orig+z"):
        cmap = LinearSegmentedColormap.from_list(arm, ramps[arm])
        for length, values in bundles[arm]:
            colour = cmap(norm(length))
            axes[0].plot(grid, values, color=colour, lw=0.8, alpha=0.8)
            axes[1].plot(grid, values - observed, color=colour, lw=0.8, alpha=0.8)
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
def cmd(frames, selected, cache: Path, outdir: Path, equation_for_seed, save):
    """Historical pooled g-r CMD from the frozen, unique-spectrum cache.

    The cache preserves the manuscript's historical first-match choice. This
    route neither silently rematches sources nor applies a new quality cut.
    """
    data = matrices(frames, "cond+z")
    equation = equation_for_seed(selected, 42)
    base = np.zeros((1, 11))
    base[:, -1] = 0.05
    slopes = []
    for column in range(10):
        plus, minus = base.copy(), base.copy()
        plus[:, column], minus[:, column] = 1e-4, -1e-4
        slopes.append(
            float(((evaluate(equation, plus) - evaluate(equation, minus)) / 2e-4)[0])
        )
    direction = np.asarray(slopes) / np.linalg.norm(slopes)
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
    cat = pd.concat(pieces, ignore_index=True)
    photometry = pd.read_parquet(cache)
    cat = cat.merge(
        photometry[["specObjID", "ABSMAG_g", "ABSMAG_r"]],
        on="specObjID",
        how="inner",
        validate="one_to_one",
    )
    for column in ("ABSMAG_g", "ABSMAG_r"):
        cat = cat[np.isfinite(cat[column]) & (cat[column] > -99) & (cat[column] < 0)]
    cat["g_r"] = cat.ABSMAG_g - cat.ABSMAG_r
    cat["Mr70"] = cat.ABSMAG_r - 0.7745
    xedges, yedges = np.linspace(-24, -18, 61), np.linspace(0.15, 1.20, 57)
    counts = np.histogram2d(cat.Mr70, cat.g_r, bins=[xedges, yedges])[0]
    totals = np.histogram2d(cat.Mr70, cat.g_r, bins=[xedges, yedges], weights=cat.s)[0]
    mean = np.where(counts >= 25, totals / np.maximum(counts, 1), np.nan)
    sample = cat.sample(60000, random_state=0) if len(cat) > 60000 else cat
    kde = gaussian_kde(np.vstack((sample.Mr70, sample.g_r)))
    kx, ky = np.linspace(-24, -18, 180), np.linspace(0.15, 1.2, 180)
    xx, yy = np.meshgrid(kx, ky)
    density = kde(np.vstack((xx.ravel(), yy.ravel()))).reshape(xx.shape)
    ranked = np.sort(density.ravel())[::-1]
    enclosed = np.cumsum(ranked) * (kx[1] - kx[0]) * (ky[1] - ky[0])
    levels = sorted(
        set(
            float(ranked[min(np.searchsorted(enclosed, f), len(ranked) - 1)])
            for f in (0.1, 0.3, 0.5, 0.7, 0.9)
        )
    )
    # Rendering matches the historical plot_cmd_latent.py --colour g-r run.
    fracs = (0.1, 0.3, 0.5, 0.7, 0.9)
    fig, axes = plt.subplots(
        1, 2, figsize=(11.0, 4.6), sharex=True, sharey=True,
        gridspec_kw={"wspace": 0.36},  # 0.30 historically; Matplotlib 3.11 needs more room
    )
    filled = axes[0].contourf(
        kx, ky, density, levels=[*levels, density.max()], cmap="Greys", extend="neither"
    )
    axes[0].contour(
        kx, ky, density, levels=levels, colors=[INK_MUTED], linewidths=0.7, zorder=4
    )
    cb0 = fig.colorbar(filled, ax=axes[0], fraction=0.046, pad=0.02, ticks=levels)
    cb0.ax.set_yticklabels([f"{f:.0%}" for f in fracs[::-1]])
    cb0.set_label("enclosed fraction of galaxies", fontsize=14)
    limit = float(np.nanpercentile(np.abs(mean), 98))
    image = axes[1].pcolormesh(
        xedges, yedges, mean.T, cmap="RdBu_r", vmin=-limit, vmax=limit,
        shading="auto", rasterized=True,
    )
    cb1 = fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.02)
    cb1.set_label("$\\langle s \\rangle$", fontsize=16)
    grid = np.linspace(-24, -18, 200)
    for ax in axes:
        ax.plot(
            grid, 0.63 - 0.02 * (grid + 20), color=INK, linewidth=1.6,
            linestyle=(0, (5.5, 2.4)), zorder=5,
            label="Blue edge of the red sequence (Masters et al. 2010)",
        )
        ax.plot(
            grid, 0.73 - 0.02 * (grid + 20), color=INK_MUTED, linewidth=1.3,
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
        kx=kx,
        ky=ky,
        density=density,
    )
    red = cat.g_r > 0.63 - 0.02 * (cat.Mr70 + 20)
    summary = {
        "rows": len(cat),
        "split_counts": cat.split.value_counts().to_dict(),
        "score_mean": float(cat.s.mean()),
        "score_sd": float(cat.s.std()),
        "red_blue_separation_sd": float(
            abs(cat.s[red].mean() - cat.s[~red].mean()) / cat.s.std()
        ),
        "kde_grid_mass": float(enclosed[-1]),
        "direction": direction.tolist(),
        "cache_policy": "frozen historical first match per spectrum; no new quality cuts",
    }
    (outdir / "cmd_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
