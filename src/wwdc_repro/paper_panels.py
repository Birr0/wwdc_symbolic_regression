"""Audited numerical recipes for the zero-latent, anatomy, and CMD figures.

Plot styling is deliberately separate from the saved numerical products. The
zero-latent panel preserves the historical six-significant-digit reductions.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
import sympy as sp
from scipy.stats import gaussian_kde

from .data import matrices
from .equations import evaluate
from .frontier import SEEDS, front_path


def anatomy(frames, selected, outdir: Path, direction, equation_for_seed, save):
    data = matrices(frames, "cond+z")
    x, y = data["X_test"], data["y_test"]
    z = x[:, -1]
    grid = np.linspace(z.min(), z.max(), 300)
    zero = np.zeros((len(grid), x.shape[1]))
    zero[:, -1] = grid
    sample = x[np.random.default_rng(0).choice(len(x), 4000, replace=False)]
    fig, axes = plt.subplots(
        1, 3, figsize=(13.2, 4.3), sharex=True, sharey=True, layout="constrained"
    )
    records = []
    for ax, seed in zip(axes, SEEDS, strict=True):
        equation = equation_for_seed(selected, seed)
        score = x[:, :-1] @ direction(equation, z, x.shape[1])
        baseline = evaluate(equation, zero)
        spread = []
        for redshift in grid:
            pinned = sample.copy()
            pinned[:, -1] = redshift
            spread.append(evaluate(equation, pinned).std())
        spread = np.asarray(spread)
        sc = ax.scatter(
            z,
            y,
            c=score,
            s=3,
            alpha=0.4,
            linewidths=0,
            cmap="RdBu_r",
            vmin=-2.2,
            vmax=2.2,
            rasterized=True,
        )
        ax.plot(grid, baseline, color="black", lw=1.6, label="zero-latent prediction")
        ax.fill_between(
            grid, baseline - spread, baseline + spread, color="black", alpha=0.06
        )
        for k, style in ((1, "--"), (2, ":")):
            ax.plot(
                grid,
                baseline + k * spread,
                color="black",
                ls=style,
                lw=0.9,
                label=f"±{k} prediction SD",
            )
            ax.plot(grid, baseline - k * spread, color="black", ls=style, lw=0.9)
        ax.set(
            title=f"seed {seed}", xlabel="redshift", xlim=(0.02, 0.275), ylim=(9, 12)
        )
        records.extend(
            {
                "seed": seed,
                "z": float(zi),
                "baseline": float(mu),
                "prediction_sd": float(sd),
            }
            for zi, mu, sd in zip(grid, baseline, spread, strict=True)
        )
    axes[0].set_ylabel(r"$\log_{10} M_*$ (observed)")
    axes[0].legend(fontsize=7, loc="lower right")
    colorbar = fig.colorbar(
        sc, ax=axes, label="latent score s (per seed)", fraction=0.02, pad=0.015
    )
    colorbar.solids.set_alpha(1)
    save(fig, outdir / "equation_anatomy_by_seed")
    pd.DataFrame(records).to_csv(outdir / "equation_anatomy_by_seed.csv", index=False)


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
    axes[0].hist2d(z, mass, bins=[120, 120], cmap="Greys", cmin=1, alpha=0.55)
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
        ylim=(lo - 0.12 * (hi - lo), hi + 0.12 * (hi - lo)), ylabel=r"$\log_{10} M_*$"
    )
    axes[0].legend(fontsize=8)
    axes[1].axhline(0, color="black", ls="--")
    axes[1].set(
        xlim=(0.02, 0.3), xlabel="redshift", ylabel="prediction − observed mean"
    )
    save(fig, outdir / "zero_latent")
    pd.DataFrame(all_records).to_csv(outdir / "zero_latent.csv", index=False)
    (outdir / "zero_latent_exclusions.json").write_text(
        json.dumps(exclusions, indent=2) + "\n"
    )


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
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.6), sharex=True, sharey=True, layout="constrained"
    )
    axes[0].contourf(kx, ky, density, levels=[*levels, density.max()], cmap="Greys")
    axes[0].contour(kx, ky, density, levels=levels, colors="#777777", linewidths=0.7)
    limit = float(np.nanpercentile(np.abs(mean), 98))
    image = axes[1].pcolormesh(
        xedges, yedges, mean.T, cmap="RdBu_r", vmin=-limit, vmax=limit, rasterized=True
    )
    fig.colorbar(image, ax=axes[1], label="mean latent score s")
    grid = np.linspace(-24, -18, 200)
    for ax in axes:
        ax.plot(
            grid,
            0.73 - 0.02 * (grid + 20),
            color="black",
            lw=1,
            label="red-sequence ridge",
        )
        ax.plot(
            grid,
            0.63 - 0.02 * (grid + 20),
            color="black",
            ls="--",
            lw=1,
            label="blue edge",
        )
        ax.set(xlim=(-18, -24), ylim=(0.15, 1), xlabel=r"$M_r$ ($H_0=70$ scale)")
    axes[0].set(ylabel="g − r", title="density contours (10–90% KDE mass)")
    axes[1].set_title("mean score; at least 25 spectra per cell")
    axes[0].legend(fontsize=7)
    save(fig, outdir / "cmd_latent_all_gr_seed42")
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
