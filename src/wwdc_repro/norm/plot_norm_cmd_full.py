"""Prepare NORM latent corrections and render the manuscript colour–magnitude diagram."""

from wwdc_repro.shared.cmd import bin_scores, join_photometry, masters10_red_edge, masters10_ridge
from pathlib import Path
import json
import numpy as np
import pandas as pd
import sympy as sp
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from wwdc_repro.shared.figstyle import STYLE, INK, INK_MUTED, SURFACE, tidy
from wwdc_repro.shared.equations import to_sympy, evaluate

OUT = Path("norm_cmd_full_2026-10-18")  # final figure dir
BASE = Path("conditioning_validation_2026-10-13")  # where equations are saved
DATA_OUT = Path("norm_cmd_s_2026-10-16")  # where plot data is saved


def prepare_data():
    """Check the additive term and save a unit-coefficient latent score with photometry."""
    DATA_OUT.mkdir(exist_ok=True)
    source = BASE / "validation_selected_final.csv"
    selected = pd.read_csv(source)
    r = selected[
        (selected.arm == "norm_cond+z")
        & (selected.seed == 42)
        & (selected.budget == 15)
    ].iloc[0]
    assert r.Length == 15
    expr = to_sympy(r.equation, 12)
    variables = [sp.Symbol(f"X{i}") for i in range(1, 11)]
    derivatives = [sp.diff(expr, v) for v in variables]
    assert all(not d.free_symbols for d in derivatives)
    weights = np.array([float(d) for d in derivatives])
    coefficient_norm = np.linalg.norm(weights)
    if not np.isfinite(coefficient_norm) or coefficient_norm == 0:
        raise ValueError("Latent coefficients must have a finite, nonzero norm")
    direction = weights / coefficient_norm
    mats_path = BASE / "norm_cond+z_matrices.npz"
    mats = np.load(mats_path)
    cache = Path("plot_cmd_latent_results/vagc_join.parquet")
    pieces = []
    for split in ("train", "val", "test"):
        x = mats[f"X_{split}"]
        ids = np.load(BASE / f"norm_{split}_ids.npy").astype("int64")
        latent_contribution = x[:, :10] @ weights
        s = x[:, :10] @ direction
        zero = x.copy()
        zero[:, :10] = 0
        np.testing.assert_allclose(
            latent_contribution,
            evaluate(r.equation, x) - evaluate(r.equation, zero),
            atol=1e-10,
            rtol=0,
        )  # quick check that the latent contribution is the same as the reduction
        pieces.append(
            pd.DataFrame({"specObjID": ids, "split": split, "z": x[:, 11], "S": s})
        )
    cat = pd.concat(pieces, ignore_index=True)
    df = join_photometry(cat, cache, validate="many_to_one")
    df = df[np.isfinite(df.S) & (df.z > 0.02) & (df.z <= 0.3)].copy()
    df.to_parquet(DATA_OUT / "plot_data.parquet", index=False)
    return df


def plot_cmd(d):
    """Render the full-sample CMD and save its numerical plotting products."""
    OUT.mkdir(exist_ok=True)
    xe = np.linspace(-24, -18, 61)
    ye = np.linspace(0.15, 1.2, 57)
    counts, mean = bin_scores(d.Mr, d.gr, d.S, xe, ye, min_count=25)
    limit = float(np.nanpercentile(np.abs(mean), 98))
    sample = d.sample(min(60000, len(d)), random_state=0)
    # Seaborn levels are low-density tail fractions: 0.1 encloses 90%.
    probability_levels = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    plt.style.use(STYLE)
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.6), sharex=True, sharey=True, gridspec_kw={"wspace": 0.30}
    )
    sns.kdeplot(
        data=sample,
        x="Mr",
        y="gr",
        ax=axes[0],
        fill=True,
        levels=probability_levels,
        cmap="Greys",
        gridsize=180,
        bw_method="scott",
        bw_adjust=1,
        cut=3,
        clip=((-24, -18), (0.15, 1.2)),
    )
    filled = axes[0].collections[-1]
    filled.set_edgecolor(INK_MUTED)
    filled.set_linewidth(0.7)
    cb0 = fig.colorbar(
        filled,
        ax=axes[0],
        fraction=0.046,
        pad=0.02,
        ticks=filled.levels[:-1],
    )
    cb0.ax.set_yticklabels([f"{1 - p:.0%}" for p in probability_levels[:-1]])
    cb0.set_label("enclosed KDE probability", fontsize=14)
    im = axes[1].pcolormesh(
        xe,
        ye,
        mean.T,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        rasterized=True,
    )
    cb1 = fig.colorbar(
        im,
        ax=axes[1],
        fraction=0.046,
        pad=0.02,
        ticks=np.linspace(-limit, limit, 5),
        format="%.1f",
    )
    cb1.set_label(r"$\langle S\rangle$ (dimensionless)", fontsize=16)
    grid = np.linspace(-24, -18, 200)
    for ax in axes:
        ax.plot(
            grid,
            masters10_red_edge(grid),
            linestyle=(0, (5.5, 2.4)),
            color=INK,
            lw=1.6,
            zorder=5,
            label="Blue edge of the red sequence (Masters et al. 2010)",
        )
        ax.plot(
            grid,
            masters10_ridge(grid),
            linestyle=(0, (1.2, 1.6)),
            color=INK_MUTED,
            lw=1.3,
            zorder=5,
            label="Red-sequence fit (Masters et al. 2010)",
        )
        ax.set(xlim=(-18, -24), ylim=(0.15, 1.0))
        ax.set_xlabel(r"$M_r$", fontsize=16)
        ax.set_ylabel(r"$g-r$", fontsize=16)
        tidy(ax, grid_axis=None, labelsize=12)
        ax.tick_params(labelleft=True)
    leg = axes[0].legend(loc="lower left", fontsize=10, frameon=True, borderpad=0.5)
    leg.get_frame().set_edgecolor(INK_MUTED)
    leg.get_frame().set_linewidth(0.6)
    leg.get_frame().set_facecolor(SURFACE)
    for cb in (cb0, cb1):
        cb.outline.set_edgecolor(INK_MUTED)
        cb.outline.set_linewidth(0.6)
        cb.ax.tick_params(labelsize=12)
    fig.savefig(
        OUT / "norm_cmd_full.pdf", dpi=400, bbox_inches="tight", facecolor=SURFACE
    )
    fig.savefig(
        OUT / "norm_cmd_full.png", dpi=400, bbox_inches="tight", facecolor=SURFACE
    )
    plt.close(fig)
    np.savez(
        OUT / "bins_and_contours.npz",
        counts=counts,
        mean_S=mean,
        xedges=xe,
        yedges=ye,
        contour_fractions=1 - np.asarray(probability_levels[:-1]),
        contour_levels=filled.levels[:-1],
    )
    meta = dict(
        rows=len(d),
        splits=d.split.value_counts().to_dict(),
        kde_rows=len(sample),
        kde_seed=0,
        kde_implementation=f"seaborn {sns.__version__} kdeplot",
        kde_bandwidth="scott",
        kde_bw_adjust=1,
        kde_cut=3,
        kde_clip=[[-24, -18], [0.15, 1.2]],
        kde_grid=180,
        kde_normalization="Probability normalized over Seaborn evaluation grid",
        min_count=25,
        colour_correlation=float(d.S.corr(d.gr)),
        score="Unit-length latent coefficients; increasing score increases predicted mass",
        colour_limit=limit,
        contours="10,30,50,70,90 percent of estimated KDE probability mass; not exact empirical sample fractions",
        counts="All matched spectra; retained for minimum-count masking of S, not displayed in left panel",
        left_panel="Filled grey KDE probability contours matching the original-latent CMD style",
    )
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def main():
    """Prepare checked plotting rows, then render the manuscript figure."""
    data = prepare_data()
    plot_cmd(data)


if __name__ == "__main__":
    main()
