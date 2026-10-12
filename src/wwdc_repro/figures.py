"""Figures using saved embeddings and validation-selected equations."""

from __future__ import annotations

import argparse
import json
import importlib.metadata
import importlib.resources
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .cli import sha256
from .data import clean_target, load_sample, matrices, sample_summary
from .equations import evaluate
from .figstyle import (
    ARM_COLOUR, ARM_LABEL, ARM_ORDER, STYLE, SURFACE, frontier_legend, tidy,
)
from .frontier import SEEDS, curves, front_path

SCORE_COLOURS = {42: "#b2182b", 43: "#2166ac", 44: "#1b7837"}


def save(fig, stem: Path, dpi: int = 250) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def selected_l24(selected: pd.DataFrame, seed: int) -> str:
    row = selected[
        (selected.arm == "cond+z") & (selected.seed == seed) & (selected.budget == 24)
    ]
    if len(row) != 1 or int(row.iloc[0].chosen_length) != 24:
        raise ValueError(f"Expected one selected L=24 cond+z equation for seed {seed}")
    return row.iloc[0].equation


def score_direction(equation: str, z: np.ndarray, n_features: int) -> np.ndarray:
    """Leading singular vector of central-difference latent slopes across z."""
    grid = np.linspace(np.percentile(z, 0.5), np.percentile(z, 99.5), 120)
    base = np.zeros((len(grid), n_features))
    base[:, -1] = grid
    step = 0.05
    slopes = []
    for column in range(n_features - 1):
        plus, minus = base.copy(), base.copy()
        plus[:, column] = step
        minus[:, column] = -step
        slopes.append(
            (evaluate(equation, plus) - evaluate(equation, minus)) / (2 * step)
        )
    u, singular, vt = np.linalg.svd(np.asarray(slopes), full_matrices=False)
    direction = u[:, 0]
    amplitude = singular[0] * vt[0]
    return -direction if amplitude.mean() < 0 else direction


def score_figure(
    frames: dict[str, pd.DataFrame], selected: pd.DataFrame, outdir: Path
) -> None:
    data = matrices(frames, "cond+z")
    x = data["X_test"]
    z = x[:, -1]
    bins = np.linspace(-4.2, 4.2, 90)
    fig, ax = plt.subplots(figsize=(6, 4.4), dpi=300)
    summary = []
    for seed in SEEDS:
        direction = score_direction(selected_l24(selected, seed), z, x.shape[1])
        score = x[:, :-1] @ direction
        ax.hist(
            score,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.4,
            color=SCORE_COLOURS[seed],
            label=f"seed {seed}",
        )
        summary.append(
            {
                "seed": seed,
                "rows": len(score),
                "mean": float(score.mean()),
                "sd": float(score.std()),
                "skew": float(stats.skew(score)),
                "excess_kurtosis": float(stats.kurtosis(score)),
                "direction": direction.tolist(),
            }
        )
    grid = np.linspace(-4.2, 4.2, 400)
    ax.plot(
        grid,
        stats.norm.pdf(grid),
        color="#222222",
        linestyle="--",
        linewidth=1.5,
        label=r"$\mathcal{N}(0,1)$",
    )
    ax.set(xlabel=r"$s$", ylabel="density", xlim=(-4.2, 4.2))
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    save(fig, outdir / "score_distribution_marginal")
    (outdir / "score_distribution_marginal.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


def latent_figure(frame: pd.DataFrame, outdir: Path) -> None:
    """Use the exact cleaned paper test sample, verified by main against reference IDs."""
    frame[["specObjID"]].to_csv(outdir / "latent_sample_ids.csv", index=False)
    orig, cond = np.stack(frame.orig.to_numpy()), np.stack(frame.cond.to_numpy())
    if orig.shape != cond.shape or orig.ndim != 2 or orig.shape[1] != 10:
        raise ValueError(
            f"Expected paired ten-dimensional latents; got {orig.shape} and {cond.shape}"
        )
    if not (np.isfinite(orig).all() and np.isfinite(cond).all()):
        raise ValueError("Non-finite latent values")
    from .latent_corner import latent_corner

    latent_corner(orig, cond, outdir)
    grid, edges = np.linspace(-5, 5, 600), np.linspace(-5, 5, 61)
    normal = stats.norm.pdf(grid)
    fig, axes = plt.subplots(5, 2, figsize=(10, 13), constrained_layout=True)
    summary = {
        "split": "test",
        "rows": len(frame),
        "bins": 60,
        "histogram_range": [-5, 5],
        "dimensions": [],
    }
    for i, ax in enumerate(axes.flat):
        ax.hist(orig[:, i], bins=edges, density=True, color="#1261a6", label="Original")
        ax.hist(
            cond[:, i],
            bins=edges,
            density=True,
            color="#46c57a",
            alpha=0.72,
            label="Conditional",
        )
        ax.plot(grid, normal, color="red", lw=1.6, label=r"$\mathcal{N}(0,1)$")
        ax.set(
            title=f"Dimension {i + 1}", xlabel="Value", ylabel="Density", xlim=(-5, 5)
        )
        summary["dimensions"].append(
            {
                "dimension": i + 1,
                "original_mean": float(orig[:, i].mean()),
                "original_sd": float(orig[:, i].std()),
                "conditional_mean": float(cond[:, i].mean()),
                "conditional_sd": float(cond[:, i].std()),
            }
        )
    axes.flat[-1].legend(loc="upper right", fontsize=8)
    save(fig, outdir / "latent_dim_distributions")
    (outdir / "latent_dim_distributions.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


def _frontier_axes(ax, lo: float, hi: float, top: float) -> None:
    ax.set_xlim(lo - 0.8, hi + 0.8)
    ax.set_ylim(0.0, top + 0.05)
    ax.set_xlabel("maximum equation length", fontsize=13)
    tidy(ax, grid_axis=None, labelsize=11)


@plt.style.context(STYLE)
def paper_frontier(selected: pd.DataFrame, outdir: Path) -> None:
    """Paper Figure 1: validation-selected test R^2, seed mean with min-max band.

    Rendering follows the historical plot_frontier.py; the selection does not: each
    point is the validation-selected equation at that length budget, scored on test.
    """
    display = frontier_display(selected)
    summary = display.groupby(["arm", "budget"]).agg(
        mean=("display_r2", "mean"), low=("display_r2", "min"),
        high=("display_r2", "max"), n=("display_r2", "size"),
        n_finite=("display_r2", "count"), fallback_count=("display_fallback", "sum")
    ).reset_index()
    summary.loc[summary.n_finite < summary.n, ["mean", "low", "high"]] = np.nan
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=300, facecolor=SURFACE)
    for arm in ARM_ORDER:  # bands first, so no ribbon covers a neighbouring line
        c = summary[summary.arm == arm].sort_values("budget")
        ax.fill_between(c.budget, c.low, c.high, color=ARM_COLOUR[arm], alpha=0.22,
                        linewidth=0, zorder=2)
    for arm in ARM_ORDER:
        c = summary[summary.arm == arm].sort_values("budget")
        ax.plot(c.budget, c["mean"], linewidth=1.5,
                color=ARM_COLOUR[arm], label=ARM_LABEL[arm], zorder=3)
        _frontier_markers(ax, c.budget, c["mean"], c.fallback_count > 0, ARM_COLOUR[arm])
    _frontier_axes(ax, summary.budget.min(), summary.budget.max(), summary.high.max())
    ax.set_ylabel("$R^2$", fontsize=13)
    frontier_legend(ax)
    fig.tight_layout()
    stem = outdir / "frontier_lgm_tot_p50"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    summary.to_csv(outdir / "frontier_lgm_tot_p50.csv", index=False)
    display.to_csv(outdir / "frontier_display_values.csv", index=False)
    curves(selected).to_csv(outdir / "frontier_selected_scores.csv", index=False)


def frontier_display(selected: pd.DataFrame) -> pd.DataFrame:
    """Display-only fallback: best finite test score at a strictly lower budget.

    This is test-informed plotting, not validation selection. Never mutate test_r2
    or promote a fallback into the pool of actually evaluated selections.
    """
    display = selected.copy()
    display["display_r2"] = display.test_r2
    display["display_fallback"] = False
    display["fallback_budget"] = np.nan
    display["fallback_length"] = np.nan
    for _, group in display.groupby(["arm", "seed"]):
        for index, row in group[~np.isfinite(group.test_r2)].iterrows():
            pool = group[(group.budget < row.budget) & np.isfinite(group.test_r2)]
            if pool.empty:
                continue
            best = pool.sort_values(["test_r2", "budget"], ascending=[False, True]).iloc[0]
            display.loc[index, ["display_r2", "display_fallback", "fallback_budget", "fallback_length"]] = [
                best.test_r2, True, best.budget, best.chosen_length]
    return display


def _frontier_markers(ax, budget, value, fallback, colour):
    ax.plot(budget[~fallback], value[~fallback], linestyle="none", marker="o",
            markersize=2.6, color=colour, zorder=4)
    ax.plot(budget[fallback], value[fallback], linestyle="none", marker="^",
            markersize=6, markerfacecolor=SURFACE, markeredgecolor=colour,
            markeredgewidth=1.1, zorder=5)


@plt.style.context(STYLE)
def seed_frontiers(selected: pd.DataFrame, outdir: Path) -> None:
    """Paper Figure 5: the same validation-selected test R^2, one panel per seed."""
    display = frontier_display(selected)
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.4), dpi=300, sharey=True,
                             facecolor=SURFACE)
    for ax, seed in zip(axes, SEEDS, strict=True):
        for arm in ARM_ORDER:
            c = display[(display.seed == seed) & (display.arm == arm)].sort_values("budget")
            ax.plot(c.budget, c.display_r2, linewidth=1.5,
                    color=ARM_COLOUR[arm], label=ARM_LABEL[arm], zorder=3)
            _frontier_markers(ax, c.budget, c.display_r2, c.display_fallback, ARM_COLOUR[arm])
        _frontier_axes(ax, selected.budget.min(), selected.budget.max(),
                       selected.test_r2.max())
        ax.set_title(f"seed {seed}", fontsize=12)
    axes[0].set_ylabel("$R^2$", fontsize=13)
    frontier_legend(axes[-1])
    fig.tight_layout()
    stem = outdir / "frontier_lgm_tot_p50_by_seed"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--galspec", type=Path, required=True)
    parser.add_argument("--specgals-home", type=Path, required=True)
    parser.add_argument(
        "--selected", type=Path, required=True, help="selected.csv from wwdc-reproduce"
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--fronts", type=Path, help="Saved front directory; required for zero-latent"
    )
    parser.add_argument(
        "--cmd-cache", type=Path, help="Frozen VAGC join parquet; required for CMD"
    )
    names = [
        "frontier",
        "score",
        "latent",
        "seed-frontiers",
        "anatomy",
        "zero-latent",
        "cmd",
        "weights",
    ]
    parser.add_argument("--only", nargs="+", default=names, choices=names)
    args = parser.parse_args()
    if args.outdir.exists() and any(args.outdir.iterdir()):
        parser.error(f"Output directory must be new or empty: {args.outdir}")
    if "zero-latent" in args.only and args.fronts is None:
        parser.error("--fronts is required for zero-latent")
    if "cmd" in args.only and args.cmd_cache is None:
        parser.error("--cmd-cache is required for cmd")
    inputs = [args.selected]
    needs_sample = bool(set(args.only) & {"score", "anatomy", "zero-latent", "cmd", "latent", "weights"})
    if needs_sample:
        inputs.extend(
            [
                args.galspec,
                *(args.specgals_home / f"SDSSspecgalsDR8_{i}.fit.gz" for i in (1, 2)),
            ]
        )
    for split in (
        ("train", "val", "test")
        if needs_sample
        else ("test",)
        if "latent" in args.only
        else ()
    ):
        shards = sorted((args.embeddings / split).glob("*.parquet"))
        if not shards:
            parser.error(f"No embeddings for {split}")
        inputs.extend(shards)
    if "zero-latent" in args.only:
        inputs.extend(
            front_path(args.fronts, arm, seed)
            for arm in ("cond+z", "orig+z")
            for seed in SEEDS
        )
    if "cmd" in args.only:
        inputs.append(args.cmd_cache)
    for path in inputs:
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "figures": args.only,
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "pandas",
                "sympy",
                "scipy",
                "matplotlib",
                "scikit-learn",
                "astropy",
            )
        },
        "inputs": [
            {"path": str(p.resolve()), "sha256": sha256(p), "bytes": p.stat().st_size}
            for p in inputs
        ],
        "code": [
            {"file": p.name, "sha256": sha256(p)}
            for p in sorted(Path(__file__).parent.glob("*.py"))
        ],
    }
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    selected = pd.read_csv(args.selected)
    if selected.duplicated(["arm", "seed", "budget"]).any():
        raise ValueError("Duplicate selection keys")
    if "frontier" in args.only:
        paper_frontier(selected, args.outdir)
    if "seed-frontiers" in args.only:
        seed_frontiers(selected, args.outdir)
    if needs_sample:
        frames = clean_target(
            load_sample(args.embeddings, args.galspec, args.specgals_home)
        )
        reference = json.loads(
            importlib.resources.files("wwdc_repro")
            .joinpath("reference.json")
            .read_text()
        )
        for row in sample_summary(frames):
            expected = reference["samples"][row["split"]]
            if any(row[key] != expected[key] for key in ("rows", "id_sha256")):
                raise ValueError(f"Sample differs from paper reference: {row['split']}")
    if "latent" in args.only:
        latent_figure(frames["test"], args.outdir)
    if "score" in args.only:
        score_figure(frames, selected, args.outdir)
    from .paper_panels import anatomy, zero_latent, cmd

    if "anatomy" in args.only:
        anatomy(frames, selected, args.outdir, score_direction, selected_l24, save)
    if "zero-latent" in args.only:
        zero_latent(frames, args.fronts, args.outdir, save)
    if "cmd" in args.only:
        cmd(frames, selected, args.cmd_cache, args.outdir, selected_l24, save)
    if "weights" in args.only:
        from .weights import weights_table

        x = matrices(frames, "cond+z")["X_test"]
        weights_table(
            selected, x[:, -1], x.shape[1], args.outdir, score_direction, selected_l24
        )
    print(f"Wrote {', '.join(args.only)} figures to {args.outdir}")


if __name__ == "__main__":
    main()
