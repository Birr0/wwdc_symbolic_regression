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

from .cli import COLOURS, draw_frontier, sha256
from .data import clean_target, load_sample, matrices, sample_summary
from .equations import evaluate
from .frontier import ARMS, SEEDS, curves, front_path

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
    ax.set(xlabel=r"$s = \sum_k w_k X_k$", ylabel="density", xlim=(-4.2, 4.2))
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    save(fig, outdir / "score_distribution_marginal")
    (outdir / "score_distribution_marginal.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


def latent_figure(embeddings: Path, outdir: Path) -> None:
    files = sorted((embeddings / "test").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No test embedding parquet files under {embeddings}")
    frame = pd.concat(
        (pd.read_parquet(path, columns=["orig", "cond"]) for path in files),
        ignore_index=True,
    )
    orig, cond = np.stack(frame.orig.to_numpy()), np.stack(frame.cond.to_numpy())
    if orig.shape != cond.shape or orig.ndim != 2 or orig.shape[1] != 10:
        raise ValueError(
            f"Expected paired ten-dimensional latents; got {orig.shape} and {cond.shape}"
        )
    if not (np.isfinite(orig).all() and np.isfinite(cond).all()):
        raise ValueError("Non-finite latent values")
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


def seed_frontiers(selected: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, dpi=250)
    for ax, seed in zip(axes, SEEDS, strict=True):
        for arm in ARMS:
            group = selected[
                (selected.seed == seed) & (selected.arm == arm)
            ].sort_values("budget")
            ax.plot(
                group.budget,
                group.test_r2,
                color=COLOURS[arm],
                label=arm,
                linewidth=1.5,
            )
        ax.set(title=f"seed {seed}", xlabel="equation length budget", ylim=(0, 0.92))
    axes[0].set_ylabel("test $R^2$ (validation selected)")
    axes[-1].legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    save(fig, outdir / "frontier_by_seed")


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
    needs_sample = bool(set(args.only) & {"score", "anatomy", "zero-latent", "cmd"})
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
        draw_frontier(curves(selected), args.outdir)
    if "seed-frontiers" in args.only:
        seed_frontiers(selected, args.outdir)
    if "latent" in args.only:
        latent_figure(args.embeddings, args.outdir)
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
    if "score" in args.only:
        score_figure(frames, selected, args.outdir)
    from .paper_panels import anatomy, zero_latent, cmd

    if "anatomy" in args.only:
        anatomy(frames, selected, args.outdir, score_direction, selected_l24, save)
    if "zero-latent" in args.only:
        zero_latent(frames, args.fronts, args.outdir, save)
    if "cmd" in args.only:
        cmd(frames, selected, args.cmd_cache, args.outdir, selected_l24, save)
    print(f"Wrote {', '.join(args.only)} figures to {args.outdir}")


if __name__ == "__main__":
    main()
