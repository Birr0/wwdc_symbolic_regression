"""Figures using saved embeddings and training- or validation-selected equations."""

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

from wwdc_repro.shared.utils import sha256
from wwdc_repro.shared.data import clean_target, load_sample, matrices, sample_summary
from wwdc_repro.shared.equations import evaluate
from wwdc_repro.shared.figstyle import (
    ARM_COLOUR, ARM_LABEL, ARM_ORDER, STYLE, SURFACE, frontier_legend, tidy,
)
from wwdc_repro.redshift.frontier import SEEDS, curves, front_path

SCORE_COLOURS = {42: "#b2182b", 43: "#2166ac", 44: "#1b7837"}


def save(fig, stem: Path, dpi: int = 250) -> None:
    '''
    Helper function to save figure as png and pdf

    Parameters
    ----------
    fig: matplotlib.figure.Figure
        Matplotlib figure to save
    stem: Path 
        path to save figure to
    dpi: int 
        Resolution of plot
    
    Returns 
    -------
    None 
    '''
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def selected_l24(selected: pd.DataFrame, seed: int) -> str:
    '''
    Helper function to select the length 42 equation from the cond+z arm 

    Parameters
    ----------
    selected: pd.DataFrame
        Dataframe of selected equations 
    seed: int 
        seed from the equation run 
    
    Returns 
    -------
    equation: str 
        The cond+z L=24 equation for a given seed
    
    Raises
    ------
    ValueError
        If equation is not found
    '''
    row = selected[
        (selected.arm == "cond+z") & (selected.seed == seed) & (selected.budget == 24)
    ]
    if len(row) != 1 or int(row.iloc[0].chosen_length) > 24:
        raise ValueError(f"Expected one selected budget-24 cond+z equation for seed {seed}")
    return row.iloc[0].equation

@plt.style.context(STYLE)
def latent_figure(frame: pd.DataFrame, outdir: Path) -> None:
    """Plot the paired test-sample latent corner figure used in the appendix.

    Parameters
    ----------
    frame : pd.DataFrame
        Cleaned reference test sample with specObjID, orig, and cond columns.
        Each embedding cell must contain a vector of shape (10,). main() checks
        the sample against reference IDs before calling this function.
    outdir : Path
        Existing output directory. Saves latent_sample_ids.csv, the corner
        plot and its metadata via latent_corner.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If embedding arrays cannot be stacked, have unexpected or mismatched
        shapes, or contain nonfinite values.

    """
    # Record the sample IDs; main() has already checked them against the reference.

    frame[["specObjID"]].to_csv(outdir / "latent_sample_ids.csv", index=False)
    orig, cond = np.stack(frame.orig.to_numpy()), np.stack(frame.cond.to_numpy())
    if orig.shape != cond.shape or orig.ndim != 2 or orig.shape[1] != 10:
        raise ValueError(
            f"Expected paired ten-dimensional latents; got {orig.shape} and {cond.shape}"
        )
    if not (np.isfinite(orig).all() and np.isfinite(cond).all()):
        raise ValueError("Non-finite latent values")
    from wwdc_repro.redshift.latent_corner import latent_corner

    latent_corner(orig, cond, outdir)  # Save the corner plot and its metadata.

def _frontier_axes(ax, lo: float, hi: float, top: float) -> None:
    '''
    Format the frontier axes: set x and y limits, x label and grid formatting.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        Axes to format 
    lo: float
        Lowest value in plot x range (lo - 0.8)
    hi: float 
        Highest value in plot x range (hi + 0.8)
    top: 
        max y value to plot (top + 0.05)
    
    Returns
    -------
    None 
    '''
    ax.set_xlim(lo - 0.8, hi + 0.8)
    ax.set_ylim(0.0, top + 0.05)
    ax.set_xlabel("Maximum equation length", fontsize=13)
    tidy(ax, grid_axis=None, labelsize=11)

@plt.style.context(STYLE)
def paper_frontier(selected: pd.DataFrame, outdir: Path) -> None:
    """Plot Paper Figure 1: test performance against maximum equation length.

    For each arm and budget, plot the mean test score across frozen choices
    with a minimum-to-maximum band. Any failed seed leaves a gap; no test-based
    replacement is applied. Selection policy is inherited from the input table.

    Parameters
    ----------
    selected : pd.DataFrame
        Output from evaluate_selected, containing training- or validation-selected equations
        and their test scores for each arm, seed, and length budget.
    outdir : Path
        Existing directory in which to save the PNG/PDF figure and CSV files
        containing the display summary, per-selection display values, and
        original selected-score summary without display replacements.

    Returns
    -------
    None
    """
    display = frontier_display(selected)  # Prepare plotting scores and fallback metadata.

    # Summarize plotted scores across seeds for each arm and budget.
    summary = display.groupby(["arm", "budget"]).agg(
        mean=("display_r2", "mean"), low=("display_r2", "min"),
        high=("display_r2", "max"), n=("display_r2", "size"),
        n_finite=("display_r2", "count"), fallback_count=("display_fallback", "sum")
    ).reset_index()

    # Leave a gap if any selected seed still has a missing score after fallback.
    summary.loc[summary.n_finite < summary.n, ["mean", "low", "high"]] = np.nan

    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=300, facecolor=SURFACE)
    for arm in ARM_ORDER:  # plot the min-max bands first.
        c = summary[summary.arm == arm].sort_values("budget")
        ax.fill_between(c.budget, c.low, c.high, color=ARM_COLOUR[arm], alpha=0.22,
                        linewidth=0, zorder=2)
    for arm in ARM_ORDER: # Plot the mean display scores across seeds.
        c = summary[summary.arm == arm].sort_values("budget")
        ax.plot(c.budget, c["mean"], linewidth=1.5,
                color=ARM_COLOUR[arm], label=ARM_LABEL[arm], zorder=3)
        _frontier_markers(ax, c.budget, c["mean"], c.fallback_count > 0, ARM_COLOUR[arm])
    _frontier_axes(ax, summary.budget.min(), summary.budget.max(), summary.high.max()) # format axes of plot
    ax.set_ylabel("$R^2$", fontsize=13)
    frontier_legend(ax, fontsize=9.5)
    fig.tight_layout()

    # Save the figure, display data, and original selected-score summary.
    stem = outdir / "frontier_lgm_tot_p50"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    summary.to_csv(outdir / "frontier_lgm_tot_p50.csv", index=False)
    display.to_csv(outdir / "frontier_display_values.csv", index=False)
    curves(selected).to_csv(outdir / "frontier_selected_scores.csv", index=False)


def frontier_display(selected: pd.DataFrame) -> pd.DataFrame:
    """Fill failed display scores with the best finite test score at a lower budget.

    This is a test-informed display fallback, not equation selection. Preserve
    the frozen equations and test_r2, and mark every substituted display value.
    """
    display = selected.copy()
    display['display_r2'] = display.test_r2.where(np.isfinite(display.test_r2))
    display['display_fallback'] = False
    display['fallback_budget'] = np.nan
    display['fallback_length'] = np.nan
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
    '''
    Select marker style. 
    This allows a marker style to be picked based on whether a lower budget equation is used. 

    Parameters 
    ----------
    ax: matplotlib.axes.Axes
        Axes to draw the markers on 
    budget: pd.Series
        Equation length budgets 
    value: pd.Series
        Scores to plot at those budgets
    fallback: pd.Series
        Boolean mask aligned with budget and value. True uses a hollow triangle for 
        fallbacks and false uses a filled circle. 
    colour: str 
        Matplotlib colour for the fills and outlines

    Returns
    -------
    None
    '''
    ax.plot(budget[~fallback], value[~fallback], linestyle="none", marker="o",
            markersize=2.6, color=colour, zorder=4)
    ax.plot(budget[fallback], value[fallback], linestyle="none", marker="^",
            markersize=6, markerfacecolor=SURFACE, markeredgecolor=colour,
            markeredgewidth=1.1, zorder=5)


@plt.style.context(STYLE)
def seed_frontiers(selected: pd.DataFrame, outdir: Path) -> None:
    """
    The training- or validation-selected equations plotted by their test score. We plot a panel for each search seed.

    The training- or validation-selected equations are plotted for each arm against complexity
    budget. Where a test score is nonfinite, use the display fallback from
    frontier_display and mark the replacement with a triangle.
    
    Parameters
    ----------
    selected: pd.DataFrame
        Validation-selected equations, including arm, seed, budget
        and test_r2.
    outdir: Path
        Directory to save figures
    
    Returns
    ------- 
    None 

    """
    display = frontier_display(selected) # return the display dataframe.

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
        ax.set_xlabel("Maximum equation length", fontsize=21)
        ax.tick_params(labelsize=19)
        ax.set_title(f"seed {seed}", fontsize=20)
    axes[0].set_ylabel("$R^2$", fontsize=21)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.10),
               ncol=3, fontsize=19, labelspacing=0.7, columnspacing=1.8,
               frameon=False)
    fig.tight_layout()
    stem = outdir / "frontier_lgm_tot_p50_by_seed"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__) # set up the command line args
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--galspec", type=Path, required=True)
    parser.add_argument("--specgals-home", type=Path, required=True)
    parser.add_argument(
        "--selected", type=Path, required=True, help="selected.csv from wwdc-reproduce"
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--fronts", type=Path, help="Saved front directory; required for zero-latent and front-tables"
    )
    parser.add_argument(
        "--cmd-cache", type=Path, help="Frozen VAGC join parquet; required for CMD plot"
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
        "front-tables",
        "diagnostic-tables",
    ]
    parser.add_argument("--only", nargs="+", default=names, choices=names)
    args = parser.parse_args()
    if args.outdir.exists() and any(args.outdir.iterdir()):
        parser.error(f"Output directory must be new or empty: {args.outdir}")
    if set(args.only) & {"zero-latent", "front-tables"} and args.fronts is None:
        parser.error("--fronts is required for zero-latent and front-tables")
    if "cmd" in args.only and args.cmd_cache is None:
        parser.error("--cmd-cache is required for cmd")
    inputs = [args.selected]

    needs_sample = bool(set(args.only) & {"score", "anatomy", "zero-latent", "cmd", "latent", "diagnostic-tables"}) 
    if needs_sample:
        inputs.extend(
            [
                args.galspec,
                *(args.specgals_home / f"SDSSspecgalsDR8_{i}.fit.gz" for i in (1, 2)),
            ]
        )
    for split in (("train", "val", "test") if needs_sample else ()):
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
    if "front-tables" in args.only:
        inputs.extend(front_path(args.fronts, "cond+z", seed) for seed in SEEDS)
    inputs = list(dict.fromkeys(inputs))
    if "cmd" in args.only:
        inputs.append(args.cmd_cache)
    for path in inputs:
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest = { # document the metadata assiciated with the figure run 
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
                "seaborn",
                "scikit-learn",
                "astropy",
            )
        },
        "inputs": [
            {"path": str(p.resolve()), "sha256": sha256(p), "bytes": p.stat().st_size}
            for p in inputs
        ],
        "code": [
            {"file": str(p.relative_to(Path(__file__).parents[1])), "sha256": sha256(p)}
            for p in sorted(Path(__file__).parents[1].rglob("*.py"))
        ],
    }
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    selected = pd.read_csv(args.selected) # get the validation selected equations
    if selected.duplicated(["arm", "seed", "budget"]).any():
        raise ValueError("Duplicate selection keys")
    if "frontier" in args.only:
        paper_frontier(selected, args.outdir) 
    if "seed-frontiers" in args.only:
        seed_frontiers(selected, args.outdir)
    if needs_sample:
        frames = clean_target(
            load_sample(args.embeddings, args.galspec, args.specgals_home)
        ) # load and clean the data (embeddings and galaxy properties)
        reference = json.loads(
            importlib.resources.files("wwdc_repro.resources")
            .joinpath("reference.json")
            .read_text()
        ) # load expected metadata for an additional check that processing has succeded 
        for row in sample_summary(frames): 
            expected = reference["samples"][row["split"]] 
            if any(row[key] != expected[key] for key in ("rows", "id_sha256")): # check these are the sames files that are used in the paper.
                raise ValueError(f"Sample differs from paper reference: {row['split']}")
    if "latent" in args.only:
        latent_figure(frames["test"], args.outdir)
    from wwdc_repro.redshift.paper_panels import anatomy, zero_latent, cmd

    if "anatomy" in args.only:
        anatomy(frames, selected, args.outdir, selected_l24)
    if "zero-latent" in args.only:
        zero_latent(frames, args.fronts, args.outdir, save)
    if "cmd" in args.only:
        cmd(frames, selected, args.cmd_cache, args.outdir, selected_l24)
    if "weights" in args.only:
        from wwdc_repro.redshift.weights import weights_table

        weights_table(selected, args.outdir, selected_l24)
    if "front-tables" in args.only:
        from wwdc_repro.redshift.front_tables import front_tables

        front_tables(args.fronts, args.outdir)
    if "diagnostic-tables" in args.only:
        from wwdc_repro.redshift.diagnostic_tables import truncation_table, coverage_table

        data = matrices(frames, "cond+z")
        truncation_table(data["X_test"], data["y_test"], selected, args.outdir, selected_l24)
        coverage_table(data["X_test"], selected, args.outdir, selected_l24)
    print(f"Wrote {', '.join(args.only)} figures to {args.outdir}")


if __name__ == "__main__":
    main()
