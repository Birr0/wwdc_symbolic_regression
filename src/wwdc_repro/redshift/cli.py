"""Command-line entry point for the first checkpoint-first reproduction stage."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.resources
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from wwdc_repro.shared.utils import sha256
from wwdc_repro.shared.data import TARGET, clean_target, load_sample, sample_summary, preprocessing_state
from wwdc_repro.redshift.frontier import ARMS, SEEDS, curves, front_path, score_fronts, select_by_budget, evaluate_selected

COLOURS = {
    "z": "#777777",
    "orig": "#9554a4",
    "uncond": "#527ab6",
    "cond": "#ce8154",
    "cond+z": "#348c75",
    "orig+z": "#a9516c",
}



def manifest(args: argparse.Namespace) -> dict:
    """Build a record for this run's inputs, code, and environment.

    Collect catalogue files, discovered embedding parquet files for each split,
    and saved fronts for every configured arm and seed. Record their absolute paths,
    sizes, and SHA-256 hashes so later runs can identify the exact inputs used.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments. Uses galspec, specgals_home, embeddings,
        and fronts as input paths, plus tolerance for training-score agreement.

    Returns
    -------
    record : dict
        JSON-serializable metadata containing the UTC timestamp, process command
        line, Python/platform details, selected dependency versions, target,
        training subsample size, and selection/test policies. Also includes
        hashes of package Python files and the recovered-training-score JSON,
        and path/size/hash records for the collected inputs. run() writes this
        dictionary to manifest.json.

    Raises
    ------
    FileNotFoundError
        If a collected input file or the recovery JSON is missing.
    importlib.metadata.PackageNotFoundError
        If metadata for a recorded dependency is unavailable.
    OSError
        If files cannot be read or their metadata cannot be accessed.
    """
    files = [
        args.galspec,
        *(args.specgals_home / f"SDSSspecgalsDR8_{i}.fit.gz" for i in (1, 2)),
    ]
    for split in ("train", "val", "test"):
        files.extend(sorted((args.embeddings / split).glob("*.parquet")))
    files.extend(front_path(args.fronts, arm, seed) for arm in ARMS for seed in SEEDS)
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))
    packages = {
        name: importlib.metadata.version(name)
        for name in (
            "numpy",
            "pandas",
            "sympy",
            "scipy",
            "scikit-learn",
            "astroML",
            "astropy",
            "pyarrow",
        )
    }
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "target": TARGET,
        "training_subsample": 10_000,
        "selection": args.selection,
        "tolerance_train_r2": args.tolerance,
        "test_policy": "Evaluate frozen validation selections; report failures without replacement or successful-seeds-only averaging",
        "recovered_training_scores_sha256": hashlib.sha256(importlib.resources.files(
            "wwdc_repro.resources").joinpath("recovered_training_scores.json").read_bytes()).hexdigest(),
        "code": [
            {"file": str(p.relative_to(Path(__file__).parents[1])), "sha256": sha256(p)}
            for p in sorted(Path(__file__).parents[1].rglob("*.py"))
        ],
        "inputs": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
    }


def draw_frontier(summary: pd.DataFrame, outdir: Path) -> None:
    """
    Plot the validation-selected equations on the test set against equation-length budget.
    This produces a Pareto-like frontier of equations.

    Parameters
    ----------
    summary : pd.DataFrame
        Output from frontier.curves. Each arm is drawn with its mean test score
        across seeds and a shaded minimum-to-maximum range. 
        Budgets with failed evaluations have missing summary values.
    outdir : Path
        Existing directory in which to save frontier.png and frontier.pdf.

    Returns
    -------
    None
    
    """
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=300)
    for arm in ARMS:
        group = summary[summary.arm == arm].sort_values("budget")
        if group.empty:
            continue
        x = group.budget.to_numpy()
        ax.fill_between(
            x,
            group.low.to_numpy(),
            group.high.to_numpy(),
            color=COLOURS[arm],
            alpha=0.17,
            linewidth=0,
        )
        ax.plot(
            x,
            group["mean"].to_numpy(),
            color=COLOURS[arm],
            linewidth=1.5,
            marker="o",
            markersize=2.5,
            label=arm,
        )
    ax.set(xlabel="equation length budget", ylabel="test $R^2$ (validation selected)")
    ax.set_ylim(0, max(0.9, float(summary.high.max()) + 0.04))
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "frontier.png", dpi=300)
    fig.savefig(outdir / "frontier.pdf")
    plt.close(fig)


def check_reference(
    samples: pd.DataFrame, scores: pd.DataFrame, selected: pd.DataFrame, selection="train", *, fresh_fronts=False
) -> list[dict]:
    """Compare this run with the expected values stored in reference.json.

    Check split row counts, sorted and ordered spectrum-ID hashes, candidate
    and eligible-candidate counts, and the number of arm/seed/budget selections.
    Also compare each seed's cond+z test score at maximum length budget 24. The selected
    equation itself can be shorter than 24. Counts and hashes must match exactly.
    Test scores use the reference metric tolerance.

    Parameters
    ----------
    samples : pd.DataFrame
        DataFrame constructed from the records returned by sample_summary.
    scores : pd.DataFrame
        Candidate training/validation scores and eligibility from score_fronts.
    selected : pd.DataFrame
        Validation-selected equations with test scores from evaluate_selected.
        Must include a cond+z selection at budget 24 for every configured seed.

    Returns
    -------
    checks : list[dict]
        Comparison records containing name, actual, expected, tolerance, and
        passed. A failed comparison is recorded here. run() saves records
        to checks.json and raises an error if any comparison failed."""
    reference = json.loads(
        importlib.resources.files("wwdc_repro.resources").joinpath("reference.json").read_text()
    )
    checks = []

    def add(name: str, actual, expected, tolerance: float = 0) -> None:
        passed = (
            abs(actual - expected) <= tolerance if tolerance else actual == expected
        )
        checks.append(
            {
                "name": name,
                "actual": actual,
                "expected": expected,
                "tolerance": tolerance,
                "passed": bool(passed),
            }
        )

    for row in samples.itertuples():
        expected = reference["samples"][row.split]
        add(f"{row.split}.rows", row.rows, expected["rows"])
        add(f"{row.split}.id_sha256", row.id_sha256, expected["id_sha256"])
        add(f"{row.split}.ordered_id_sha256", row.ordered_id_sha256, expected["ordered_id_sha256"])
    if fresh_fronts:
        return checks  # Fresh searches need not reproduce historical equations/counts.
    add("candidate_count", len(scores), reference["candidate_count"])
    if selection != "validation":
        return checks  # Remaining archived references describe validation selection.
    add("selectable_count", int(scores.selectable.sum()), reference["selectable_count"])
    add("selected_budget_count", len(selected), reference["selected_budget_count"])
    for seed in SEEDS:
        row = selected[
            (selected.arm == "cond+z")
            & (selected.seed == seed)
            & (selected.budget == 24)
        ]
        add(
            f"seed{seed}.L24.test_r2",
            float(row.iloc[0].test_r2),
            reference["l24_test_r2"][str(seed)],
            reference["metric_tolerance"],
        )
    return checks


def run(args: argparse.Namespace) -> None:
    '''
    Coordinate the analysis and write outputs for the equation frontiers.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line options:
        embeddings : Path
            Root directory containing train/val/test embeddings.
        galspec : Path
            Path to the galSpecExtra FITS catalogue.
        specgals_home : Path
            Directory containing the astroML SDSS catalogue files.
        fronts : Path
            Root directory containing saved equation fronts.
        outdir : Path
            Output directory, which must be new or empty.
        tolerance : float
            Maximum absolute training score difference allowed.
            The command-line default is 1e-6.
 
    Returns
    -------
    None
    '''
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise ValueError(f"Output directory must be new or empty: {args.outdir}")
    record = manifest(args)
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")

    raw = load_sample(args.embeddings, args.galspec, args.specgals_home)
    clean = clean_target(raw)

    # Save preprocessing/sample metadata; check reference consistency and split integrity.
    (args.outdir / "preprocessing.json").write_text(
        json.dumps(preprocessing_state(clean), indent=2) + "\n"
    )
    pd.concat(
        [frame[["specObjID"]].assign(split=split) for split, frame in clean.items()],
        ignore_index=True,
    ).to_csv(args.outdir / "sample_ids.csv.gz", index=False)
    samples = pd.DataFrame(sample_summary(clean))
    samples.to_csv(args.outdir / "samples.csv", index=False)
    reference = json.loads(importlib.resources.files("wwdc_repro.resources").joinpath("reference.json").read_text())
    for row in samples.itertuples():
        if row.ordered_id_sha256 != reference["samples"][row.split]["ordered_id_sha256"]:
            raise ValueError(f"Sample row order differs for {row.split}; inspect samples.csv before drawing training rows")
    expected = {"train": 352896, "val": 44246, "test": 44141}
    for row in samples.itertuples():
        if row.rows != expected[row.split] or row.unique_specObjID != row.rows:
            raise ValueError(
                f"Sample mismatch for {row.split}: {row.rows} rows, {row.unique_specObjID} unique IDs"
            )
    ids = {split: set(clean[split].specObjID) for split in expected}
    if any(
        ids[a] & ids[b]
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise ValueError("Spectrum IDs overlap between splits")
    
    # TODO: Replace the recovery JSON lookup with documented corrections to saved fronts.
    scores = score_fronts(clean, args.fronts, args.tolerance, selection=args.selection)
    selected = select_by_budget(scores, selection=args.selection)
    selected.to_csv(args.outdir / "selected_before_test.csv", index=False)
    selected = evaluate_selected(clean, selected)
    evaluate_selected(clean, scores).to_csv(args.outdir / "front_test_scores.csv", index=False)
    summary = curves(selected)

    checks = check_reference(samples, scores, selected, selection=args.selection, fresh_fronts=getattr(args, "fresh_fronts", False))
    (args.outdir / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    scores.to_csv(args.outdir / "candidate_scores.csv", index=False)
    selected.to_csv(args.outdir / "selected.csv", index=False)
    summary.to_csv(args.outdir / "curves.csv", index=False)
    if not all(row["passed"] for row in checks):
        raise ValueError(
            "Reference checks failed; inspect checks.json and candidate_scores.csv"
        )
    draw_frontier(summary, args.outdir)
    print(
        f"Wrote {args.outdir}; {len(checks)} reference checks passed; "
        f"{int(scores.selectable.sum())}/{len(scores)} candidates selectable"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--galspec", type=Path, required=True)
    parser.add_argument("--specgals-home", type=Path, required=True)
    parser.add_argument("--fronts", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--selection", choices=("train", "validation"), default="train")
    parser.add_argument("--fresh-fronts", action="store_true", help="Check sample integrity without requiring historical search results")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
