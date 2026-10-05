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

from .data import TARGET, clean_target, load_sample, sample_summary, preprocessing_state
from .frontier import ARMS, SEEDS, curves, front_path, score_fronts, select_by_budget

COLOURS = {
    "z": "#777777",
    "orig": "#9554a4",
    "uncond": "#527ab6",
    "cond": "#ce8154",
    "cond+z": "#348c75",
    "orig+z": "#a9516c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest(args: argparse.Namespace) -> dict:
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
        "selection": "highest validation R2 for each arm, seed and length budget among printed equations reproducing training R2 within tolerance",
        "tolerance_train_r2": args.tolerance,
        "code": [
            {"file": p.name, "sha256": sha256(p)}
            for p in sorted(Path(__file__).parent.glob("*.py"))
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
    samples: pd.DataFrame, scores: pd.DataFrame, selected: pd.DataFrame
) -> list[dict]:
    """Compare independent audit values with this run; fail on any mismatch."""
    reference = json.loads(
        importlib.resources.files("wwdc_repro").joinpath("reference.json").read_text()
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
    add("candidate_count", len(scores), reference["candidate_count"])
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
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise ValueError(f"Output directory must be new or empty: {args.outdir}")
    record = manifest(args)
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    raw = load_sample(args.embeddings, args.galspec, args.specgals_home)
    clean = clean_target(raw)
    (args.outdir / "preprocessing.json").write_text(
        json.dumps(preprocessing_state(clean), indent=2) + "\n"
    )
    pd.concat(
        [frame[["specObjID"]].assign(split=split) for split, frame in clean.items()],
        ignore_index=True,
    ).to_csv(args.outdir / "sample_ids.csv.gz", index=False)
    samples = pd.DataFrame(sample_summary(clean))
    samples.to_csv(args.outdir / "samples.csv", index=False)
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
    scores = score_fronts(clean, args.fronts, args.tolerance)
    selected = select_by_budget(scores)
    summary = curves(selected)
    checks = check_reference(samples, scores, selected)
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
    parser.add_argument("--tolerance", type=float, default=1e-6)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
