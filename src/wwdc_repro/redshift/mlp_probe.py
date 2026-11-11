"""Reproduce the matched orig+z and cond+z stellar-mass MLP probes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from wwdc_repro.shared.data import TARGET, clean_target, load_sample, matrices

ARMS = ("orig+z", "cond+z")
SEEDS = (42, 43, 44)


def r2(y: np.ndarray, predicted: np.ndarray) -> float:
    """Use the historical probe's test R² calculation."""
    return float(1.0 - np.mean((y - predicted) ** 2) / np.var(y))


def run(args: argparse.Namespace) -> None:
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise ValueError(f"Output directory must be new or empty: {args.outdir}")
    frames = clean_target(load_sample(args.embeddings, args.galspec, args.specgals_home))
    training_ids = frames["train"].specObjID.to_numpy(dtype="<i8")
    test_ids = frames["test"].specObjID.to_numpy(dtype="<i8")
    rows = []
    for arm in ARMS:
        data = matrices(frames, arm)
        # The notebook's final fit_model_fn scales and fits on the full train split.
        scaler = StandardScaler().fit(data["X_train"])
        x_fit = scaler.transform(data["X_train"])
        y_fit = data["y_train"]
        x_test = scaler.transform(data["X_test"])
        for seed in SEEDS:
            model = MLPRegressor(
                hidden_layer_sizes=(64, 32),
                early_stopping=False,
                max_iter=100,
                random_state=seed,
            )
            model.fit(x_fit, y_fit)
            rows.append({
                "arm": arm,
                "seed": seed,
                "n_train": len(training_ids),
                "n_test": len(test_ids),
                "test_r2": r2(data["y_test"], model.predict(x_test)),
                "n_iter": model.n_iter_,
            })

    scores = pd.DataFrame(rows)
    summary = scores.groupby("arm", sort=False).test_r2.agg(["mean", "min", "max"])
    summary["seed_range"] = summary["max"] - summary["min"]
    paired = scores.pivot(index="seed", columns="arm", values="test_r2")
    paired["orig_minus_cond"] = paired["orig+z"] - paired["cond+z"]
    gap = float(summary.loc["orig+z", "mean"] - summary.loc["cond+z", "mean"])

    args.outdir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.outdir / "scores.csv", index=False)
    summary.to_csv(args.outdir / "summary.csv")
    paired.to_csv(args.outdir / "paired.csv")
    metadata = {
        "target": TARGET,
        "arms": list(ARMS),
        "features": "10 train-standardized latents plus redshift; full-train StandardScaler on assembled inputs",
        "model": "sklearn.neural_network.MLPRegressor",
        "hidden_layer_sizes": [64, 32],
        "max_iter": 100,
        "early_stopping": False,
        "other_model_options": "scikit-learn defaults",
        "seeds": list(SEEDS),
        "training_draw": "full cleaned training split, as in the notebook's final fit_model_fn",
        "train_rows_clean": len(training_ids),
        "train_rows_selected": len(training_ids),
        "test_rows": len(test_ids),
        "selected_train_ids_sha256": hashlib.sha256(training_ids.tobytes()).hexdigest(),
        "test_ids_sha256": hashlib.sha256(test_ids.tobytes()).hexdigest(),
        "scikit_learn_version": importlib.metadata.version("scikit-learn"),
        "mean_test_r2_orig_minus_cond": gap,
        "inputs": {
            "embeddings": str(args.embeddings.resolve()),
            "galspec": str(args.galspec.resolve()),
            "specgals_home": str(args.specgals_home.resolve()),
        },
    }
    (args.outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string())
    print(f"orig+z minus cond+z mean test R²: {gap:.6f}")
    print(f"Wrote {args.outdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--galspec", type=Path, required=True)
    parser.add_argument("--specgals-home", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
