# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "pyoperon==0.4.0", "numpy==2.4.4", "pandas==3.0.2",
#   "scikit-learn==1.8.0", "scipy==1.17.1", "sympy==1.14.0",
#   "astropy==7.2.0", "astroML==1.0.2.post1", "pyarrow==24.0.0",
#   "matplotlib==3.10.9", "tqdm==4.67.3",
# ]
# ///
"""Rerun the historical seeded search and compare freshly scored fronts.

Run using uv run --script --locked experiments/rerun_sr.py. The script's
separate lockfile pins its search environment. This audit deliberately calls
the original fit_sym_fn instead of copying its settings. Comparison scores
are historical diagnostics, not the current production selection workflow.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

REPRO = Path(__file__).resolve().parents[1]
# Byte-identical copies of the flower workspace's examples/spectra/symbolic_regression
# modules that produced the archived fronts; the hashes pin that version.
HISTORICAL = REPRO / "experiments/historical"
HISTORICAL_SHA256 = {
    "utils.py": "9ccbe597ea6b604d1eabcad03b79238b0749486db06db0e634e01679ba0a80f0",
    "processing_fun.py": "7a00ac4012d309c170a6a408976227a83ebbb31afb20f6716b4bec9dabf73562",
}
for name, expected in HISTORICAL_SHA256.items():
    if hashlib.sha256((HISTORICAL / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"{HISTORICAL / name} differs from the historical search code")
sys.path[:0] = [str(REPRO / "src"), str(HISTORICAL)]
from wwdc_repro.data import load_sample, clean_target, matrices, sample_summary, TARGET
from wwdc_repro.frontier import front_path, score_expression
from utils import fit_sym_fn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="cond+z", choices=["cond+z", "cond", "orig", "orig+z", "uncond", "z"])
    parser.add_argument("--seed", type=int, default=42, choices=[42, 43, 44])
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Root containing the archived embeddings, sdss catalogues and galSpecExtra-dr8.fits; see docs/INPUTS.md")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    embeddings = args.data_root / "sdss_II/spender_I_flow_v2/spender_I_flow_v2/embeddings/7655991_0"
    input_files = [args.data_root / "galSpecExtra-dr8.fits",
                   *(args.data_root / "sdss" / f"SDSSspecgalsDR8_{i}.fit.gz" for i in (1, 2))]
    for split in ("train", "val", "test"):
        shards = sorted((embeddings / split).glob("*.parquet"))
        if not shards:
            parser.error(f"No archived embedding shards in {embeddings / split}")
        input_files.extend(shards)
    for path in input_files:
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    out = args.outdir
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    source = HISTORICAL / "utils.py"
    manifest = {
        "arm": args.arm, "seed": args.seed, "python": sys.version,
        "platform": platform.platform(), "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "historical_sha256": HISTORICAL_SHA256,
        "versions": {p: importlib.metadata.version(p) for p in
                     ["pyoperon", "numpy", "pandas", "scikit-learn", "scipy",
                      "sympy", "astropy", "astroML", "pyarrow", "matplotlib", "tqdm"]},
        "data_root": str(args.data_root.resolve()),
        "comparison_policy": "Historical diagnostic comparisons only; use wwdc-reproduce for current validation-only selection of archived exported equations.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "script_lock_sha256": hashlib.sha256(Path(__file__).with_suffix(".py.lock").read_bytes()).hexdigest(),
    }
    def file_hash(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    manifest["inputs"] = [{"path": str(path.resolve()), "bytes": path.stat().st_size,
                           "sha256": file_hash(path)} for path in input_files]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Loading and checking the historical sample", flush=True)
    frames = clean_target(load_sample(
        embeddings,
        args.data_root / "galSpecExtra-dr8.fits", args.data_root / "sdss"))
    reference = json.loads((REPRO / "src/wwdc_repro/reference.json").read_text())
    for row in sample_summary(frames):
        for key in ("rows", "id_sha256", "ordered_id_sha256"):
            assert row[key] == reference["samples"][row["split"]][key]
    data = matrices(frames, args.arm)
    idx = np.random.default_rng(args.seed).choice(len(frames["train"]), 10000, replace=False)
    data["X_train"] = data["X_train"][idx]
    data["y_train"] = data["y_train"][idx]
    frames["train"].iloc[idx][["specObjID"]].to_csv(out / "train_ids.csv", index=False)
    manifest["train_ids_sha256"] = file_hash(out / "train_ids.csv")
    archived = front_path(REPRO / "artifacts/fronts", args.arm, args.seed)
    manifest["archived_front_sha256"] = file_hash(archived)
    old = pd.read_csv(archived)
    old_scores = []
    for row in old.itertuples():
        scores = {f"{split}_r2": score_expression(row.Equation, data[f"X_{split}"], data[f"y_{split}"])
                  for split in ("train", "val", "test")}
        old_scores.append(dict(length=row.Length, equation=row.Equation,
            stored_train_r2=row.Train_R2, stored_test_r2=row.Test_R2, **scores,
            selectable=bool(abs(scores["train_r2"] - row.Train_R2) <= 1e-6 and np.isfinite(scores["val_r2"]))))
    old_scores = pd.DataFrame(old_scores)
    old_scores.to_csv(out / "original_rescored.csv", index=False)
    assert np.nanmedian(abs(old_scores.train_r2 - old_scores.stored_train_r2)) < 1e-6
    print("Sample checks passed; starting original fit_sym_fn", flush=True)
    fit_frames = frames
    fit_arm = args.arm
    if args.arm == "orig+z":
        fit_frames = {split: frame.assign(cond=frame["orig"]) for split, frame in frames.items()}
        fit_arm = "cond+z"
    result = fit_sym_fn(TARGET, fit_arm, fit_frames, args.seed)
    for key in ("X_train", "y_train", "X_test", "y_test"):
        np.testing.assert_array_equal(result[key], data[key])
    model = result["Model"]
    manifest["parameters"] = model.get_params()
    manifest["elapsed_seconds"] = time.time() - started
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    rows = []
    for candidate in model.pareto_front_:
        tree = candidate["tree"]
        row = dict(length=int(tree.Length), equation=candidate["model"],
                   equation_p12=model.get_model_string(tree, precision=12))
        for split in ("train", "val", "test"):
            pred = model.evaluate_model(tree, data[f"X_{split}"])
            row[f"{split}_r2"] = float(r2_score(data[f"y_{split}"], pred)) if np.isfinite(pred).all() else np.nan
        printed_train = score_expression(row["equation"], data["X_train"], data["y_train"])
        row["printed_train_r2"] = printed_train
        row["selectable"] = bool(abs(printed_train - row["train_r2"]) <= 1e-6 and np.isfinite(row["val_r2"]))
        rows.append(row)
    new = pd.DataFrame(rows)
    new.to_csv(out / "rerun_front.csv", index=False)
    checks = {
        "ordered_lengths_match": old.Length.tolist() == new.length.tolist(),
        "ordered_equations_match": old.Equation.tolist() == new.equation.tolist(),
        "ordered_equations_p12_match": old.Equation_p12.tolist() == new.equation_p12.tolist(),
        "sample_reference_checks_passed": True,
        "fit_input_arrays_match": True,
    }
    checks["all_passed"] = all(checks.values())
    (out / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    comparisons = []
    for budget in (24, 29):
        entry = {"budget": budget}
        for label, front in (("original", old_scores), ("rerun", new)):
            eligible = front.loc[front.selectable & (front.length <= budget)]
            winner = eligible.sort_values(["val_r2", "length"], ascending=[False, True]).iloc[0]
            entry[label] = winner.to_dict()
        entry["test_r2_change"] = entry["rerun"]["test_r2"] - entry["original"]["test_r2"]
        comparisons.append(entry)
    summary = {"comparisons": comparisons, "original_candidates": len(old),
               "rerun_candidates": len(new), "exact_equation_matches": len(set(old.Equation) & set(new.equation)),
               "elapsed_seconds": time.time() - started}
    (out / "comparison.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if not checks["all_passed"]:
        raise SystemExit("Search did not reproduce the archived front exactly; inspect checks.json and rerun_front.csv")


if __name__ == "__main__":
    main()
