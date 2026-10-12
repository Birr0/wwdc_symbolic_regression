"""Verify seeded reruns and recover training references hidden by test flags.

Run after rerun_sr.py for orig seeds 42/43 and orig+z seeds 42/43/44.
Original fronts are immutable. Every recovery is bound to its front hash and
equation; the whole rerun front must match at both saved string precisions.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO / "src"))
from wwdc_repro.frontier import ARMS, SEEDS, front_path


def main():
    candidates, runs = [], []
    for arm in ARMS:
        for seed in SEEDS:
            path = front_path(REPRO / "artifacts/fronts", arm, seed)
            old = pd.read_csv(path)
            bad = ~np.isfinite(old.Train_R2)
            if not bad.any():
                continue
            slug = arm.replace("+", "")
            directory = REPRO / f"outputs/sr-rerun-{slug}-seed{seed}"
            fresh_path = directory / "rerun_front.csv"
            fresh = pd.read_csv(fresh_path)
            assert old.Length.tolist() == fresh.length.tolist(), (arm, seed, "lengths")
            assert old.Equation.tolist() == fresh.equation.tolist(), (arm, seed, "expressions")
            assert old.Equation_p12.tolist() == fresh.equation_p12.tolist(), (arm, seed, "precision 12")
            np.testing.assert_allclose(old.loc[~bad, "Train_R2"], fresh.loc[~bad, "train_r2"], atol=1e-6, rtol=0)
            manifest = json.loads((directory / "manifest.json").read_text())
            runs.append({"arm": arm, "seed": seed, "all_expressions_match": True,
                         "candidate_count": len(old), "manifest": manifest,
                         "rerun_front_sha256": hashlib.sha256(fresh_path.read_bytes()).hexdigest(),
                         "train_ids_sha256": hashlib.sha256((directory / "train_ids.csv").read_bytes()).hexdigest()})
            for index in old.index[bad]:
                native = float(fresh.loc[index, "train_r2"])
                candidates.append({"arm": arm, "seed": seed, "length": int(old.loc[index, "Length"]),
                    "equation": old.loc[index, "Equation"],
                    "front_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "native_train_r2": native if np.isfinite(native) else None})
    result = {"description": "Native training-only scores recovered from exact seeded reruns; no test metrics used for eligibility.",
              "candidates": candidates, "verified_runs": runs}
    output = REPRO / "src/wwdc_repro/recovered_training_scores.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Recovered {len(candidates)} references from {len(runs)} verified runs: {output}")


if __name__ == "__main__":
    main()
