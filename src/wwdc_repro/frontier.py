"""Validation selection and held-out reporting for saved symbolic fronts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
from sklearn.metrics import r2_score

from .data import TARGET, matrices
from .equations import evaluate

ARMS = ("z", "orig", "uncond", "cond", "cond+z", "orig+z")
SEEDS = (42, 43, 44)
TRAIN_SUBSAMPLE = 10_000


def front_path(fronts: Path, arm: str, seed: int, target: str = TARGET) -> Path:
    name = (f"job_results_origz_{target}_cond+z_seed{seed}" if arm == "orig+z"
            else f"job_results_{target}_{arm}_seed{seed}")
    return fronts / name / "pareto_fronts.csv"


def score_expression(expression: str, x: np.ndarray, y: np.ndarray) -> float:
    try:
        prediction = evaluate(expression, x)
        return float(r2_score(y, prediction)) if np.isfinite(prediction).all() else np.nan
    except (TypeError, ValueError, KeyError, ZeroDivisionError, OverflowError, sp.SympifyError):
        return np.nan


def score_fronts(frames: dict[str, pd.DataFrame], fronts: Path, tolerance: float = 1e-6,
                 target: str = TARGET) -> pd.DataFrame:
    """Reject printed equations that do not reproduce saved training R²."""
    records = []
    for arm in ARMS:
        data = matrices(frames, arm, target)
        for seed in SEEDS:
            path = front_path(fronts, arm, seed, target)
            raw = pd.read_csv(path)
            if set(raw.Seed.astype(int)) != {seed}:
                raise ValueError(f"Unexpected seed in {path}")
            rng = np.random.default_rng(seed=seed)
            idx = rng.choice(len(data["X_train"]), size=min(TRAIN_SUBSAMPLE, len(data["X_train"])), replace=False)
            for row in raw.itertuples():
                train = score_expression(row.Equation, data["X_train"][idx], data["y_train"][idx])
                valid = score_expression(row.Equation, data["X_val"], data["y_val"])
                test = score_expression(row.Equation, data["X_test"], data["y_test"])
                records.append({
                    "arm": arm, "seed": seed, "length": int(row.Length),
                    "equation": row.Equation, "source": str(path),
                    "stored_train_r2": float(row.Train_R2), "recomputed_train_r2": train,
                    "validation_r2": valid, "stored_test_r2": float(row.Test_R2),
                    "recomputed_test_r2": test,
                    "train_delta": abs(train - float(row.Train_R2)),
                    "test_delta": abs(test - float(row.Test_R2)),
                    "selectable": bool(np.isfinite(train) and abs(train - float(row.Train_R2)) <= tolerance
                                       and np.isfinite(valid)),
                })
    scores = pd.DataFrame(records)
    if not np.isfinite(scores.train_delta.median()) or scores.train_delta.median() > tolerance:
        raise ValueError("Systematic training-score mismatch; inspect preprocessing and expression parsing")
    return scores


def select_by_budget(scores: pd.DataFrame) -> pd.DataFrame:
    """Choose on validation separately by arm, seed and maximum equation length."""
    rows = []
    for arm in ARMS:
        arm_scores = scores[scores.arm == arm]
        for seed in SEEDS:
            group = arm_scores[arm_scores.seed == seed]
            for budget in range(int(arm_scores.length.min()), int(arm_scores.length.max()) + 1):
                pool = group[group.selectable & (group.length <= budget)]
                if pool.empty:
                    continue
                chosen = pool.sort_values(["validation_r2", "length"], ascending=[False, True]).iloc[0]
                rows.append({
                    "arm": arm, "seed": seed, "budget": budget,
                    "chosen_length": int(chosen.length), "equation": chosen.equation,
                    "validation_r2": chosen.validation_r2, "test_r2": chosen.stored_test_r2,
                    "eligible_count": len(pool),
                    "excluded_count_to_budget": int(((group.length <= budget) & ~group.selectable).sum()),
                    "source": chosen.source,
                })
    return pd.DataFrame(rows)


def curves(selected: pd.DataFrame) -> pd.DataFrame:
    return (selected.groupby(["arm", "budget"]).test_r2
            .agg(mean="mean", low="min", high="max", n="count").reset_index())
