"""Validation selection and held-out reporting for saved symbolic fronts."""

from __future__ import annotations

from pathlib import Path
import hashlib
import importlib.resources
import json

import numpy as np
import pandas as pd
import sympy as sp
from sklearn.metrics import r2_score

from .data import TARGET, matrices
from .equations import evaluate, ExpressionError, EvaluationInputError

ARMS = ("z", "orig", "uncond", "cond", "cond+z", "orig+z")
SEEDS = (42, 43, 44)
TRAIN_SUBSAMPLE = 10_000


def front_path(fronts: Path, arm: str, seed: int, target: str = TARGET) -> Path:
    name = (
        f"job_results_origz_{target}_cond+z_seed{seed}"
        if arm == "orig+z"
        else f"job_results_{target}_{arm}_seed{seed}"
    )
    return fronts / name / "pareto_fronts.csv"


def score_expression(expression: str, x: np.ndarray, y: np.ndarray) -> float:
    """Compatibility interface; detailed failures are available to audit callers."""
    return score_expression_details(expression, x, y)[0]


def score_expression_details(expression: str, x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    try:
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(x) or len(y) < 2 or np.iscomplexobj(y):
            raise EvaluationInputError("Targets must be a real vector matching at least two feature rows")
        if not np.isfinite(y).all():
            raise EvaluationInputError("Targets must be finite")
        prediction = evaluate(expression, x)
        if not np.isfinite(prediction).all():
            return np.nan, "nonfinite_prediction"
        value = float(r2_score(y, prediction))
        return (value, "") if np.isfinite(value) else (np.nan, "nonfinite_score")
    except EvaluationInputError as exc:
        return np.nan, f"invalid_input: {exc}"
    except ExpressionError as exc:
        return np.nan, f"invalid_expression: {exc}"
    except (
        TypeError,
        ValueError,
        KeyError,
        ZeroDivisionError,
        OverflowError,
        sp.SympifyError,
    ) as exc:
        return np.nan, f"evaluation_error: {type(exc).__name__}"


def score_fronts(
    frames: dict[str, pd.DataFrame],
    fronts: Path,
    tolerance: float = 1e-6,
    target: str = TARGET,
) -> pd.DataFrame:
    """Establish eligibility on training/validation only; never evaluate test here.

    Historical nonfinite training scores are ambiguous: the exporter also set
    them after test failures. Recover those references from verified native reruns.
    """
    recovered = json.loads(importlib.resources.files("wwdc_repro").joinpath(
        "recovered_training_scores.json").read_text())
    recovery = {(r["arm"], r["seed"], r["length"]): r for r in recovered["candidates"]}
    records = []
    for arm in ARMS:
        data = matrices(frames, arm, target)
        for seed in SEEDS:
            path = front_path(fronts, arm, seed, target)
            raw = pd.read_csv(path)
            front_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if set(raw.Seed.astype(int)) != {seed}:
                raise ValueError(f"Unexpected seed in {path}")
            rng = np.random.default_rng(seed=seed)
            idx = rng.choice(
                len(data["X_train"]),
                size=min(TRAIN_SUBSAMPLE, len(data["X_train"])),
                replace=False,
            )
            for row in raw.itertuples():
                train, train_failure = score_expression_details(
                    row.Equation, data["X_train"][idx], data["y_train"][idx]
                )
                valid, validation_failure = score_expression_details(row.Equation, data["X_val"], data["y_val"])
                reference_train = float(row.Train_R2)
                reference_source = "archived independent finite training score"
                if not np.isfinite(reference_train):
                    key = (arm, seed, int(row.Length))
                    if key not in recovery:
                        raise ValueError(f"Missing independent training reference for {key}")
                    recovered_row = recovery[key]
                    if recovered_row["front_sha256"] != front_hash or recovered_row["equation"] != row.Equation:
                        raise ValueError(f"Recovered training reference does not match {key}")
                    reference_train = (float(recovered_row["native_train_r2"])
                                       if recovered_row["native_train_r2"] is not None else np.nan)
                    reference_source = "verified native seeded rerun"
                records.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "length": int(row.Length),
                        "equation": row.Equation,
                        "source": str(path),
                        "stored_train_r2": float(row.Train_R2),
                        "recomputed_train_r2": train,
                        "validation_r2": valid,
                        "train_failure_reason": train_failure,
                        "validation_failure_reason": validation_failure,
                        "reference_train_r2": reference_train,
                        "training_reference_source": reference_source,
                        "train_delta": abs(train - reference_train),
                        "selectable": bool(
                            np.isfinite(train)
                            and np.isfinite(reference_train)
                            and abs(train - reference_train) <= tolerance
                            and np.isfinite(valid)
                        ),
                    }
                )
    scores = pd.DataFrame(records)
    if (
        not np.isfinite(scores.train_delta.median())
        or scores.train_delta.median() > tolerance
    ):
        raise ValueError(
            "Systematic training-score mismatch; inspect preprocessing and expression parsing"
        )
    return scores


def select_by_budget(scores: pd.DataFrame) -> pd.DataFrame:
    """Choose on validation separately by arm, seed and maximum equation length."""
    rows = []
    for arm in ARMS:
        arm_scores = scores[scores.arm == arm]
        if arm_scores.empty:
            continue
        for seed in SEEDS:
            group = arm_scores[arm_scores.seed == seed]
            for budget in range(
                int(arm_scores.length.min()), int(arm_scores.length.max()) + 1
            ):
                pool = group[group.selectable & (group.length <= budget)]
                if pool.empty:
                    continue
                chosen = pool.sort_values(
                    ["validation_r2", "length"], ascending=[False, True]
                ).iloc[0]
                rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "budget": budget,
                        "chosen_length": int(chosen.length),
                        "equation": chosen.equation,
                        "validation_r2": chosen.validation_r2,
                        "eligible_count": len(pool),
                        "excluded_count_to_budget": int(
                            ((group.length <= budget) & ~group.selectable).sum()
                        ),
                        "source": chosen.source,
                    }
                )
    return pd.DataFrame(rows)


def evaluate_selected(frames: dict[str, pd.DataFrame], selected: pd.DataFrame,
                      target: str = TARGET) -> pd.DataFrame:
    """Evaluate frozen choices; a failed test prediction never triggers replacement."""
    evaluated = selected.copy()
    evaluated["test_r2"] = np.nan
    evaluated["test_status"] = "evaluation_failure"
    evaluated["test_failure_reason"] = ""
    for arm, group in evaluated.groupby("arm"):
        data = matrices(frames, arm, target)
        cache = {}
        for index, row in group.iterrows():
            if row.equation not in cache:
                cache[row.equation] = score_expression_details(row.equation, data["X_test"], data["y_test"])
            value, failure = cache[row.equation]
            evaluated.loc[index, "test_r2"] = value
            evaluated.loc[index, "test_status"] = "ok" if np.isfinite(value) else "evaluation_failure"
            evaluated.loc[index, "test_failure_reason"] = failure
    return evaluated


def curves(selected: pd.DataFrame) -> pd.DataFrame:
    summary = selected.groupby(["arm", "budget"]).test_r2.agg(
        mean="mean", low="min", high="max", n="size", n_finite="count").reset_index()
    summary["evaluation_failures"] = summary.n - summary.n_finite
    # Do not silently average only successful seeds at a failed budget.
    summary.loc[summary.evaluation_failures > 0, ["mean", "low", "high"]] = np.nan
    return summary
