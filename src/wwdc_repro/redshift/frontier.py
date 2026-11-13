"""Training-default selection and held-out reporting for saved symbolic fronts."""

from __future__ import annotations

from pathlib import Path
import importlib.resources
import json

import numpy as np
import pandas as pd

from wwdc_repro.shared.utils import sha256
from wwdc_repro.shared.data import TARGET, matrices
from wwdc_repro.shared.selection import score_expression, score_expression_details, audit_candidate, select_candidate

ARMS = ("z", "orig", "uncond", "cond", "cond+z", "orig+z")
SEEDS = (42, 43, 44)
TRAIN_SUBSAMPLE = 10_000


def front_path(fronts: Path, arm: str, seed: int, target: str = TARGET) -> Path:
    """Return the saved Pareto-front path.
    
    Parameters
    ----------
    fronts: Path
        Root directory containing the saved Pareto fronts. 
    arm: str 
        Feature representation used in the fit, such as cond+z or orig.
    seed: int
        Value of the random seed 
    target: str 
        Regression target. Default is lgm_tot_p50
    
    Returns
    ------- 
    pareto_front_file_path: Path 
        Path to the Pareto front
    """
    name = f"job_results_{target}_{arm}_seed{seed}"
    return fronts / name / "pareto_fronts.csv"

def score_fronts(
    frames: dict[str, pd.DataFrame],
    fronts: Path,
    tolerance: float = 1e-6,
    target: str = TARGET,
    *, selection: str = "train",
) -> pd.DataFrame:
    """Calculate training/validation scores and eligibility for all arms and seeds.

    Training scores use up to TRAIN_SUBSAMPLE rows sampled without replacement
    using each search seed. Validation mode also scores the full validation split.
    Test scores are not evaluated here; see evaluate_selected below.

    Parameters
    ----------
    frames : dict[str, pd.DataFrame]
        Cleaned train, val, and test DataFrames used to construct feature arrays.
    fronts : Path
        Root directory containing the saved front directory for each arm/seed.
    tolerance : float
        Maximum absolute difference between recalculated and reference training
        score for a candidate to be eligible. Defaults to 1e-6.
    target : str
        Column to predict. Defaults to lgm_tot_p50.

    Returns
    -------
    scores : pd.DataFrame
        One row per candidate, including training and validation score, reference
        provenance, failure reasons, and eligibility. Eligibility requires finite
        training/reference scores and training agreement within tolerance. Validation
        mode additionally requires a finite validation score.

    Raises
    ------
    ValueError
        If a front contains an unexpected seed, a required recovery record is
        missing or does not match the front hash/equation, or the median training
        score difference is nonfinite or exceeds tolerance. Data preprocessing
        can also raise ValueError for invalid inputs.

    Notes
    -----
    The bundled recovery JSON supplies independent training-score references
    where archived Train_R2 values are nonfinite; it does not supply predictions.
    Fresh fronts supply their own native training scores and precise equations;
    the archive recovery JSON is loaded only if a training score is nonfinite.
    """
    recovery = None
    records = []
    for arm in ARMS:
        data = matrices(frames, arm, target)
        for seed in SEEDS:
            path = front_path(fronts, arm, seed, target)
            raw = pd.read_csv(path)
            front_hash = sha256(path)
            if set(raw.Seed.astype(int)) != {seed}:
                raise ValueError(f"Unexpected seed in {path}") # Require every row to match the requested seed.
            rng = np.random.default_rng(seed=seed) # Reproduce the training subsample for this search seed.
            idx = rng.choice(
                len(data["X_train"]),
                size=min(TRAIN_SUBSAMPLE, len(data["X_train"])),
                replace=False,
            ) # Sample up to 10,000 distinct training-row indices.
            for row in raw.itertuples():
                reference_train = float(row.Train_R2)
                reference_source = "saved native training score"
                if not np.isfinite(reference_train):
                    if recovery is None:
                        recovered = json.loads(importlib.resources.files(
                            "wwdc_repro.resources").joinpath(
                            "recovered_training_scores.json").read_text())
                        recovery = {(r["arm"], r["seed"], r["length"]): r
                                    for r in recovered["candidates"]}
                    key = (arm, seed, int(row.Length))
                    if key not in recovery:
                        raise ValueError(f"Missing independent training reference for {key}")
                    recovered_row = recovery[key]
                    if recovered_row["front_sha256"] != front_hash or recovered_row["equation"] != row.Equation:
                        raise ValueError(f"Recovered training reference does not match {key}")
                    reference_train = (float(recovered_row["native_train_r2"])
                                       if recovered_row["native_train_r2"] is not None else np.nan)
                    reference_source = "verified native seeded rerun"
                equation = getattr(row, "Equation_p60", row.Equation)
                audit = audit_candidate(equation, data, idx, reference_train, tolerance=tolerance, selection=selection)
                records.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "length": int(row.Length),
                        "equation": equation,
                        "source": str(path),
                        "stored_train_r2": float(row.Train_R2),
                        "recomputed_train_r2": audit["train_r2"],
                        "validation_r2": audit["val_r2"],
                        "train_failure_reason": audit["train_failure_reason"],
                        "validation_failure_reason": audit["validation_failure_reason"],
                        "reference_train_r2": reference_train,
                        "training_reference_source": reference_source,
                        "train_delta": audit["train_delta"],
                        "selectable": audit["selectable"],
                        "selection": selection,
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


def select_by_budget(scores: pd.DataFrame, selection: str = "train") -> pd.DataFrame:
    """Choose the highest requested selection score within each arm, seed, and length budget.

    Parameters
    ----------
    scores : pd.DataFrame
        Candidate scores returned by score_fronts. Selection uses arm, seed,
        length, selectable, the requested score, equation, and source columns.

    Returns
    -------
    selected : pd.DataFrame
        One chosen candidate for each arm/seed/budget with an eligible equation.
        Candidates must be selectable and have length <= budget. Ties in
        the selection score favour shorter equations. Budgets span every integer from
        the minimum to maximum candidate length for the arm, across its seeds.
        Each row includes the chosen equation and length, validation score, source,
        and eligible/excluded candidate counts. The same equation can be selected
        at multiple budgets; budgets with no eligible candidate are omitted."""
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
                chosen = select_candidate(group, budget, training_column="recomputed_train_r2", selection=selection)
                rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "budget": budget,
                        "chosen_length": int(chosen.length),
                        "equation": chosen.equation,
                        "validation_r2": chosen.validation_r2,
                        "train_r2": chosen.get("recomputed_train_r2", np.nan),
                        "selection": selection,
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
    """Evaluate training- or validation-selected equations on the test set.

    Parameters
    ----------
    frames : dict[str, pd.DataFrame]
        Cleaned train, val, and test DataFrames used to construct feature arrays.
    selected : pd.DataFrame
        Selected equations returned by select_by_budget.
    target : str
        Column to predict. Defaults to lgm_tot_p50.

    Returns
    -------
    evaluated : pd.DataFrame
        Copy of selected with test_r2, test_status, and test_failure_reason added.
        Failed evaluations retain their selected rows with NaN scores and a
        failure reason. Repeated equations are evaluated once per arm and their scores reused.
    """
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
    """Summarize test score across selected seeds for each arm and length budget.

    Parameters
    ----------
    selected : pd.DataFrame
        Output from evaluate_selected, where failed test scores are NaN.

    Returns
    -------
    summary : pd.DataFrame
        One row per arm/budget with mean, low (minimum), high (maximum), n
        (selected row count), n_finite (non-missing score count), and
        evaluation_failures. If any selected score is missing, mean/low/high
        are all set to NaN. Different seeds may select different equations.
        These counts cover only rows present in selected; an omitted seed is
        not counted as an evaluation failure. 
        """
    summary = selected.groupby(["arm", "budget"]).test_r2.agg(
        mean="mean", low="min", high="max", n="size", n_finite="count").reset_index()
    if "selection" in selected:
        summary["selection"] = selected.selection.iloc[0]
    summary["evaluation_failures"] = summary.n - summary.n_finite
    # Do not silently average only successful seeds at a failed budget.
    summary.loc[summary.evaluation_failures > 0, ["mean", "low", "high"]] = np.nan
    return summary
