"""Shared equation scoring, candidate eligibility and training-default selection.

Used by redshift and conditioning workflows. Data loading, reference recovery,
arm names and complexity-budget ranges remain explicit caller responsibilities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import sympy as sp
from sklearn.metrics import r2_score

from wwdc_repro.shared.equations import evaluate, ExpressionError, EvaluationInputError

def score_expression_details(expression: str, x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    """Return an expression's R² score and a failure description.

    Parameters
    ----------
    expression : str
        Equation from the symbolic regression search.
    x : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Finite real target vector of shape (n_samples,), with at least two rows.

    Returns
    -------
    score : float
        R² on the supplied data, or NaN on a handled evaluation failure.
    failure : str
        Empty on success. Otherwise, nonfinite_prediction, nonfinite_score,
        invalid_input: <reason>, invalid_expression: <reason>, or
        evaluation_error: <exception type>.

    Notes
    -----
    Input, expression, and numerical exceptions handled below are converted
    into (NaN, failure) results rather than propagated to the caller."""
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

def score_expression(expression: str, x: np.ndarray, y: np.ndarray) -> float:
    return score_expression_details(expression, x, y)[0]


def audit_candidate(equation, data, indices, reference_train, *, tolerance=1e-6,
                    check_full_training=False, selection="train"):
    """Audit training-score agreement; optionally gate on validation performance.

    Parameters
    ----------
    equation : str
        Saved expression, after any verified export recovery.
    data : Mapping
        X_train and y_train arrays; validation mode also requires X_val/y_val.
        Test arrays are never accessed.
    indices : np.ndarray
        The original search's training-subsample row indices.
    reference_train : float
        Independent native training score to reproduce.
    tolerance : float
        Maximum permitted absolute training-score difference.
    check_full_training : bool
        Also require finite full-training predictions/scores. True for the
        historical conditioning policy; False for the redshift policy.

    selection : {"train", "validation"}
        Training by default. Validation scores and gates are used only when selected.

    Returns
    -------
    dict
        Training/validation scores, failure reasons, score difference,
        eligibility and (when requested) full-training finiteness.
    """
    if selection not in ('train', 'validation'):
        raise ValueError(f'Unknown selection policy: {selection}')
    train, train_failure = score_expression_details(
        equation, data['X_train'][indices], data['y_train'][indices])
    valid, valid_failure = (score_expression_details(equation, data['X_val'], data['y_val'])
                            if selection == 'validation' else (np.nan, 'not_evaluated'))
    delta = abs(train-reference_train)
    result = dict(train_r2=train, val_r2=valid, train_delta=delta,
                  train_failure_reason=train_failure, validation_failure_reason=valid_failure)
    full_finite = True
    if check_full_training:
        full, failure = score_expression_details(equation, data['X_train'], data['y_train'])
        full_finite = bool(np.isfinite(full))
        result.update(train_full_finite=full_finite, full_training_failure_reason=failure)
    result['selectable_train'] = bool(np.isfinite(train) and np.isfinite(reference_train)
                                      and delta <= tolerance)
    result['selectable_validation'] = bool(result['selectable_train'] and np.isfinite(valid) and full_finite)
    result['selectable'] = result['selectable_train'] if selection == 'train' else result['selectable_validation']
    result['selection'] = selection
    return result


def select_candidate(rows: pd.DataFrame, budget: int, *, length_column='length',
                     validation_column='validation_r2', training_column='train_r2',
                     selection='train') -> pd.Series | None:
    """Select the best eligible equation within a complexity budget.
 
    Choose the highest training R² by default, or validation R² when requested. Ties favour
    shorter equations, then the original row order. Test scores are
    not used.
    
    Parameters
    ----------
    rows : pd.DataFrame
        Candidate records for a single arm and seed. Must contain
        eligibility, length and the score column for the requested policy.
    budget : int
        Maximum permitted equation length.
    length_column : str, optional
        Equation-length column name. Default is "length".
    validation_column : str, optional
        Validation-R² column name. Default is "validation_r2".
    
    training_column : str
        Training-R² column used by the default policy.
    selection : {"train", "validation"}
        Ranking policy. Test scores never affect selection.

    Returns
    -------
    pd.Series or None
        The selected candidate's complete row, or None if no eligible
        candidate has length less than or equal to budget.
    
    Raises
    ------
    KeyError
        If a required column is missing.
    
    Notes
    -----
    Uses policy-specific eligibility flags when available. For historical audits,
    reconstructs the training gate from the recorded training-score difference.
    """
    if selection not in ('train', 'validation'):
        raise ValueError(f'Unknown selection policy: {selection}')
    column = training_column if selection == 'train' else validation_column
    eligible = rows.get(f'selectable_{selection}', rows.selectable)
    # Older audits only stored validation eligibility; reconstruct the training gate.
    if selection == 'train' and 'selectable_train' not in rows and 'train_delta' in rows:
        eligible = np.isfinite(rows[training_column]) & (rows.train_delta <= 1e-6)
    pool = rows[eligible & np.isfinite(rows[column]) & (rows[length_column] <= budget)]
    if pool.empty:
        return None
    return pool.sort_values([column, length_column],
                            ascending=[False, True], kind='stable').iloc[0]


def report_front(audit, data, *, selection="train", budgets=range(5, 30)):
    """Re-audit saved front candidates, freeze budget choices, then score on test.

    Uses the search's seeded training subsample. Existing selected equations and
    test scores in archived tables do not participate in the new choices.
    """
    frozen = []
    for seed, group in audit.groupby('seed', sort=False):
        indices = np.random.default_rng(int(seed)).choice(
            len(data['X_train']), min(10000, len(data['X_train'])), replace=False)
        records = []
        for _, row in group.iterrows():
            checks = audit_candidate(row.equation, data, indices, row.train_r2_stored,
                                     check_full_training=selection == 'validation', selection=selection)
            identity = {key: value for key, value in row.to_dict().items() if 'test' not in key}
            records.append({**identity, **checks})
        scored = pd.DataFrame(records)
        for budget in budgets:
            chosen = select_candidate(scored, budget, length_column='Length',
                                      validation_column='val_r2', selection=selection)
            if chosen is not None:
                frozen.append(dict(chosen.to_dict(), budget=budget))
    result = pd.DataFrame(frozen)
    if result.empty:
        raise ValueError('No eligible equations for the requested selection policy')
    # Choices are fixed for all seeds/budgets before any test evaluation.
    cache = {}
    for index, row in result.iterrows():
        if row.equation not in cache:
            cache[row.equation] = score_expression_details(row.equation, data['X_test'], data['y_test'])
        value, failure = cache[row.equation]
        result.loc[index, 'test_r2'] = value
        result.loc[index, 'test_status'] = 'ok' if np.isfinite(value) else 'evaluation_failure'
        result.loc[index, 'test_failure_reason'] = failure
    return result
