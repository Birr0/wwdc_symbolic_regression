"""Scikit-learn MLP training with a separate, fixed validation sample."""

from copy import deepcopy
from pathlib import Path
import json
import time

import joblib
import sklearn

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from wwdc_repro.shared.utils import sha256

MLP_PARAMETERS = dict(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=0,
    batch_size=1024,
    learning_rate_init=0.001,
)


def scaled_asinh(values, *, scale):
    """Apply the saved training flux scale before the inverse hyperbolic sine."""
    return np.arcsinh(np.asarray(values) / scale)


def fit_mlp(x_train, y_train, x_val, y_val, *, seed, max_epochs=500, patience=40,
            model_parameters=None, asinh=False, threads=2, train_indices=None):
    """Fit a small MLP and retain the minimum-validation-MSE checkpoint.

    Parameters
    ----------
    x_train, x_val : numpy.ndarray
        Full training and existing validation feature matrices. Preprocessing
        is fitted on all supplied training rows, before optional subsampling.
    y_train, y_val : numpy.ndarray
        Corresponding one-dimensional target arrays.
    seed : int
        Seed for initialization and batch shuffling.
    max_epochs : int, optional
        Maximum number of passes over the training sample.
    patience : int, optional
        Stop after this many epochs without a lower validation MSE.

    model_parameters : dict, optional
        Overrides for MLP_PARAMETERS (hidden sizes, activation, solver, alpha,
        batch size and initial learning rate). Existing callers retain defaults.
    asinh : bool, optional
        For spectra, apply arcsinh(x / q) before standardisation, with q the
        median absolute training flux across all rows and pixels. The fitted
        pipeline includes this transformation for prediction on raw inputs.
    train_indices : array-like of int or None, optional
        Distinct row indices used for model fitting after preprocessing has
        been fitted on the full training split. None fits the model on all
        supplied training rows, preserving the behaviour of existing callers.
    threads : int, optional
        Numerical thread limit during fitting; default is 2.

    Returns
    -------
    checkpoint : dict
        Fitted input-scaling/model pipeline, target scaler, best epoch and
        epochs run. Scalers use only the supplied training rows. The model
        predicts standardized targets; use predict_mlp to recover target units.
    history : pandas.DataFrame
        Training and validation MSE in squared target units for each epoch.
    """
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be positive")
    parameters = dict(MLP_PARAMETERS)
    if model_parameters:
        unknown = set(model_parameters) - parameters.keys()
        if unknown:
            raise ValueError(f"Unknown MLP parameters: {sorted(unknown)}")
        parameters.update(model_parameters)
    if train_indices is None:
        indices = np.arange(len(x_train))
    else:
        indices = np.asarray(train_indices)
        if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                or len(indices) == 0 or len(np.unique(indices)) != len(indices)
                or np.any(indices < 0) or np.any(indices >= len(x_train))):
            raise ValueError("Invalid training row indices")
    if len(x_train) != len(y_train):
        raise ValueError("Training features and targets must have matching rows")
    steps = []
    if asinh:
        q = float(np.median(np.abs(x_train)))
        if not np.isfinite(q) or q <= 0:
            raise ValueError("Invalid training asinh scale")
        steps.append(("asinh", FunctionTransformer(scaled_asinh, kw_args={"scale": q})))
    steps.append(("scaler", StandardScaler()))
    preprocessing = Pipeline(steps)
    preprocessing.fit(x_train)
    xt = preprocessing.transform(np.asarray(x_train)[indices])
    xv = preprocessing.transform(x_val)
    if not np.isfinite(y_train).all() or np.std(y_train) <= 0:
        raise ValueError("Invalid training target scale")
    target_scaler = StandardScaler().fit(np.asarray(y_train).reshape(-1, 1))
    yt = target_scaler.transform(np.asarray(y_train)[indices].reshape(-1, 1)).ravel()
    yv = target_scaler.transform(np.asarray(y_val).reshape(-1, 1)).ravel()
    model = MLPRegressor(
        **parameters,
        early_stopping=False,
        shuffle=True,
        random_state=np.random.RandomState(seed),
        tol=0,
        n_iter_no_change=max_epochs + 1,
    )
    best_loss, best_epoch, best_model = np.inf, 0, None
    history = []
    with threadpool_limits(limits=threads):
        for epoch in range(1, max_epochs + 1):
            model.partial_fit(xt, yt)
            val_loss = mean_squared_error(yv, model.predict(xv))
            train_loss = mean_squared_error(yt, model.predict(xt))
            history.append(
                dict(
                    epoch=epoch,
                    train_mse_dex=train_loss * target_scaler.scale_[0] ** 2,
                    val_mse_dex=val_loss * target_scaler.scale_[0] ** 2,
                )
            )
            if np.isfinite(val_loss) and val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch
                best_model = deepcopy(model)
            if epoch - best_epoch >= patience:
                break
    if best_model is None:
        raise RuntimeError("No finite validation checkpoint")
    checkpoint = dict(
        model=Pipeline([*preprocessing.steps, ("mlp", best_model)]),
        target_scaler=target_scaler,
        best_epoch=best_epoch,
        epochs_run=epoch,
        n_preprocessing_train=len(x_train),
        n_fit_train=len(indices),
    )
    return checkpoint, pd.DataFrame(history)


def predict_mlp(checkpoint, x, *, threads=2):
    """Predict in the original target units using a frozen checkpoint."""
    with threadpool_limits(limits=threads):
        standardized = checkpoint["model"].predict(x)
    return (
        checkpoint["target_scaler"]
        .inverse_transform(standardized.reshape(-1, 1))
        .ravel()
    )


def run_mlp_experiment(
    base,
    outdir,
    *,
    arms,
    n_features,
    target,
    inputs,
    seeds=(42, 43, 44),
    max_epochs=500,
    patience=40,
):
    """Train matched MLPs and save frozen checkpoints, predictions and scores.

    Parameters
    ----------
    base : pathlib.Path
        Prepared per-arm matrices and per-seed 10000-row training indices.
    outdir : pathlib.Path
        New output directory; existing directories are refused.
    arms : tuple of str
        Representation names to compare. All must have identical targets.
    n_features : int
        Expected input width: ten latents, or ten latents plus NORM/redshift.
    target, inputs : str
        Descriptions recorded with the results.
    seeds : tuple of int, optional
        Training subsets and random seeds to use.
    max_epochs, patience : int, optional
        Training cap and epochs without validation improvement before stopping.

    Returns
    -------
    pandas.DataFrame
        Mean scores and test-score range across seeds for each representation.
    """
    base = Path(base)
    parameters = dict(
        MLP_PARAMETERS,
        max_epochs=max_epochs,
        patience=patience,
        selection="minimum MSE on existing full validation split",
        inputs=inputs,
        target=target,
    )
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    results = []
    with np.load(base / f"{arms[0]}_matrices.npz") as ref:
        for arm in arms:
            with np.load(base / f"{arm}_matrices.npz") as data:
                for split in ("train", "val", "test"):
                    np.testing.assert_array_equal(data[f"y_{split}"], ref[f"y_{split}"])
                    if data[f"X_{split}"].shape[1] != n_features:
                        raise ValueError(f"{arm}/{split}: expected {n_features} inputs")
                    if n_features > 10:
                        np.testing.assert_array_equal(
                            data[f"X_{split}"][:, 10:], ref[f"X_{split}"][:, 10:]
                        )
                for seed in seeds:
                    start = time.time()
                    idx = np.load(base / f"train_indices_seed{seed}.npy")
                    if len(idx) != 10000 or len(np.unique(idx)) != 10000:
                        raise ValueError(
                            "Expected 10000 distinct matched training rows"
                        )
                    checkpoint, history = fit_mlp(
                        data["X_train"][idx].astype(float),
                        data["y_train"][idx].astype(float),
                        data["X_val"].astype(float),
                        data["y_val"].astype(float),
                        seed=seed,
                        max_epochs=max_epochs,
                        patience=patience,
                    )
                    checkpoint["parameters"] = parameters
                    name = f"{arm}_seed{seed}"
                    # Freeze and persist the top validation score before predicting test rows.
                    joblib.dump(checkpoint, outdir / f"{name}.joblib")
                    history.to_csv(outdir / f"{name}_history.csv", index=False)
                    pred = predict_mlp(checkpoint, data["X_test"])
                    pv = predict_mlp(checkpoint, data["X_val"])
                    if not np.isfinite(pred).all():
                        raise RuntimeError(
                            "Nonfinite selected test prediction; no replacement"
                        )
                    np.save(outdir / f"{name}_test_predictions.npy", pred)
                    row = dict(
                        arm=arm,
                        seed=seed,
                        best_epoch=checkpoint["best_epoch"],
                        epochs_run=checkpoint["epochs_run"],
                        hit_epoch_cap=checkpoint["epochs_run"] == max_epochs,
                        train_rows=len(idx),
                        val_r2=r2_score(data["y_val"], pv),
                        test_r2=r2_score(data["y_test"], pred),
                        test_rmse=np.sqrt(mean_squared_error(data["y_test"], pred)),
                        seconds=time.time() - start,
                    )
                    results.append(row)
                    pd.DataFrame(results).to_csv(outdir / "results.csv", index=False)
                    print(row, flush=True)
    summary = (
        pd.DataFrame(results)
        .groupby("arm")
        .agg(
            test_r2=("test_r2", "mean"),
            test_r2_min=("test_r2", "min"),
            test_r2_max=("test_r2", "max"),
            test_rmse=("test_rmse", "mean"),
            val_r2=("val_r2", "mean"),
            capped_runs=("hit_epoch_cap", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(outdir / "summary.csv", index=False)
    manifest = dict(
        implementation="sklearn.neural_network.MLPRegressor.partial_fit",
        sklearn_version=sklearn.__version__,
        parameters=parameters,
        device="cpu",
        numerical_threads=2,
        seeds=seeds,
        hashes={
            str(p): sha256(p)
            for p in [
                *[base / f"{a}_matrices.npz" for a in arms],
                *[base / f"train_indices_seed{s}.npy" for s in seeds],
            ]
        },
    )
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(summary.to_string(index=False), flush=True)
    return summary
