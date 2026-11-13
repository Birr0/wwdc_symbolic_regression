"""Shared symbolic search and diagnostic regression fitting."""

from copy import deepcopy

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

N = 10000
SEARCH_PARAMETERS = {
    "allowed_symbols": "add,sub,mul,div,exp,log1p,logabs,constant,variable,pow",
    "offspring_generator": "basic",
    "optimizer_iterations": 1000,
    "max_length": 25,
    "initialization_method": "btc",
    "n_threads": 8,
    "objectives": ["mse", "length"],
    "epsilon": 1e-6,
    "reinserter": "keep-best",
    "max_evaluations": 10_000_000,
    "symbolic_mode": False,
    "tournament_size": 3,
}


def search_parameters(seed, overrides=None):
    """Return independent search settings with the requested random seed.

    Optional overrides change individual settings but cannot change the seed.
    Defaults reproduce the historical pyoperon 0.4.0 search configuration.
    """
    parameters = deepcopy(SEARCH_PARAMETERS)
    parameters["random_state"] = seed
    parameters.update(deepcopy(overrides or {}))
    if parameters["random_state"] != seed:
        raise ValueError("Search random_state must match the requested seed")
    return parameters


def fit_sym_fn(
    feature,
    embed_type,
    merged_dfs,
    seed,
    *,
    matrices=None,
    train_indices=None,
    parameter_overrides=None,
):
    """Fit a symbolic Pareto front using shared preprocessing and search settings.

    Parameters
    ----------
    feature : str
        Target column when preparing DataFrames; a metadata label when using
        matrices. Flux targets are filtered and log10-transformed by the builder.
    embed_type : str
        Input representation: z, cond, cond+z, uncond, orig, or orig+z. When matrices are
        supplied, their columns are used unchanged and this is only a label.
    merged_dfs : dict[str, pandas.DataFrame] or None
        Split DataFrames to clean and scale. Pass None when supplying matrices.
    seed : int
        Random seed for sampling and the symbolic search.
    matrices : mapping or None, optional
        Already prepared X_train, y_train, X_test and y_test arrays. They are
        neither cleaned nor scaled again. Additional split arrays are ignored.
    train_indices : array-like or None, optional
        Explicit distinct training-row positions, used in the supplied order.
        Otherwise, sample at most 10,000 rows without replacement using seed.
    parameter_overrides : dict or None, optional
        Overrides to the shared search settings, retaining the requested seed.

    Returns
    -------
    dict
        Fitted Model, sampled X_train/y_train, full X_test/y_test, metadata,
        and the model's default expression at precision 12 (best_model).

    Notes
    -----
    DataFrame preprocessing fits the latent scaler on all cleaned training rows
    before sampling, and appends raw redshift after scaling for +z arms. Prepared
    matrices must already follow the intended preprocessing. Validation and test
    scores do not participate in the search. No previous run files are required.
    """
    if (merged_dfs is None) == (matrices is None):
        raise ValueError("Supply either merged_dfs or matrices, not both")
    if matrices is None:
        from wwdc_repro.shared.conditioning_data import build_matrices

        matrices = build_matrices(
            {split: merged_dfs[split] for split in ("train", "test")},
            feature,
            embed_type,
        )
    x_train, y_train = matrices["X_train"], matrices["y_train"]
    if len(x_train) != len(y_train):
        raise ValueError("Training inputs and targets must have the same row count")
    if train_indices is None:
        train_indices = np.random.default_rng(seed).choice(
            len(x_train),
            size=min(N, len(x_train)),
            replace=False,
        )
    indices = np.asarray(train_indices)
    if (
        indices.ndim != 1
        or not np.issubdtype(indices.dtype, np.integer)
        or len(indices) == 0
        or len(np.unique(indices)) != len(indices)
        or np.any(indices < 0)
        or np.any(indices >= len(x_train))
    ):
        raise ValueError(
            "Training indices must be distinct valid integer row positions"
        )

    from pyoperon.sklearn import SymbolicRegressor

    reg = SymbolicRegressor(**search_parameters(seed, parameter_overrides))
    x_fit, y_fit = x_train[indices], y_train[indices]
    reg.fit(x_fit, y_fit)
    return {
        "Feature": feature,
        "Embed_Type": embed_type,
        "Seed": seed,
        "Model": reg,
        "X_train": x_fit,
        "y_train": y_fit,
        "X_test": matrices["X_test"],
        "y_test": matrices["y_test"],
        "best_model": reg.get_model_string(reg.model_, precision=12),
    }


def fit_model_fn(feature, embed_type, estimator, merged_dfs):
    """Fit a diagnostic regression baseline and return training and test scores.

    Parameters
    ----------
    feature : str
        Target column. Targets whose names contain ``flux`` are restricted to
        positive values and transformed with log10.
    embed_type : str
        Input representation: ``z``, ``cond``, ``cond+z``, ``uncond``, ``orig``, or ``orig+z``.
    estimator : sklearn-compatible regressor
        Model implementing ``fit`` and ``predict``; fitted in place.
    merged_dfs : dict[str, pandas.DataFrame]
        DataFrames containing embeddings and catalogue properties, keyed by
        split. Only ``train`` and ``test`` are used.

    Returns
    -------
    dict
        Training and test R2 and MSE, under ``Train_R2``, ``Test_R2``,
        ``Train_MSE``, and ``Test_MSE``.

    Notes
    -----
    Remove missing and sentinel (-9999) targets, then standard-scale all input
    columns, including redshift when present, using the cleaned training data.
    Fit on the full cleaned training split without subsampling or external
    validation-based early stopping. These diagnostic baselines are separate
    from the manuscript's dedicated MLP comparison workflows.
    """
    is_flux = "flux" in feature.lower()
    train_clean = merged_dfs["train"].dropna(subset=[feature])
    test_clean = merged_dfs["test"].dropna(subset=[feature])
    train_clean = train_clean[train_clean[feature] != -9999.0]
    test_clean = test_clean[test_clean[feature] != -9999.0]
    if is_flux:
        train_clean = train_clean[train_clean[feature] > 0]
        test_clean = test_clean[test_clean[feature] > 0]

    if embed_type in ("cond+z", "orig+z"):
        column = embed_type.removesuffix("+z")
        X_train = np.stack(train_clean[column].values)
        X_test = np.stack(test_clean[column].values)
        y_train = train_clean[feature].values
        y_test = test_clean[feature].values

        z_train = train_clean["z_x"].values.reshape(-1, 1)
        z_test = test_clean["z_x"].values.reshape(-1, 1)
        if len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)
        X_train = np.hstack((X_train, z_train))
        X_test = np.hstack((X_test, z_test))

    elif embed_type == "z":
        y_train = train_clean[feature].values
        y_test = test_clean[feature].values
        X_train = train_clean["z_x"].values.reshape(-1, 1)
        X_test = test_clean["z_x"].values.reshape(-1, 1)
    else:
        X_train = np.stack(train_clean[embed_type].values)
        X_test = np.stack(test_clean[embed_type].values)
        y_train = train_clean[feature].values
        y_test = test_clean[feature].values
        if len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)

    if is_flux:
        y_train = np.log10(y_train)
        y_test = np.log10(y_test)

    y_train = y_train.reshape(-1, 1).flatten()
    y_test = y_test.reshape(-1, 1).flatten()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    estimator.fit(X_train_scaled, y_train)

    train_predictions = estimator.predict(X_train_scaled)
    test_predictions = estimator.predict(X_test_scaled)

    return {
        "Train_R2": r2_score(y_train, train_predictions),
        "Test_R2": r2_score(y_test, test_predictions),
        "Train_MSE": mean_squared_error(y_train, train_predictions),
        "Test_MSE": mean_squared_error(y_test, test_predictions),
    }
