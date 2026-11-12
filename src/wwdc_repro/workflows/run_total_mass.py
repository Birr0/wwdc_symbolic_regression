"""Run symbolic-regression searches for a mass target across representations and seeds.

Load embeddings and catalogue properties, fit each requested representation/seed
combination, and evaluate its training-derived Pareto front. Also fit MLP and
linear-regression reference models.

Write ``pareto_fronts.csv`` and ``summary_metrics.csv`` to
``job_results_{feature}_{embed_type}_seed{seed}/``. The default target is total
stellar mass (``lgm_tot_p50``). Use ``--conditioning norm`` for NORM embeddings
and ``--append-norm`` to include their conditioning values explicitly.
``--out`` selects the parent output directory. Verified completed runs are skipped;
``--restart-incomplete`` preserves and restarts interrupted/unverified attempts.
``--threads`` sets Operon threads per search (default eight).

Usage:
    python -m wwdc_repro.workflows.run_total_mass --embed-type cond+z --seed 42
    python -m wwdc_repro.workflows.run_total_mass  # All six representations and three seeds.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from wwdc_repro.shared.fitting import fit_model_fn, fit_sym_fn, search_parameters
from wwdc_repro.shared.search_runs import search_run, input_hashes, add_search_options
from wwdc_repro.shared.conditioning_data import build_merged

EMBED_TYPES = ["z", "cond", "cond+z", "uncond", "orig", "orig+z"]  # embeddings to be fit
SEEDS = [42, 43, 44]


def run_cell(
    feature: str, embed_type: str, seed: int, merged: dict[str, pd.DataFrame], *, threads: int = 8
) -> tuple[pd.DataFrame, dict]:
    """Fit and score symbolic and reference models for specified embeddings and seed.

    Parameters
    ----------
    feature : str
        Name of the target column in the merged DataFrames.
    embed_type : str
        Input representation: ``z``, ``cond``, ``cond+z``, ``uncond``, ``orig``, or ``orig+z``.
    seed : int
        Random seed passed to the symbolic fitter and MLP model.
    merged : dict[str, pandas.DataFrame]
        Training, validation, and test DataFrames containing embeddings and
        catalogue properties, keyed by ``train``, ``val``, and ``test``.
    threads : int, optional
        Number of Operon threads for the symbolic search. Default is 8.

    Returns
    -------
    pareto : pandas.DataFrame
        Every equation on the training-derived Pareto front from Operon, with its length,
        training and test MSE and R2, and standard and higher-precision exports.
    summary : dict
        Target, representation, and seed identifiers; MLP and linear-regression
        test scores; and the test MSE of the best training equation.
    """
    started = time.time()
    results_sym = fit_sym_fn(feature, embed_type, merged, seed, parameter_overrides={"n_threads": threads})
    model = results_sym["Model"]

    best_len, best_train_mse, selected_test_mse = None, float("inf"), float("nan")
    pareto = []
    for m in model.pareto_front_:
        # prepare a list of equation results
        tree = m["tree"]
        pred_test = model.evaluate_model(tree, results_sym["X_test"])
        pred_train = model.evaluate_model(tree, results_sym["X_train"])

        # Score each split independently: test failures must not erase training scores.
        def native_scores(prediction, target):
            if not np.isfinite(prediction).all():
                return float("inf"), float("-inf")
            return mean_squared_error(target, prediction), r2_score(target, prediction)

        mse_train, r2_train = native_scores(pred_train, results_sym["y_train"])
        mse_test, r2_test = native_scores(pred_test, results_sym["y_test"])

        pareto.append(
            {
                "Feature": feature,
                "Embed_Type": embed_type,
                "Seed": seed,
                "Length": tree.Length,
                "Train_MSE": mse_train,
                "Train_R2": r2_train,
                "Test_MSE": mse_test,
                "Test_R2": r2_test,
                "Equation": m["model"],
                "Equation_p60": model.get_model_string(tree, precision=60),
                "Equation_p12": model.get_model_string(
                    tree, precision=12
                ),  # more precise saving of the constants etc.
            }
        )
        if mse_train < best_train_mse or (mse_train == best_train_mse and best_len is not None and tree.Length < best_len):
            best_train_mse, best_len = mse_train, tree.Length
            selected_test_mse = mse_test

    mlp = fit_model_fn(
        feature,
        embed_type,
        MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=100, random_state=seed),
        merged,
    )
    linear = fit_model_fn(feature, embed_type, LinearRegression(), merged)

    summary = {
        "Feature": feature,
        "Embed_Type": embed_type,
        "Seed": seed,
        "MLP_Test_R2": mlp["Test_R2"],
        "MLP_Test_MSE": mlp["Test_MSE"],
        "LR_Test_R2": linear["Test_R2"],
        "LR_Test_MSE": linear["Test_MSE"],
        "SYM_Training_Selected_Test_MSE": selected_test_mse,
    }
    elapsed = time.time() - started
    print(
        f"  {len(pareto)} front entries, training-selected length {best_len}, "
        f"MLP {mlp['Test_R2']:.3f} / LR {linear['Test_R2']:.3f}"
        f"  [{elapsed / 60:.1f} min]"
    )
    return pd.DataFrame(pareto), summary


def main(conditioning: str = "redshift") -> None:
    """Load a dataset, run requested searches, and save their fronts and summaries.

    Parameters
    ----------
    conditioning : str, optional
        Default embedding dataset: ``redshift`` or ``norm``. Both entry points
        use this runner; ``--conditioning`` can override the default.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    data_root = os.environ.get("DATA_ROOT", "data")
    parser.add_argument(
        "--conditioning", choices=("redshift", "norm", "znorm"), default=conditioning
    )
    parser.add_argument("--feature", default="lgm_tot_p50")
    parser.add_argument("--embed-type", default=None, choices=EMBED_TYPES)
    parser.add_argument("--seed", type=int, default=None, choices=SEEDS)
    parser.add_argument("--embeddings", help="Override the selected embedding dataset")
    parser.add_argument("--galspec", default=f"{data_root}/galSpecExtra-dr8.fits")
    parser.add_argument("--specgals-home", default=f"{data_root}/sdss")
    parser.add_argument(
        "--out", type=Path, default=Path("."), help="Parent output directory"
    )
    parser.add_argument(
        "--append-norm",
        action="store_true",
        help="Append stored NORM to cond; only cond and cond+z are supported",
    )
    add_search_options(parser)
    args = parser.parse_args()
    if args.append_norm and (
        args.conditioning not in ("norm", "znorm") or args.embed_type not in (None, "cond", "cond+z")
    ):
        parser.error("--append-norm requires NORM or joint embeddings and cond or cond+z")

    defaults = {
        "redshift": "sdss_II/spender_I_flow_v2/spender_I_flow_v2/embeddings/7655991_0",
        "norm": "sdss_II/spender_I_normalisation_flow/embeddings/norm_20261012183217_0",
        "znorm": "sdss_II/spender_I_z_normalisation_flow/embeddings/znorm_20261013104946_0",
    }
    embeddings = args.embeddings or str(Path(data_root) / defaults[args.conditioning])
    if args.embed_type:
        arms = [args.embed_type]
    elif args.append_norm:
        arms = ["cond", "cond+z"]
    elif args.conditioning == "norm":
        arms = ["cond+z"] + [arm for arm in EMBED_TYPES if arm != "cond+z"]
    else:
        arms = EMBED_TYPES
    seeds = [args.seed] if args.seed is not None else SEEDS

    # Preserve archived output names without creating artificial target columns.
    prefix = ""
    if args.conditioning in ("norm", "znorm"):
        prefix = f"{args.conditioning}flow_condnorm_" if args.append_norm else f"{args.conditioning}flow_"
    output_feature = f"{prefix}{args.feature}"
    cells = [
        (arm, seed, args.out / f"job_results_{output_feature}_{arm}_seed{seed}")
        for arm in arms
        for seed in seeds
    ]
    sources = [Path(args.galspec), *(Path(args.specgals_home) / f"SDSSspecgalsDR8_{i}.fit.gz" for i in (1, 2))]
    for split in ('train', 'val', 'test'):
        shards = sorted((Path(embeddings) / split).glob('*.parquet'))
        if not shards:
            raise FileNotFoundError(f'No embeddings in {embeddings}/{split}')
        sources.extend(shards)
    hashes = input_hashes(sources) # create input hashes of sources
    merged = None
    for embed_type, seed, job_dir in cells:
        configuration = dict(target=args.feature, arm=embed_type, seed=seed,
                             append_norm=args.append_norm, conditioning=args.conditioning,
                             parameters=search_parameters(seed, {"n_threads": args.threads}), input_hashes=hashes) # create search configuration
        with search_run(job_dir, configuration, ('pareto_fronts.csv', 'summary_metrics.csv'),
                        restart_incomplete=args.restart_incomplete) as required:
            if not required:
                continue
            if merged is None:
                merged = build_merged(embeddings, args.galspec, args.specgals_home)
                if args.append_norm:
                    for frame in merged.values():
                        cond = np.stack(frame['cond'].values)
                        norm = np.stack(frame['y'].values).reshape(len(frame), -1)[:, -1:]
                        frame['cond'] = list(np.hstack([cond, norm]))
            print(f"=== {output_feature} | {embed_type} | seed {seed} ===", flush=True)
            front, summary = run_cell(args.feature, embed_type, seed, merged, threads=args.threads)
            front['Feature'] = output_feature
            summary['Feature'] = output_feature
            front.to_csv(job_dir / 'pareto_fronts.csv', index=False)
            pd.DataFrame([summary]).to_csv(job_dir / 'summary_metrics.csv', index=False)
            print(f"Wrote {job_dir}/", flush=True)


if __name__ == "__main__":
    main()
