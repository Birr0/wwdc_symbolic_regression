"""Prepare and fit symbolic-regression controls for the NORM-conditioned analysis.

Compare original and redshift-conditioned latents with explicit NORM and redshift
inputs, matching the NORM-conditioned runs' spectra, targets, training samples,
and search settings. Select equations by validation R2 at each length budget,
freeze the selections, then evaluate their test scores.

Save prepared inputs and per-run equations, candidate audits, selections, and
run metadata under ``conditioning_validation_2026-10-13/matched_controls/`` by
default. The shared fitting functions also support other prepared control inputs.

Usage:
    python -m wwdc_repro.norm.matched_conditioning_controls prepare
    python -m wwdc_repro.norm.matched_conditioning_controls fit --arm original --seed 42
"""

from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from wwdc_repro.shared.conditioning_data import build_merged, build_matrices
from wwdc_repro.norm.validate_conditioning import ROOT, SPECS
from wwdc_repro.shared.selection import (
    audit_candidate,
    score_expression as score,
    select_candidate,
)
from wwdc_repro.shared.utils import sha256 as digest

BASE = Path("conditioning_validation_2026-10-13")
OUT = BASE / "matched_controls"


def clean(frames):
    """Remove missing or sentinel total masses and reset each split's row index.

    Parameters
    ----------
    frames : dict[str, pandas.DataFrame]
        DataFrames keyed by split, each containing ``lgm_tot_p50``.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Cleaned splits with missing and -9999 target values removed and
        consecutive row indices.
    """
    return {
        k: v.dropna(subset=["lgm_tot_p50"])
        .query("lgm_tot_p50 != -9999")
        .reset_index(drop=True)
        for k, v in frames.items()
    }


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "preparation.json").exists() or list(OUT.glob("*_frames.pkl")):
        # check if prepared DataFrame files, such as original_frames.pkl or zflow_frames.pkl already
        # exist to stop old files being overwritten
        raise FileExistsError("Refusing to overwrite prepared controls")
    norm_frames = clean(
        build_merged(
            str(ROOT / SPECS["norm"][0]),
            str(ROOT / "galSpecExtra-dr8.fits"),
            str(ROOT / "sdss"),
        )
    )
    redshift_frames = clean(
        build_merged(
            str(ROOT / SPECS["z"][0]),
            str(ROOT / "galSpecExtra-dr8.fits"),
            str(ROOT / "sdss"),
        )
    )
    frames = {"original": {}, "zflow": {}}
    summary = {}
    for split, frame in norm_frames.items():
        ids = frame.specObjID.astype(str).to_numpy()
        # check ids are preserved
        np.testing.assert_array_equal(ids, np.load(BASE / f"norm_{split}_ids.npy"))
        np.testing.assert_array_equal(ids, np.load(BASE / f"znorm_{split}_ids.npy"))
        left = pd.DataFrame({"id": ids, "row": np.arange(len(ids))})
        right = pd.DataFrame(
            {
                "id": redshift_frames[split].specObjID.astype(str),
                "redshift_row": np.arange(len(redshift_frames[split])),
            }
        )
        for f in (left, right):
            f["occurrence"] = f.groupby("id", sort=False).cumcount()
        aligned = left.merge(
            right,
            on=["id", "occurrence"],
            how="left",
            validate="one_to_one",
            sort=False,
        )  # match the redshift-data to each NORM-data row

        # check norm rows have a match and their order was preserved.
        assert aligned.redshift_row.notna().all(), (
            "normalisation rows missing from redshift embeddings"
        )
        np.testing.assert_array_equal(aligned.row, np.arange(len(frame)))

        aligned_redshift_frame = (
            redshift_frames[split]
            .iloc[aligned.redshift_row.astype(int)]
            .reset_index(drop=True)
        )  # select redshift data rows

        np.testing.assert_array_equal(
            aligned_redshift_frame.specObjID.astype(str), ids
        )  # verify matching ids
        np.testing.assert_array_equal(
            aligned_redshift_frame.lgm_tot_p50, frame.lgm_tot_p50
        )  # verify masses match
        np.testing.assert_allclose(
            aligned_redshift_frame.z_x, frame.z_x, atol=0, rtol=0
        )  # verify redshift matches

        orig = np.stack(frame.orig)
        oldorig = np.stack(aligned_redshift_frame.orig)
        # Independently generated embeddings differ at floating-point inference
        # precision; IDs, targets and redshifts above must still match exactly.
        np.testing.assert_allclose(orig, oldorig, atol=1e-4, rtol=0)
        normval = np.stack(frame.y).reshape(len(frame), 1)
        for arm, latent in [
            ("original", orig),
            ("zflow", np.stack(aligned_redshift_frame.cond)),
        ]:
            frames[arm][split] = pd.DataFrame(
                {
                    "cond": list(np.hstack([latent, normval])),
                    "z_x": frame.z_x.to_numpy(),
                    "lgm_tot_p50": frame.lgm_tot_p50.to_numpy(),
                    "specObjID": frame.specObjID.to_numpy(),
                }
            )
        np.save(OUT / f"{split}_ids.npy", ids.astype(str))
        summary[split] = dict(
            rows=len(frame),
            repeated_ids=int(left.id.duplicated().sum()),
            old_orig_max_abs_difference=float(np.max(np.abs(orig - oldorig))),
            ordered_ids_sha256=hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        )  # record metadata for data reproduction
    for arm in frames:
        pd.to_pickle(frames[arm], OUT / f"{arm}_frames.pkl")
        mats = build_matrices(frames[arm], "lgm_tot_p50", "cond+z")  # make the matrices
        reference = np.load(BASE / "norm_cond+z_matrices.npz")
        for split in frames[arm]:
            np.testing.assert_array_equal(
                mats[f"y_{split}"], reference[f"y_{split}"]
            )  # check y alignment
            np.testing.assert_array_equal(
                mats[f"X_{split}"][:, 10:], reference[f"X_{split}"][:, 10:]
            )  # # check X alignment
        np.savez(OUT / f"{arm}_matrices.npz", **mats)  # save the matrices
    for seed in (42, 43, 44):
        idx = np.random.default_rng(seed).choice(
            len(norm_frames["train"]), 10000, replace=False
        )
        np.save(OUT / f"train_indices_seed{seed}.npy", idx)  # save training indicies
        pd.DataFrame(
            {
                "row": idx,
                "specObjID": norm_frames["train"].specObjID.iloc[idx].to_numpy(),
            }
        ).to_csv(OUT / f"train_ids_seed{seed}.csv", index=False)
    manifest = dict(
        samples=summary,
        inputs="Exact NORM flow conditioning values, appended to original or aligned redshift-flow latents; common rows and existing split assignments retained.",
        policy="Best validation R2 per seed at each length budget 5..29; finite full-train and validation predictions; native training agreement <=1e-6; test evaluated only after freezing selections.",
        sha256={
            str(p): digest(p)
            for p in [
                *OUT.glob("*frames.pkl"),
                *OUT.glob("train_ids*.csv"),
            ]
        },
    )  # save preparation metadata for later checking
    (OUT / "preparation.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def run_searches(out, arms, seeds, *, prepared_indices=False, max_workers=2, selection="train"):
    """Launch shared fitting jobs for each requested representation and seed.

    Parameters
    ----------
    out : pathlib.Path
        Directory containing prepared inputs and receiving search outputs.
    arms : iterable of str
        Representations to fit.
    seeds : iterable of int
        Random seeds to run for each representation.
    prepared_indices : bool, optional
        Fit saved matrices on the recorded training indices.
    max_workers : int, optional
        Maximum number of concurrent search processes, default two.

    Returns
    -------
    None

    Raises
    ------
    subprocess.CalledProcessError
        If a search process fails.

    Notes
    -----
    Jobs with existing completion manifests are skipped. Each process writes
    a separate log and limits numerical-library threads to one. Operon retains
    its configured search parallelism. This function does not generate reports.
    """
    out = Path(out)
    pairs = [(arm, seed) for seed in seeds for arm in arms]

    def task(pair):
        arm, seed = pair
        manifest_path = out / f"{arm}_seed{seed}/manifest.json"
        if manifest_path.exists():
            saved_policy = json.loads(manifest_path.read_text()).get('selection', 'validation')
            if saved_policy != selection:
                raise ValueError(f'{manifest_path}: saved {saved_policy} selection differs from {selection}; use reporting to reselect or a fresh output directory')
            return
        env = dict(
            os.environ,
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "fit",
            "--out",
            str(out),
            "--arm",
            arm,
            "--seed",
            str(seed), "--selection", selection,
        ]
        if prepared_indices:
            command.append("--prepared-indices")
        with (out / f"{arm}_seed{seed}.log").open("w") as log:
            subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True
            )
        print("Completed", arm, seed, flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(task, pairs))


def fit(
    arm, seed, target="lgm_tot_p50", prepared_indices=False, parameter_overrides=None, selection="train"
):
    """Fit one symbolic search, select equations on validation data, and score tests.

    Parameters
    ----------
    arm : str
        Representation name identifying the prepared files under ``OUT``.
    seed : int
        Search seed and identifier of the saved training-row indices.
    target : str, optional
        Target column for the DataFrame fitting route, default ``lgm_tot_p50``.
        With prepared matrices, targets come from their saved y arrays; this
        name is recorded as metadata and does not change those arrays.
    prepared_indices : bool, optional
        Use saved matrices instead of rebuilding them from saved DataFrames.
        Both input formats use ``fit_sym_fn`` and the recorded training indices.
    parameter_overrides : dict or None, optional
        Overrides to the shared search settings, retaining the requested seed.

    Returns
    -------
    None

    Notes
    -----
    Audit exported equations against native training predictions and require
    training agreement before selection; validation mode also requires finite
    full-training and validation scores. Save selected equations before evaluating their test scores.
    Write ``candidate_audit.csv``, ``selection_frozen_before_test.csv``,
    ``selected.csv``, and a completion ``manifest.json`` under
    ``OUT/{arm}_seed{seed}/``. An existing run directory is not overwritten.
    """
    from wwdc_repro.shared.fitting import fit_sym_fn

    dest = OUT / f"{arm}_seed{seed}"
    dest.mkdir(exist_ok=False)
    mats = np.load(OUT / f"{arm}_matrices.npz")
    idx = np.load(OUT / f"train_indices_seed{seed}.npy")
    started = time.time()
    print("START", arm, seed, flush=True)
    frames = None if prepared_indices else pd.read_pickle(OUT / f"{arm}_frames.pkl")
    result = fit_sym_fn(
        target,
        "cond+z",
        frames,
        seed,
        matrices=mats if prepared_indices else None,
        train_indices=idx,
        parameter_overrides=parameter_overrides,
    )
    np.testing.assert_array_equal(result["X_train"], mats["X_train"][idx])
    np.testing.assert_array_equal(result["y_train"], mats["y_train"][idx])
    model = result["Model"]
    records = []
    trees = {}
    for i, member in enumerate(model.pareto_front_):
        tree = member["tree"]
        trees[i] = tree
        eq = model.get_model_string(tree, precision=60)
        native = model.evaluate_model(tree, result["X_train"])
        native_r2 = (
            float(r2_score(result["y_train"], native))
            if np.isfinite(native).all()
            else np.nan
        )
        checks = audit_candidate(
            eq,
            mats,
            idx,
            native_r2,
            check_full_training=selection == "validation", selection=selection,
        )
        records.append(
            dict(
                arm=arm,
                seed=seed,
                row=i,
                Length=int(tree.Length),
                equation=eq,
                equation_p12=model.get_model_string(tree, precision=12),
                train_r2_stored=native_r2,
                **checks,
            )
        )
    audit = pd.DataFrame(records)
    audit.to_csv(dest / "candidate_audit.csv", index=False)
    chosen = []
    for budget in range(5, model.get_params()["max_length"] + 5):
        r = select_candidate(
            audit,
            budget,
            length_column="Length",
            validation_column="val_r2", selection=selection,
        )
        if r is not None:
            chosen.append(dict(r.to_dict(), budget=budget))
    chosen = pd.DataFrame(chosen)
    chosen.to_csv(dest / "selection_frozen_before_test.csv", index=False)
    evaluations = []
    for r in chosen.itertuples():
        te = score(r.equation, mats["X_test"], mats["y_test"])
        native = model.evaluate_model(trees[r.row], mats["X_test"])
        nt = (
            float(r2_score(mats["y_test"], native))
            if np.isfinite(native).all()
            else np.nan
        )
        evaluations.append(
            dict(
                r._asdict(),
                test_r2=te,
                native_test_r2=nt,
                test_status="ok" if np.isfinite(te) else "evaluation_failure",
            )
        )
    pd.DataFrame(evaluations).to_csv(dest / "selected.csv", index=False)
    front_tests = audit.copy()
    front_tests['test_r2'] = [score(eq, mats['X_test'], mats['y_test']) for eq in front_tests.equation]
    front_tests['test_status'] = np.where(np.isfinite(front_tests.test_r2), 'ok', 'evaluation_failure')
    front_tests.to_csv(dest / 'front_test_scores.csv', index=False)
    manifest = dict(
        arm=arm,
        seed=seed,
        target=target, selection=selection,
        prepared_indices=prepared_indices,
        parameter_overrides=parameter_overrides or {},
        parameter_source="wwdc_repro.shared.fitting.search_parameters",
        elapsed_seconds=time.time() - started,
        parameters=model.get_params(),
        versions={
            p: importlib.metadata.version(p)
            for p in ("pyoperon", "numpy", "scikit-learn", "sympy", "pandas")
        },
        input_hashes={
            str(p): digest(p)
            for p in [
                *(
                    [OUT / f"{arm}_matrices.npz", OUT / f"train_indices_seed{seed}.npy"]
                    if prepared_indices
                    else [OUT / f"{arm}_frames.pkl"]
                ),
                OUT / f"train_ids_seed{seed}.csv",
            ]
        },
    )  # note metadata from this fit.
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(
        "DONE",
        arm,
        seed,
        "minutes",
        manifest["elapsed_seconds"] / 60,
        "eligible",
        int(audit.selectable.sum()),
        "of",
        len(audit),
        flush=True,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["prepare", "fit"])
    ap.add_argument("--arm", choices=["original", "zflow", "norm"])
    ap.add_argument("--seed", type=int, choices=[42, 43, 44])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--selection", choices=("train", "validation"), default="train")
    ap.add_argument("--prepared-indices", action="store_true")
    args = ap.parse_args()
    OUT = args.out
    if args.stage == "prepare":
        prepare()
    else:
        if args.arm is None or args.seed is None:
            ap.error("fit requires --arm and --seed")
        fit(args.arm, args.seed, prepared_indices=args.prepared_indices, selection=args.selection)
