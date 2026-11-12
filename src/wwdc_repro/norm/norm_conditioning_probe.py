"""Measure residual NORM information in original and conditioned latent representations.

Predict log10(NORM) from ten latent coordinates using linear regression and the
shared MLP with hidden layers of 64 and 32 neurons. Also measure each coordinate's
Pearson and Spearman correlations with the target. Compare original,
redshift-conditioned, and NORM-conditioned latents using matched splits and the
same training subsets for seeds 42–44. Neither NORM nor redshift is supplied as
a predictor.

Save prepared arrays, correlations, model results, and predictions under
``norm_probe/``. Verify the saved MLP predictions and
validation checkpoint epochs, then print a summary for the manuscript's NORM
probe comparison. Recover target scaling directly from raw NORM embeddings and
the flow checkpoint YAML under DATA_ROOT; no diagnostic metadata is required.
Existing output directories are not overwritten.
"""

from pathlib import Path
import shutil
import yaml
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import wwdc_repro.norm.latent_mlp as mlp
from wwdc_repro.shared.conditioning_data import build_merged
from wwdc_repro.norm.validate_conditioning import ROOT, SPECS

OUT = Path("norm_probe")
BASE = Path("norm_analysis")
MATCH = BASE / "matched_controls"
ARMS = ("original", "zflow", "norm")


def norm_target_scaling(data_root: Path, prepared: Path) -> tuple[float, float]:
    """Recover log10(NORM) scaling for the cleaned SR training sample.
        The prepared matrices keep the scaled norm so this recovers norm from
        the saved matrices. Norm is avaiable but this is a bit easier than reloading 
        the values and ensuring alignment.

    Parameters
    ----------
    data_root : pathlib.Path
        Root containing the NORM embeddings, flow checkpoint YAML and catalogues.
    prepared : pathlib.Path
        Conditioning input directory containing saved norm_{split}_ids.npy files.

    Returns
    -------
    mean, std : tuple[float, float]
        Mean and population standard deviation of log10(NORM) on the full
        cleaned training split, for undoing the SR matrix's NORM scaling.

    Notes
    -----
    The embedding y values are standardised using the flow's training sample.
    First undo that transform using its YAML statistics, then calculate new
    statistics on the cleaned mass sample. Saved spectrum IDs must match every
    split in order. No validation/test values enter the training statistics.
    """
    embeddings = data_root / SPECS["norm"][0]
    frames = build_merged(
        str(embeddings), str(data_root / "galSpecExtra-dr8.fits"),
        str(data_root / "sdss"),
    )
    frames = {
        split: frame.dropna(subset=["lgm_tot_p50"])
        .query("lgm_tot_p50 != -9999").reset_index(drop=True)
        for split, frame in frames.items()
    }
    for split, frame in frames.items():
        np.testing.assert_array_equal(
            frame.specObjID.astype(str), np.load(prepared / f"norm_{split}_ids.npy")
        )
    config_path = embeddings.parent.parent / "ckpts" / f"{embeddings.name}.yaml"
    stats = yaml.safe_load(config_path.read_text())["normalisation"]["train"]
    log_norm = np.stack(frames["train"].y).reshape(-1) * stats["std"] + stats["mean"]
    return float(log_norm.mean()), float(log_norm.std())


def main():
    OUT.mkdir(exist_ok=False)
    paths = {
        a: MATCH / f"{a}_matrices.npz"
        if a in ("original", "zflow")
        else BASE / f"{a}_cond+z_matrices.npz"
        for a in ARMS
    }
    reference = np.load(paths["original"])
    corr = []
    linear = []
    mu, sd = norm_target_scaling(ROOT, BASE)
    for seed in (42, 43, 44):
        shutil.copy2(
            MATCH / f"train_indices_seed{seed}.npy",
            OUT / f"train_indices_seed{seed}.npy",
        )
    for arm, path in paths.items():
        source = np.load(path)
        mats = {}
        idsbase = MATCH if arm in ("original", "zflow") else BASE
        for split in ("train", "val", "test"):
            ids = np.load(
                idsbase
                / (
                    f"{split}_ids.npy"
                    if arm in ("original", "zflow")
                    else f"{arm}_{split}_ids.npy"
                )
            )
            np.testing.assert_array_equal(ids, np.load(MATCH / f"{split}_ids.npy"))
            np.testing.assert_array_equal(
                source[f"X_{split}"][:, 10], reference[f"X_{split}"][:, 10]
            )
            mats[f"X_{split}"] = source[f"X_{split}"][:, :10]
            mats[f"y_{split}"] = source[f"X_{split}"][:, 10].astype(float) * sd + mu
        np.savez(OUT / f"{arm}_matrices.npz", **mats) # save latent-only inputs
        for j in range(10):
            x = mats["X_test"][:, j]
            y = mats["y_test"]
            corr.append(
                dict(
                    arm=arm,
                    coordinate=j + 1,
                    pearson=float(np.corrcoef(x, y)[0, 1]),
                    spearman=float(spearmanr(x, y).statistic),
                )
            ) # measure correlation of each latent variable with norm
        for seed in (42, 43, 44):
            idx = np.load(OUT / f"train_indices_seed{seed}.npy")
            model = LinearRegression().fit(mats["X_train"][idx], mats["y_train"][idx]) # fit a linear regression model
            pred = model.predict(mats["X_test"])
            np.save(OUT / f"{arm}_linear_seed{seed}_predictions.npy", pred)
            linear.append(
                dict(
                    arm=arm,
                    seed=seed,
                    test_r2=r2_score(mats["y_test"], pred),
                    test_rmse=np.sqrt(mean_squared_error(mats["y_test"], pred)),
                )
            )
    # save results
    pd.DataFrame(corr).to_csv(OUT / "coordinate_correlations.csv", index=False)
    pd.DataFrame(linear).to_csv(OUT / "linear_results.csv", index=False)
    mlp.run(base=OUT, outdir=OUT / "mlp_sklearn", arms=ARMS)
    report()


def report():
    corr = pd.read_csv(OUT / "coordinate_correlations.csv")
    lin = pd.read_csv(OUT / "linear_results.csv")
    m = pd.read_csv(OUT / "mlp_sklearn/summary.csv")
    results = pd.read_csv(OUT / "mlp_sklearn/results.csv")
    for r in results.itertuples():
        # check results align
        history = pd.read_csv(OUT / f"mlp_sklearn/{r.arm}_seed{r.seed}_history.csv")
        assert int(history.loc[history.val_mse_dex.idxmin(), "epoch"]) == r.best_epoch
        pred = np.load(OUT / f"mlp_sklearn/{r.arm}_seed{r.seed}_test_predictions.npy")
        y = np.load(OUT / f"{r.arm}_matrices.npz")["y_test"]
        assert np.isfinite(pred).all()
        np.testing.assert_allclose(r2_score(y, pred), r.test_r2, atol=1e-12)
    rows = []
    for arm in ARMS:
        correlations = corr[corr.arm == arm]
        mlp_summary = m[m.arm == arm].iloc[0]
        rows.append({
            "Representation": arm,
            "Max |Pearson r|": correlations.pearson.abs().max(),
            "Max |Spearman rho|": correlations.spearman.abs().max(),
            "Linear mean R2": lin.loc[lin.arm == arm, "test_r2"].mean(),
            "MLP mean R2": mlp_summary.test_r2,
            "MLP min R2": mlp_summary.test_r2_min,
            "MLP max R2": mlp_summary.test_r2_max,
            "MLP RMSE (dex)": mlp_summary.test_rmse,
        })
    print(pd.DataFrame(rows).to_string(index=False, float_format="{:.4f}".format), flush=True)



if __name__ == "__main__":
    main()
