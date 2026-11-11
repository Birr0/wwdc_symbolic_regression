"""Matched fibre-mass SR for original and conditioned latent representations.

All four searches use ten latents + exact NORM + raw z. Prepare all inputs, then
fit --arm original|zflow|norm|znorm --seed 42|43|44.
"""

from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import pandas as pd
from wwdc_repro.shared.conditioning_data import build_merged, build_matrices
from wwdc_repro.norm.validate_conditioning import ROOT, SPECS
import wwdc_repro.norm.matched_conditioning_controls as controls

BASE = Path("conditioning_validation_2026-10-13")  # existing total mass data dir
OUT = Path("fibre_mass_conditioning_2026-10-14")  # existing fibre mass data dir
TARGET = "LGM_FIB_P50"
CONDITIONED_ARMS = ("zflow", "norm", "znorm")
ARMS = (*CONDITIONED_ARMS, "original")


def prepare():
    '''Prepare the embedding and fibre mass data.'''
    OUT.mkdir(exist_ok=False)
    frames = {arm: {} for arm in ARMS}
    summary = {}
    source = build_merged(
        str(ROOT / SPECS["norm"][0]),
        str(ROOT / "galSpecExtra-dr8.fits"),
        str(ROOT / "sdss"),
    )  # build norm-conditioned embedding dataframe with catalogue properties
    joint = build_merged(
        str(ROOT / SPECS["znorm"][0]),
        str(ROOT / "galSpecExtra-dr8.fits"),
        str(ROOT / "sdss"),
    )  # build z+norm-conditioned embedding dataframe with catalogue properties
    zframes = pd.read_pickle(
        BASE / "matched_controls/zflow_frames.pkl"
    )  # load redshift-conditioned data
    original_frames = pd.read_pickle(BASE / "matched_controls/original_frames.pkl")
    for split in ("train", "val", "test"):
        ids = np.load(BASE / "matched_controls" / f"{split}_ids.npy").astype(
            "int64"
        )  # load ids

        def align(frame):
            # check ids
            assert not frame.specObjID.duplicated().any()
            indexed = frame.set_index("specObjID", drop=False)
            return indexed.loc[ids].reset_index(drop=True)

        n = align(source[split])
        j = align(joint[split])
        z = zframes[split].reset_index(drop=True)
        original = original_frames[split].reset_index(drop=True)
        np.testing.assert_array_equal(z.specObjID, ids)  # check spectrums align
        np.testing.assert_array_equal(original.specObjID.astype(str), ids.astype(str))
        np.testing.assert_array_equal(original.z_x, n.z_x)
        np.testing.assert_array_equal(n[TARGET], j[TARGET])  # check mass data alignment
        valid = np.isfinite(n[TARGET].to_numpy()) & (
            n[TARGET].to_numpy() != -9999
        )  # mask non-valid values
        norm = np.stack(n.y).reshape(len(n), 1)
        np.testing.assert_array_equal(norm, np.stack(j.y)[:, -1:])  # check norm aligns
        np.testing.assert_array_equal(np.stack(original.cond)[:, 10:], norm)
        latents = {
            "norm": np.hstack([np.stack(n.cond), norm]),
            "znorm": np.hstack([np.stack(j.cond), norm]),
            "zflow": np.stack(z.cond),
            "original": np.stack(original.cond),
        }  # collate latents
        for arm in ARMS:
            frame = pd.DataFrame(
                {
                    "cond": list(latents[arm]),
                    "z_x": n.z_x.to_numpy(),
                    TARGET: n[TARGET].to_numpy(),
                    "specObjID": ids,
                }
            )  # create new dataframe
            frames[arm][split] = frame[valid].reset_index(drop=True)
        np.save(
            OUT / f"{split}_ids.npy", ids[valid].astype(str)
        )  # save the valid ids for the spectra
        np.save(
            OUT / f"{split}_parent_rows.npy", np.flatnonzero(valid)
        )  # save the valid spectra data
        summary[split] = dict(
            parent_rows=len(ids),
            rows=int(valid.sum()),
            excluded_invalid_fibre_mass=int((~valid).sum()),
            fibre_mass_min=float(n.loc[valid, TARGET].min()),
            fibre_mass_max=float(n.loc[valid, TARGET].max()),
            ordered_ids_sha256=hashlib.sha256(
                "\n".join(ids[valid].astype(str)).encode()
            ).hexdigest(),
        )  # create a summary for each split to track what is used in fitting.
    reference = None
    for arm in ARMS:
        pd.to_pickle(frames[arm], OUT / f"{arm}_frames.pkl")
        mats = build_matrices(frames[arm], TARGET, "cond+z")
        if reference is None:
            reference = mats
        else:
            for split in ("train", "val", "test"):
                np.testing.assert_array_equal(
                    mats[f"y_{split}"], reference[f"y_{split}"]
                )
                np.testing.assert_array_equal(
                    mats[f"X_{split}"][:, 10:], reference[f"X_{split}"][:, 10:]
                )
        np.savez(OUT / f"{arm}_matrices.npz", **mats)
        parent = np.load(
            BASE
            / (
                f"matched_controls/{arm}_matrices.npz"
                if arm in ("zflow", "original")
                else f"{arm}_cond+z_matrices.npz"
            )
        )
        if all(summary[s]["excluded_invalid_fibre_mass"] == 0 for s in summary):
            for split in summary:
                np.testing.assert_array_equal(
                    mats[f"X_{split}"], parent[f"X_{split}"]
                )  # check the variables used for the total mass align with the fibre mass

    # Retain exactly the same 10,000 training spectra as the total-mass fits.
    subset_checks = {}
    kept_parent = np.load(OUT / "train_parent_rows.npy")
    inverse = np.full(summary["train"]["parent_rows"], -1, dtype=int)
    inverse[kept_parent] = np.arange(len(kept_parent))
    for seed in (42, 43, 44):
        parent_idx = np.load(
            BASE / "matched_controls" / f"train_indices_seed{seed}.npy"
        )
        idx = inverse[parent_idx]
        missing = idx < 0
        if missing.any():
            raise ValueError(
                f"Seed {seed}: {int(missing.sum())} selected training spectra have invalid "
                "fibre masses; cannot preserve the matched total-mass training subset"
            )
        subset_checks[seed] = dict(shared_training_objects=len(idx))
        np.save(
            OUT / f"train_indices_seed{seed}.npy", idx
        )  # save the row positions within the fibre-mass training data
        pd.DataFrame(
            {
                "row": idx,
                "specObjID": frames["norm"]["train"].specObjID.iloc[idx].to_numpy(),
            }
        ).to_csv(OUT / f"train_ids_seed{seed}.csv", index=False)
        np.testing.assert_array_equal(
            kept_parent[idx], parent_idx
        )  # check ids align from parent sample
        for arm in ARMS:
            np.testing.assert_array_equal(
                frames[arm]["train"].specObjID.iloc[idx].to_numpy(),
                frames["norm"]["train"].specObjID.iloc[idx].to_numpy(),
            )
    # log details
    manifest = dict(
        target=TARGET,
        target_definition="MPA-JHU median log10 stellar mass inside fibre; no additional log transform",
        samples=summary,
        training_subsets=subset_checks,
        models="original: original Spender latents; zflow: redshift-conditioned; norm: NORM-conditioned; znorm: jointly conditioned",
        inputs="Ten representation latents, exact standardised-log NORM conditioning value, raw z. Full matched training scaler; same seeded 10000 rows across arms.",
        hashes={
            str(p): controls.digest(p)
            for p in [
                Path(__file__),
                Path(controls.__file__),
                Path(__file__).parents[1] / "shared/fitting.py",
                *OUT.glob("*_frames.pkl"),
                *OUT.glob("train_ids_seed*.csv"),
            ]
        },
    )
    (OUT / "preparation.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["samples"], indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["prepare", "fit"])
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--seed", type=int, choices=[42, 43, 44])
    ap.add_argument("--selection", choices=("train", "validation"), default="train")
    args = ap.parse_args()
    if args.stage == "prepare":
        prepare()
    else:
        if args.arm is None or args.seed is None:
            ap.error("fit requires --arm and --seed")
        controls.OUT = OUT
        controls.fit(args.arm, args.seed, target=TARGET, prepared_indices=True, selection=args.selection)
