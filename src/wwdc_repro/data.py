"""Rebuild the historical spectrum-level paper sample and feature matrices."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from astroML.datasets import fetch_sdss_specgals
from astropy.table import Table
from sklearn.preprocessing import StandardScaler

SPLITS = ("train", "val", "test")
TARGET = "lgm_tot_p50"


def load_sample(embeddings: Path, galspec: Path, specgals_home: Path) -> dict[str, pd.DataFrame]:
    """Join embeddings, astroML SDSS specgals, and galSpecExtra by spectrum ID.

    The astroML catalogue itself imposes z > 0.02. The saved embedding split is
    by spectrum row; this function does not claim physical-object separation.
    """
    sdss = pd.DataFrame(fetch_sdss_specgals(data_home=str(specgals_home)))
    sdss["merge_id"] = sdss["specObjID"].astype("int64")
    extra = Table.read(galspec, format="fits").to_pandas().dropna(subset=["SPECOBJID"]).copy()
    extra["specObjID"] = extra["SPECOBJID"].astype(str).str.extract(r"(\d+)")[0].astype("int64")
    result = {}
    for split in SPLITS:
        files = sorted((embeddings / split).glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No embedding parquet files in {embeddings / split}")
        frame = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
        frame["merge_id"] = frame["id"].astype(str).str.extract(r"(\d+)")[0].astype("int64")
        matched = frame.merge(sdss, on="merge_id", how="inner").drop(columns="merge_id")
        matched = matched[(matched.mask_ratio <= 0.5) & (matched.z_x <= 0.3)].reset_index(drop=True)
        result[split] = matched.merge(extra, on="specObjID", how="inner", suffixes=("_", ""))
    return result


def clean_target(frames: dict[str, pd.DataFrame], target: str = TARGET) -> dict[str, pd.DataFrame]:
    return {split: frame.loc[frame[target].notna() & (frame[target] != -9999)].copy()
            for split, frame in frames.items()}


def sample_summary(frames: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    for split in SPLITS:
        frame = frames[split]
        ids = frame.specObjID.to_numpy(dtype="int64")
        rows.append({
            "split": split, "rows": len(frame), "unique_specObjID": len(np.unique(ids)),
            "id_sha256": hashlib.sha256(np.sort(ids).astype("<i8").tobytes()).hexdigest(),
            "z_min": float(frame.z_x.min()), "z_max": float(frame.z_x.max()),
        })
    return rows


def matrices(frames: dict[str, pd.DataFrame], arm: str, target: str = TARGET) -> dict[str, np.ndarray]:
    """Fit the latent scaler on all cleaned training rows, then append raw z.

    The historical original+z control put original latents in the conditional
    column before taking the conditional+z route; that is reproduced here.
    """
    if arm == "orig+z":
        frames = {split: frame.assign(cond=frame["orig"]) for split, frame in frames.items()}
        arm = "cond+z"
    out = {}
    if arm == "z":
        for split in SPLITS:
            out[f"X_{split}"] = frames[split].z_x.to_numpy().reshape(-1, 1)
    else:
        column = "cond" if arm == "cond+z" else arm
        raw = {split: np.stack(frames[split][column].to_numpy()) for split in SPLITS}
        raw = {split: value.reshape(len(value), -1) for split, value in raw.items()}
        scaler = StandardScaler().fit(raw["train"])
        for split in SPLITS:
            x = scaler.transform(raw[split])
            if arm == "cond+z":
                x = np.column_stack((x, frames[split].z_x.to_numpy()))
            out[f"X_{split}"] = x
    for split in SPLITS:
        out[f"y_{split}"] = frames[split][target].to_numpy(dtype=float)
    return out
