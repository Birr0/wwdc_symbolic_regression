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
LATENT_COLUMNS = ("orig", "cond", "uncond")
LATENT_DIMENSIONS = 10


def spectrum_ids(values: pd.Series, source: str) -> pd.Series:
    """Read decimal IDs, including the archived stringified byte representation.

    Do not extract arbitrary digit substrings or convert through floating point:
    SDSS spectrum IDs exceed the exact-integer range of float64.
    """
    def parse(value):
        if isinstance(value, bytes):
            value = value.decode("ascii")
        text = str(value).strip()
        if len(text) >= 3 and text[:2] in ("b'", 'b"') and text[-1] == text[1]:
            text = text[2:-1].strip()
        if not text or not text.isascii() or not text.isdecimal():
            raise ValueError(f"{source}: invalid spectrum ID {value!r}")
        number = int(text)
        if number > np.iinfo(np.int64).max:
            raise ValueError(f"{source}: spectrum ID exceeds int64: {value!r}")
        return number

    return values.map(parse).astype("int64")


def finite_values(values, label: str) -> None:
    try:
        finite = np.isfinite(np.asarray(values, dtype=float))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: expected numeric values") from exc
    if not finite.all():
        raise ValueError(f"{label}: {int((~finite).sum())} nonfinite values")


def latent_array(frame: pd.DataFrame, column: str, split: str) -> np.ndarray:
    if frame.empty:
        raise ValueError(f"{split}: empty sample")
    try:
        raw = np.stack(frame[column].to_numpy())
    except ValueError as exc:
        raise ValueError(f"{split}.{column}: inconsistent latent shapes") from exc
    if raw.shape[1:] not in ((LATENT_DIMENSIONS,), (1, LATENT_DIMENSIONS)):
        raise ValueError(f"{split}.{column}: expected {LATENT_DIMENSIONS} latent coordinates; got {raw.shape[1:]}")
    raw = raw.reshape(len(frame), LATENT_DIMENSIONS)
    finite_values(raw, f"{split}.{column}")
    return raw


def validate_split_ids(frames: dict[str, pd.DataFrame]) -> None:
    seen: set[int] = set()
    for split in SPLITS:
        ids = frames[split].specObjID
        if ids.duplicated().any():
            raise ValueError(f"{split}: duplicate spectrum IDs")
        current = set(ids)
        if seen & current:
            raise ValueError(f"{split}: spectrum IDs overlap between splits")
        seen.update(current)


def load_sample(
    embeddings: Path, galspec: Path, specgals_home: Path
) -> dict[str, pd.DataFrame]:
    """Join embeddings, astroML SDSS specgals, and galSpecExtra by spectrum ID.

    The astroML catalogue itself imposes z > 0.02. The saved embedding split is
    by spectrum row; this function does not claim physical-object separation.
    """
    for path in (galspec, *(specgals_home / f"SDSSspecgalsDR8_{i}.fit.gz" for i in (1, 2))):
        if not path.is_file():
            raise FileNotFoundError(f"Missing local catalogue: {path}; see docs/INPUTS.md")
    sdss = pd.DataFrame(fetch_sdss_specgals(data_home=str(specgals_home), download_if_missing=False))
    sdss["merge_id"] = spectrum_ids(sdss["specObjID"], "astroML.specObjID")
    extra = (
        Table.read(galspec, format="fits")
        .to_pandas()
        .dropna(subset=["SPECOBJID"])
        .copy()
    )
    extra["specObjID"] = spectrum_ids(extra["SPECOBJID"], "galSpecExtra.SPECOBJID")
    result = {}
    for split in SPLITS:
        files = sorted((embeddings / split).glob("*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"No embedding parquet files in {embeddings / split}"
            )
        frame = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
        counts = {"embedding_rows": len(frame)}
        frame["merge_id"] = spectrum_ids(frame["id"], f"{split}.id")
        matched = frame.merge(
            sdss, on="merge_id", how="inner", validate="one_to_one"
        ).drop(columns="merge_id")
        counts["after_specgals_join"] = len(matched)
        for column in ("mask_ratio", "z_x", "z_y"):
            finite_values(matched[column], f"{split}.{column}")
        matched = matched[matched.mask_ratio <= 0.5]
        counts["after_mask_cut"] = len(matched)
        matched = matched[matched.z_x <= 0.3].reset_index(drop=True)
        counts["after_redshift_cut"] = len(matched)
        result[split] = matched.merge(
            extra,
            on="specObjID",
            how="inner",
            suffixes=("_", ""),
            validate="one_to_one",
        )
        counts["after_galspec_join"] = len(result[split])
        result[split].attrs["sample_counts"] = counts
    validate_split_ids(result)
    return result


def clean_target(
    frames: dict[str, pd.DataFrame], target: str = TARGET
) -> dict[str, pd.DataFrame]:
    result = {}
    for split, frame in frames.items():
        clean = frame.loc[frame[target].notna() & (frame[target] != -9999)].copy()
        if clean.empty:
            raise ValueError(f"{split}: no rows after target cleaning")
        finite_values(clean[target], f"{split}.{target}")
        finite_values(clean.z_x, f"{split}.z_x")
        for column in LATENT_COLUMNS:
            latent_array(clean, column, split)
        clean.attrs["sample_counts"] = {
            **frame.attrs.get("sample_counts", {}), "after_target_cleaning": len(clean)
        }
        result[split] = clean
    validate_split_ids(result)
    return result


def sample_summary(frames: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    for split in SPLITS:
        frame = frames[split]
        ids = frame.specObjID.to_numpy(dtype="int64")
        rows.append(
            {
                "split": split,
                "rows": len(frame),
                "unique_specObjID": len(np.unique(ids)),
                "id_sha256": hashlib.sha256(
                    np.sort(ids).astype("<i8").tobytes()
                ).hexdigest(),
                "ordered_id_sha256": hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest(),
                **frame.attrs.get("sample_counts", {}),
                "z_min": float(frame.z_x.min()),
                "z_max": float(frame.z_x.max()),
            }
        )
    return rows


def matrices(
    frames: dict[str, pd.DataFrame], arm: str, target: str = TARGET
) -> dict[str, np.ndarray]:
    """Fit the latent scaler on all cleaned training rows, then append raw z.

    The historical original+z control put original latents in the conditional
    column before taking the conditional+z route; that is reproduced here.
    """
    if arm not in (*LATENT_COLUMNS, "z", "orig+z", "cond+z"):
        raise ValueError(f"Unknown representation arm: {arm!r}")
    for split in SPLITS:
        finite_values(frames[split].z_x, f"{split}.z_x")
        finite_values(frames[split][target], f"{split}.{target}")
    if arm == "orig+z":
        frames = {
            split: frame.assign(cond=frame["orig"]) for split, frame in frames.items()
        }
        arm = "cond+z"
    out = {}
    if arm == "z":
        for split in SPLITS:
            out[f"X_{split}"] = frames[split].z_x.to_numpy().reshape(-1, 1)
    else:
        column = "cond" if arm == "cond+z" else arm
        raw = {split: latent_array(frames[split], column, split) for split in SPLITS}
        scaler = StandardScaler().fit(raw["train"])
        for split in SPLITS:
            x = scaler.transform(raw[split])
            if arm == "cond+z":
                x = np.column_stack((x, frames[split].z_x.to_numpy()))
            out[f"X_{split}"] = x
    for split in SPLITS:
        out[f"y_{split}"] = frames[split][target].to_numpy(dtype=float)
    return out


def preprocessing_state(frames: dict[str, pd.DataFrame]) -> dict:
    """Serialize the train-fitted latent scaler; redshift remains unscaled."""
    result = {
        "target": TARGET,
        "redshift": "raw; appended after latent standardisation",
        "fit_split": "all cleaned training rows",
        "scalers": {},
    }
    for column in LATENT_COLUMNS:
        raw = latent_array(frames["train"], column, "train")
        scaler = StandardScaler().fit(raw)
        result["scalers"][column] = {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "variance": scaler.var_.tolist(),
            "n_samples": int(scaler.n_samples_seen_),
        }
    return result
