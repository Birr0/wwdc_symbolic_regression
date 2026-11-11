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

    Parameters
    ----------
    values: pd.Series
        Series of SDSS spectrum IDs
    source: str
        Source label used in error messages, such as "astroML.specObjID".

    Returns
    -------
    values: pd.Series
        Parsed spectrum IDs with int64 dtype and the original index.

    Raises
    ------
    ValueError
        If an ID is not an ASCII decimal integer or exceeds int64.

    Notes
    -----
    IDs are parsed without converting through floating point because SDSS
    spectrum IDs exceed the exact-integer range of float64.
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
    """
    Check that all input values can be converted to finite floats.

    Parameters
    ----------
    values: array-like
        Input values to be checked
    label: str
        Descriptive label used in error messages.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If values cannot be converted to floats or contain non-finite values.
    """
    try:
        finite = np.isfinite(np.asarray(values, dtype=float))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: expected numeric values") from exc
    if not finite.all():
        raise ValueError(f"{label}: {int((~finite).sum())} nonfinite values")


def latent_array(frame: pd.DataFrame, column: str, split: str) -> np.ndarray:
    """
    Stack and validate a latent column as a two-dimensional NumPy array.

    Parameters
    ----------
    frame: pd.DataFrame
        Data split in a dataframe
    column: str
        Specified column to turn into the array.
    split: str
        Split label (train, val, or test) used in error messages.

    Returns
    -------
    raw: np.ndarray
        Latent array with shape (len(frame), LATENT_DIMENSIONS).

    Raises
    ------
    ValueError
        If the frame is empty, latent shapes are inconsistent or are not
        (LATENT_DIMENSIONS,) or (1, LATENT_DIMENSIONS), or latent values
        cannot be converted to finite floats.
    """
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
    """
    Check that spectrum IDs are unique within and across splits.

    Parameters
    ----------
    frames: dict[str, pd.DataFrame]
        Dictionary of the data splits

    Returns
    -------
    None

    Raises
    ------
    ValueError
        when duplicate spectrum ID is detected within or across data splits
    """
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

    The astroML catalogue already implements z > 0.02. Here we additionally
    retain rows with mask_ratio <= 0.5 and embedding redshift z_x <= 0.3.

    Parameters
    ----------
    embeddings: Path
        Directory containing train, val, and test embedding parquet subdirectories.
    galspec: Path
        Path to galspec catalog data
    specgals_home: Path
        Path to the saved astroML specgals data

    Returns
    -------
    result: dict[str, pd.DataFrame]
        Merged DataFrame for each split, with sample counts in its attrs.

    Raises
    ------
    ValueError
        If spectrum ID conversion fails, mask_ratio or either redshift column
        cannot be converted to finite floats, or spectrum IDs are duplicated
        within or across splits.
    pandas.errors.MergeError
        If catalogue or embedding keys violate the one-to-one joins.
    FileNotFoundError
        If a required local catalogue or a split's embedding files are missing.
    """
    for path in (galspec, *(specgals_home / f"SDSSspecgalsDR8_{i}.fit.gz" for i in (1, 2))):
        if not path.is_file():
            raise FileNotFoundError(f"Missing local catalogue: {path}.")
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
    """
    Remove rows with missing target values or the sentinel -9999, then validate
    the remaining targets, redshifts, latents, and spectrum IDs.
    The default target is lgm_tot_p50.

    Parameters
    ----------
    frames: dict[str, pd.DataFrame]
        Dictionary of the data splits

    target: str
        Column to be cleaned.

    Returns
    -------
    result: dict[str, pd.DataFrame]
        Cleaned frames

    Raises
    ------
    ValueError
        If any split is empty after cleaning, retained target/redshift/latent
        values cannot be converted to finite floats, latent shapes are
        invalid, or spectrum IDs are duplicated within or across splits.

    """
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
    '''
    Produce a metadata summary for the data splits

    Parameters
    ----------
    frames: dict[str, pd.DataFrame]
        Dictionary of data splits
    
    Returns
    -------
    rows: list[dict] 
        Metadata in the form
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
    '''
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
    frames: dict[str, pd.DataFrame], arm: str, target: str = TARGET,
    *, scaled_columns: tuple[str, ...] = (),
) -> dict[str, np.ndarray]:
    """
    Build feature and target arrays for each split.

    For latent arms, fit StandardScaler on all cleaned training rows and use
    that same scaler to transform every split. Optional scalar columns follow
    the ten latents and are included in that scaler. Append raw redshift for +z
    arms; the z-only arm uses raw redshift without a scaler. Targets remain
    unscaled.

    Parameters
    ----------
    frames: dict[str, pd.DataFrame]
        Dictionary containing a training split and any validation/test splits.
    arm: str
        Feature selection: orig, cond, uncond, z, orig+z, or cond+z.
        Without scaled_columns these produce 10, 10, 10, 1, 11, or 11
        feature columns, respectively.
    target: str
        Label for y. By default this is lgm_tot_p50
    scaled_columns : tuple[str, ...], optional
        Numeric scalar columns appended after the latents, in the given order,
        and standardised using the training split. For example, ("norm",)
        supplies NORM as an explicit SR input without modifying the latents.
        Not supported for the redshift-only arm. Redshift in +z arms remains
        raw and is appended after these scaled features.

    Returns
    -------
    out: dict
        Dictionary containing the processed training, validation and test X and y.
        {
            "X_train": ...,  # shape (n_train, X_var_size)
            "X_val":   ...,  # shape (n_val, X_var_size)
            "X_test":  ...,  # shape (n_test, X_var_size)
            "y_train": ...,  # shape (n_train,)
            "y_val":   ...,  # shape (n_val,)
            "y_test":  ...,  # shape (n_test,)
        }

    Raises
    ------
    ValueError
        If the arm is unknown, redshift/target/selected latent values cannot be
        converted to finite floats, or selected latent arrays are empty or
        have invalid shapes, or extra scaled columns are duplicated, missing,
        nonfinite, non-scalar, or requested for the redshift-only arm.
    """
    if "train" not in frames:
        raise ValueError("Training split is required to build matrices")
    if arm not in (*LATENT_COLUMNS, "z", "orig+z", "cond+z"):
        raise ValueError(f"Unknown representation arm: {arm!r}")
    if isinstance(scaled_columns, str) or len(set(scaled_columns)) != len(scaled_columns):
        raise ValueError("scaled_columns must be a sequence of distinct column names")
    if arm == "z" and scaled_columns:
        raise ValueError("Extra scaled columns are not supported for the redshift-only arm")
    for split in frames:
        for column in scaled_columns:
            if column not in frames[split]:
                raise ValueError(f"{split}: missing scaled column {column!r}")
            values = frames[split][column].to_numpy()
            finite_values(values, f"{split}.{column}")
            if values.ndim != 1:
                raise ValueError(f"{split}.{column}: expected a scalar column")
        finite_values(frames[split].z_x, f"{split}.z_x")
        finite_values(frames[split][target], f"{split}.{target}")
    if arm == "orig+z":
        frames = {
            split: frame.assign(cond=frame["orig"]) for split, frame in frames.items()
        }
        arm = "cond+z"  # Reuse the cond+z path with original latents in copied frames.
    out = {}
    if arm == "z":
        for split in frames:
            out[f"X_{split}"] = frames[split].z_x.to_numpy().reshape(-1, 1)
    else:
        column = "cond" if arm == "cond+z" else arm
        raw = {split: latent_array(frames[split], column, split) for split in frames}
        if scaled_columns:
            raw = {
                split: np.column_stack((raw[split], *(
                    frames[split][column].to_numpy() for column in scaled_columns
                )))
                for split in frames
            }
        scaler = StandardScaler().fit(raw["train"])
        for split in frames:
            x = scaler.transform(raw[split])
            if arm == "cond+z":
                x = np.column_stack((x, frames[split].z_x.to_numpy()))
            out[f"X_{split}"] = x
    for split in frames:
        out[f"y_{split}"] = frames[split][target].to_numpy(dtype=float)
    return out


def preprocessing_state(frames: dict[str, pd.DataFrame]) -> dict:
    """
    Return JSON-serializable metadata for each train-fitted latent scaler.

    This returns a Python dictionary; the caller performs JSON serialization.

    Parameters
    ----------
    frames: dict[str, pd.DataFrame]
        Dictionary of data splits

    Returns
    -------
    result: dict
        Metadata associated with the scalers fitted on the training data.
        {
            "target": TARGET,
            "redshift": "raw; appended after latent standardisation",
            "fit_split": "all cleaned training rows",
            "scalers": {
                "orig": {"mean": [...], "scale": [...],
                         "variance": [...], "n_samples": ...},
                "cond": {...},
                "uncond": {...},
            },
        }
    """
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