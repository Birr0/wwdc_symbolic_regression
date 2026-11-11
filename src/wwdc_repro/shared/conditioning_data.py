"""Data preparation helpers for conditioning experiments.

Load and join embedding splits to SDSS catalogues, clean target values, and
build feature matrices using a scaler fitted on all cleaned training rows.
For cond+z, append raw redshift after scaling; leave the z-only arm unscaled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from wwdc_repro.shared.data import LATENT_DIMENSIONS, load_sample, matrices

def build_merged(
    root: str, galspec_path: str, specgals_home: str
) -> dict[str, pd.DataFrame]:
    """Load matched embedding splits using the shared redshift sample loader.

    Parameters
    ----------
    root : str
        Embedding directory containing train, val, and test Parquet subdirectories.
    galspec_path : str
        Path to the local galSpecExtra FITS catalogue.
    specgals_home : str
        Directory containing the local astroML SDSS catalogue files.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Catalogue-matched splits with mask_ratio <= 0.5 and z_x <= 0.3.
        Target values are not cleaned here.

    Notes
    -----
    This adapter converts paths for wwdc_repro.shared.data.load_sample, which validates
    IDs, finite mask/redshift values, one-to-one joins, and disjoint split IDs.
    Missing files or invalid data raise the shared loader's exceptions.
    """
    return load_sample(Path(root), Path(galspec_path), Path(specgals_home))


def build_matrices(
    merged_dfs: dict[str, pd.DataFrame], feature: str, embed_type: str
) -> dict[str, np.ndarray]:
    """Clean fitting targets and delegate feature scaling to the shared builder.

    Parameters
    ----------
    merged_dfs : dict[str, pandas.DataFrame]
        Training split and any validation/test splits with embeddings and targets.
    feature : str
        Target column. Flux targets retain positive values and use log10 units.
    embed_type : str
        Representation arm accepted by wwdc_repro.shared.data.matrices.

    Returns
    -------
    dict[str, numpy.ndarray]
        X/y arrays for each supplied split. Missing and -9999 targets are removed.

    Notes
    -----
    Legacy arrays with coordinates appended after the ten latents are separated
    into scalar columns for the shared builder. They remain scaled together with
    the latents, before raw redshift is appended for +z arms.
    """
    is_flux = "flux" in feature.lower()
    clean = {}
    for split, df in merged_dfs.items():
        sub = df.loc[df[feature].notna() & (df[feature] != -9999.0)].copy() # remove nan and filler
        if is_flux:
            sub = sub.loc[sub[feature] > 0].copy()
            sub[feature] = np.log10(sub[feature]) # log10 flux
        clean[split] = sub

    scaled_columns = ()
    if embed_type != "z":
        column = embed_type.removesuffix("+z")
        raw = {split: np.stack(frame[column].to_numpy()) for split, frame in clean.items()}
        raw = {split: values.reshape(len(values), -1) if values.ndim == 3 else values
               for split, values in raw.items()}
        widths = {values.shape[1] for values in raw.values()}
        if len(widths) != 1:
            raise ValueError("Feature widths differ between splits")
        width = widths.pop()
        if width > LATENT_DIMENSIONS:
            # Temporary columns for norm+z etc.
            scaled_columns = tuple(f"__appended_feature_{i}" for i in range(width - LATENT_DIMENSIONS))
            for split, frame in clean.items():
                if any(name in frame for name in scaled_columns):
                    raise ValueError("Temporary appended-feature column already exists")
                frame[column] = list(raw[split][:, :LATENT_DIMENSIONS])
                for i, name in enumerate(scaled_columns):
                    frame[name] = raw[split][:, LATENT_DIMENSIONS + i]

    return matrices(clean, embed_type, feature, scaled_columns=scaled_columns)
