"""Shared catalogue preparation and binning for colour–magnitude diagrams."""

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM


def join_photometry(rows, cache, *, validate="one_to_one", magnitude_shift=None):
    """Join spectrum rows to valid photometry and add ``gr`` and ``Mr``.

    Parameters
    ----------
    rows : pandas.DataFrame
        Spectrum rows containing specObjID and z, plus any score columns.
    cache : pathlib.Path
        Cached catalogue with specObjID, ABSMAG_g and ABSMAG_r.
    validate : str, optional
        Pandas merge cardinality check; use many_to_one for repeated spectra.
    magnitude_shift : float or None, optional
        Fixed legacy magnitude shift. By default calculate the h=1 to H0=70
        conversion with Astropy, at fixed Om0=0.3 and no radiation.

    Returns
    -------
    pandas.DataFrame
        Matched rows with finite magnitudes between -99 and zero, colour gr,
        and converted absolute magnitude Mr. No score or redshift cuts apply.
    """
    phot = pd.read_parquet(cache)[["specObjID", "ABSMAG_g", "ABSMAG_r"]]
    result = rows.merge(phot, on="specObjID", how="inner", validate=validate)
    for column in ("ABSMAG_g", "ABSMAG_r"):
        result = result[np.isfinite(result[column]) & (result[column] > -99)
                        & (result[column] < 0)]
    result = result.copy()
    result["gr"] = result.ABSMAG_g - result.ABSMAG_r
    if magnitude_shift is None:
        catalogue = FlatLambdaCDM(H0=100, Om0=0.3, Tcmb0=0)
        plot = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=0)
        magnitude_shift = (catalogue.distmod(result.z.to_numpy())
                           - plot.distmod(result.z.to_numpy())).value
    result["Mr"] = result.ABSMAG_r + magnitude_shift
    return result


def bin_scores(magnitude, colour, score, xedges, yedges, *, min_count=25):
    """Return counts and mean scores, masking bins below min_count."""
    counts = np.histogram2d(magnitude, colour, bins=(xedges, yedges))[0]
    totals = np.histogram2d(magnitude, colour, bins=(xedges, yedges), weights=score)[0]
    mean = np.divide(totals, counts, out=np.full_like(totals, np.nan),
                     where=counts >= min_count)
    return counts, mean


def masters10_red_edge(mr):
    """Return the Masters et al. (2010) blue edge of the red sequence."""
    return 0.63 - 0.02 * (mr + 20.0)


def masters10_ridge(mr):
    """Return the Masters et al. (2010) red-sequence ridge."""
    return 0.73 - 0.02 * (mr + 20.0)
