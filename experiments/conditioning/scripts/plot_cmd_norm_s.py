"""CMD of the validation-selected NORM-conditioned additive mass correction.

Run from this directory with the project Python. Saves test and all-split PDFs,
400 dpi PNGs, the joined plotting data, bin values, provenance and caption.
No extra normalisation cut or restandardisation of S is applied.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from figstyle import use_science
from cmd_helpers import H1_TO_H70, masters10_red_edge, masters10_ridge
from val_rescore import to_sympy, evaluate_equation
from matched_conditioning_controls import digest

BASE = Path('conditioning_validation_2026-10-13')
OUT = Path('norm_cmd_s_2026-10-16')


def main():
    OUT.mkdir(exist_ok=True)
    source = BASE / 'validation_selected_final.csv'
    selected = pd.read_csv(source)
    r = selected[(selected.arm == 'norm_cond+z') & (selected.seed == 42) & (selected.budget == 15)].iloc[0]
    assert r.Length == 15
    expr = to_sympy(r.equation, 12)
    variables = [sp.Symbol(f'X{i}') for i in range(1, 11)]
    derivatives = [sp.diff(expr, v) for v in variables]
    assert all(not d.free_symbols for d in derivatives)
    weights = np.array([float(d) for d in derivatives])
    mats_path = BASE / 'norm_cond+z_matrices.npz'
    mats = np.load(mats_path)
    cache = Path('plot_cmd_latent_results/vagc_join.parquet')
    phot = pd.read_parquet(cache)[['specObjID', 'ABSMAG_g', 'ABSMAG_r']]
    assert phot.specObjID.is_unique
    pieces = []
    for split in ('train', 'val', 'test'):
        x = mats[f'X_{split}']
        ids = np.load(BASE / f'norm_{split}_ids.npy').astype('int64')
        s = x[:, :10] @ weights
        zero = x.copy(); zero[:, :10] = 0
        np.testing.assert_allclose(s, evaluate_equation(r.equation, x)-evaluate_equation(r.equation, zero), atol=1e-10, rtol=0)
        pieces.append(pd.DataFrame({'specObjID': ids, 'split': split, 'z': x[:, 11], 'S': s}))
    cat = pd.concat(pieces, ignore_index=True)
    df = cat.merge(phot, on='specObjID', how='inner', validate='many_to_one')
    for col in ('ABSMAG_g', 'ABSMAG_r'):
        df = df[np.isfinite(df[col]) & (df[col] > -99) & (df[col] < 0)]
    df = df[np.isfinite(df.S) & (df.z > .02) & (df.z <= .3)].copy()
    df['gr'] = df.ABSMAG_g-df.ABSMAG_r
    df['Mr'] = df.ABSMAG_r+H1_TO_H70
    df.to_parquet(OUT / 'plot_data.parquet', index=False)
    style = use_science()
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 17, 'xtick.labelsize': 13, 'ytick.labelsize': 13})
    xe = np.linspace(-24, -18, 61)
    ye = np.linspace(.15, 1.2, 57)
    metadata = dict(seed=42, length=15, budget=15, test_r2=float(r.test_r2),
                    equation=r.equation, weights=weights.tolist(), style=style,
                    input_sha256={str(p): digest(p) for p in (source, mats_path, cache)},
                    filters='Saved SR sample (mask_ratio<=0.5, valid target, z<=0.3); 0.02<z<=0.3; finite valid ABSMAG_g/r. No extra NORM cut.',
                    colour='ABSMAG_g - ABSMAG_r', magnitude_shift=H1_TO_H70,
                    min_count=20, colour_limits_dex=[-.2, .2], samples={})
    for split in ('test', 'all'):
        d = df if split == 'all' else df[df.split == split]
        counts, _, _ = np.histogram2d(d.Mr, d.gr, bins=(xe, ye))
        total, _, _ = np.histogram2d(d.Mr, d.gr, bins=(xe, ye), weights=d.S)
        mean = np.divide(total, counts, out=np.full_like(total, np.nan), where=counts >= 20)
        np.savez(OUT / f'bins_{split}.npz', magnitude_edges=xe, colour_edges=ye, counts=counts, mean_S=mean)
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True, sharey=True, layout='constrained')
        density = np.log10(np.where(counts > 0, counts, np.nan))
        im = axes[0].pcolormesh(xe, ye, density.T, cmap='Greys', shading='auto', rasterized=True)
        fig.colorbar(im, ax=axes[0], pad=.02).set_label(r'$\log_{10}(\mathrm{count\ per\ bin})$')
        im = axes[1].pcolormesh(xe, ye, mean.T, cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-.2, vcenter=0, vmax=.2), shading='auto', rasterized=True)
        fig.colorbar(im, ax=axes[1], pad=.02, extend='both').set_label(r'$\langle S\rangle$ [dex]')
        grid = np.linspace(-24, -18, 200)
        for ax in axes:
            ax.plot(grid, masters10_red_edge(grid), '--', color='.15', lw=1.3, label='Blue edge of red sequence')
            ax.plot(grid, masters10_ridge(grid), ':', color='.4', lw=1.3, label='Red-sequence ridge')
            ax.set(xlim=(-18.1, -23.9), ylim=(.2, .9), xlabel=r'$M_r$ [mag]', ylabel=r'$g-r$ [mag]')
            ax.tick_params(labelleft=True)
        axes[0].set_title('Galaxy counts')
        axes[1].set_title('NORM-conditioned latent correction')
        axes[0].legend(loc='lower left', fontsize=10, frameon=True)
        stem = OUT / f'cmd_norm_S_seed42_L15_{split}'
        fig.savefig(stem.with_suffix('.pdf'), dpi=400)
        fig.savefig(stem.with_suffix('.png'), dpi=400)
        plt.close(fig)
        red = d.gr > masters10_red_edge(d.Mr)
        metadata['samples'][split] = dict(rows=len(d), rows_in_binning_range=int(counts.sum()),
            displayed_mean_bins=int(np.isfinite(mean).sum()), pearson_S_colour=float(d.S.corr(d.gr)),
            mean_S_red=float(d.loc[red, 'S'].mean()), mean_S_blue=float(d.loc[~red, 'S'].mean()))
    (OUT / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    (OUT / 'caption.tex').write_text(r'''Colour--magnitude diagram for the matched test sample with valid catalogue photometry.
The left panel shows the logarithm of the number of spectra per bin; the right
panel shows the mean additive latent correction $S$ [dex] from the
validation-selected NORM-conditioned seed-42 equation at $L=15$.
Colours are calculated as $\mathrm{ABSMAG}_g-\mathrm{ABSMAG}_r$; absolute
$r$-band magnitudes are converted to the $H_0=70$ convention used in the existing CMD.
The mean correction is shown only in bins containing at least 20 spectra.
The diverging colour scale is centred on $S=0$ and saturates beyond $\pm0.2$ dex.
Dashed and dotted lines show the blue edge and ridge of the red sequence,
respectively, using the Masters et al. (2010) relations adopted in the existing figure.
$S$ is the additive contribution to predicted logarithmic total stellar mass,
not a direct measurement of the mass-to-light ratio.
''')
    (OUT / 'README.md').write_text('# NORM-conditioned latent correction on the CMD\n\n'
        'Primary figure: `cmd_norm_S_seed42_L15_test.pdf` (also 400 dpi PNG). '
        'The `_all` companion pools the existing train/validation/test splits for a denser descriptive plot. '
        'Neither plot is used to select the equation. See `caption.tex` for the suggested test-set caption.\n\n'
        'S uses the full-precision coefficients of the validation-selected equation; it is not normalised to unit variance. '
        'Rows are spectra, so repeated catalogue IDs are retained as in the SR sample. '
        'No new NORM threshold is applied. Photometry is the cached NYU VAGC join used by the existing paper CMD. '
        'The weights are verified against f(x,N,z)-f(0,N,z) on all saved rows. '
        'Raw colour associations are descriptive and are not controlled for NORM or redshift.\n\n'
        'Reproduce from the symbolic_regression directory: `../../../.venv/bin/python plot_cmd_norm_s.py`.\n')
    print(json.dumps(metadata['samples'], indent=2))


if __name__ == '__main__':
    main()
