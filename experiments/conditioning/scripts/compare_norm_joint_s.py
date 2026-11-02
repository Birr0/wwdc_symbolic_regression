"""Compare additive L<=15 corrections on matched test spectra, three SR seeds."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import sympy as sp
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from figstyle import use_science
from cmd_helpers import H1_TO_H70, masters10_red_edge, masters10_ridge
from val_rescore import to_sympy, evaluate_equation
from matched_conditioning_controls import digest

BASE = Path('conditioning_validation_2026-10-13')
OUT = Path('norm_joint_s_comparison_2026-10-16')


def main():
    OUT.mkdir(exist_ok=True)
    sources = [BASE/'validation_selected_final.csv', Path('plot_cmd_latent_results/vagc_join.parquet')]
    selected = pd.read_csv(sources[0])
    ids = np.load(BASE/'norm_test_ids.npy').astype('int64')
    np.testing.assert_array_equal(ids, np.load(BASE/'znorm_test_ids.npy').astype('int64'))
    data = {}; equations = []
    for arm in ('norm', 'znorm'):
        path = BASE/f'{arm}_cond+z_matrices.npz'; sources.append(path)
        mats = np.load(path); data[arm] = mats
        for seed in (42, 43, 44):
            r = selected[(selected.arm == arm+'_cond+z') & (selected.seed == seed) & (selected.budget == 15)].iloc[0]
            expr = to_sympy(r.equation, 12)
            derivatives = [sp.diff(expr, sp.Symbol(f'X{i}')) for i in range(1, 11)]
            assert all(not d.free_symbols for d in derivatives)
            w = np.array([float(d) for d in derivatives])
            x = mats['X_test']; s = x[:, :10] @ w
            zero = x.copy(); zero[:, :10] = 0
            np.testing.assert_allclose(s, evaluate_equation(r.equation, x)-evaluate_equation(r.equation, zero), atol=1e-10, rtol=0)
            data[f'{arm}_{seed}'] = s
            equations.append(dict(arm=arm, seed=seed, Length=int(r.Length), equation=r.equation, weights=w.tolist()))
    for key in ('y_test',):
        np.testing.assert_array_equal(data['norm'][key], data['znorm'][key])
    np.testing.assert_array_equal(data['norm']['X_test'][:, 10:], data['znorm']['X_test'][:, 10:])
    df = pd.DataFrame({'row': np.arange(len(ids)), 'specObjID': ids,
                       'n': data['norm']['X_test'][:, 10], 'z': data['norm']['X_test'][:, 11]})
    for arm in ('norm', 'znorm'):
        for seed in (42, 43, 44): df[f'{arm}_{seed}'] = data[f'{arm}_{seed}']
    phot = pd.read_parquet(sources[1])[['specObjID', 'ABSMAG_g', 'ABSMAG_r']]
    df = df.merge(phot, on='specObjID', how='inner', validate='many_to_one')
    for col in ('ABSMAG_g', 'ABSMAG_r'):
        df = df[np.isfinite(df[col]) & (df[col] > -99) & (df[col] < 0)]
    df = df[(df.z > .02) & (df.z <= .3)].copy()
    df['gr'] = df.ABSMAG_g-df.ABSMAG_r; df['Mr'] = df.ABSMAG_r+H1_TO_H70
    # Descriptive within-cell centring, not a causal or exact conditional test.
    df['nbin'] = pd.qcut(df.n, 10, labels=False, duplicates='drop')
    df['zbin'] = pd.qcut(df.z, 10, labels=False, duplicates='drop')
    group = df.groupby(['nbin', 'zbin'], observed=True)
    keep = group.row.transform('size') >= 20
    cols = ['gr']+[f'{a}_{s}' for a in ('norm', 'znorm') for s in (42, 43, 44)]
    centred = (df[cols]-group[cols].transform('mean')).loc[keep]
    rows = []; red = df.gr > masters10_red_edge(df.Mr)
    for seed in (42, 43, 44):
        a, b = df[f'norm_{seed}'], df[f'znorm_{seed}']
        row = dict(seed=seed, rows=len(df), all_test_rows=len(ids),
            pearson_all_test=float(np.corrcoef(data[f'norm_{seed}'], data[f'znorm_{seed}'])[0, 1]),
            pearson=float(a.corr(b)), spearman=float(spearmanr(a, b).statistic),
            difference_rmse_dex=float(np.sqrt(np.mean((b-a)**2))),
            within_bin_pair_pearson=float(centred[f'norm_{seed}'].corr(centred[f'znorm_{seed}'])))
        for arm in ('norm', 'znorm'):
            s = df[f'{arm}_{seed}']
            row.update({f'{arm}_colour_pearson': float(s.corr(df.gr)),
                f'{arm}_within_bin_colour_pearson': float(centred[f'{arm}_{seed}'].corr(centred.gr)),
                f'{arm}_red_mean': float(s[red].mean()), f'{arm}_blue_mean': float(s[~red].mean()),
                f'{arm}_sd': float(s.std())})
        rows.append(row)
    result = pd.DataFrame(rows); result.to_csv(OUT/'comparison.csv', index=False)
    df.to_parquet(OUT/'matched_test_data.parquet', index=False)
    use_science(); plt.rcParams.update({'font.size': 13, 'axes.labelsize': 16})
    xe = np.linspace(-24, -18, 61); ye = np.linspace(.15, 1.2, 57)
    counts, _, _ = np.histogram2d(df.Mr, df.gr, bins=(xe, ye))
    bins = {'counts': counts, 'magnitude_edges': xe, 'colour_edges': ye}
    fig, axes = plt.subplots(3, 2, figsize=(11, 12), sharex=True, sharey=True, layout='constrained')
    for i, seed in enumerate((42, 43, 44)):
        for j, arm in enumerate(('norm', 'znorm')):
            total, _, _ = np.histogram2d(df.Mr, df.gr, bins=(xe, ye), weights=df[f'{arm}_{seed}'])
            mean = np.divide(total, counts, out=np.full_like(total, np.nan), where=counts>=20)
            bins[f'{arm}_{seed}'] = mean
            ax = axes[i, j]
            im = ax.pcolormesh(xe, ye, mean.T, cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-.2, vcenter=0, vmax=.2), rasterized=True)
            grid = np.linspace(-24, -18, 200)
            ax.plot(grid, masters10_red_edge(grid), '--', color='.2', lw=1)
            ax.plot(grid, masters10_ridge(grid), ':', color='.4', lw=1)
            ax.set(xlim=(-18.1, -23.9), ylim=(.2, .9), title=('NORM' if arm=='norm' else 'NORM + z')+f' conditioning: seed {seed}')
            if j == 0: ax.set_ylabel(r'$g-r$ [mag]')
            if i == 2: ax.set_xlabel(r'$M_r$ [mag]')
    fig.colorbar(im, ax=axes, pad=.02, shrink=.7, extend='both', label=r'$\langle S\rangle$ [dex]')
    fig.savefig(OUT/'cmd_comparison_three_seeds.pdf', dpi=400)
    fig.savefig(OUT/'cmd_comparison_three_seeds.png', dpi=300)
    plt.close(fig); np.savez(OUT/'cmd_bins.npz', **bins)
    meta = dict(equations=equations, rows=len(df), centred_rows=int(keep.sum()),
        selection='Existing validation-selected equations at budget 15, no new selection.',
        checks='Exact ID order, targets and NORM/z equality; constant latent derivatives; S equals f(x,N,z)-f(0,N,z).',
        input_sha256={str(p): digest(p) for p in sources})
    (OUT/'metadata.json').write_text(json.dumps(meta, indent=2))
    lines = ['# NORM versus joint NORM+z: additive latent corrections', '',
        f'Matched test spectra with valid CMD photometry: {len(df):,}. Same filtering, bins and colour scale as the NORM CMD. '
        'Validation-selected equations at length budget 15; full-precision weights. S is in dex and is not renormalised. '
        'All six equations have constant linear latent coefficients. Neither equations nor signs are selected to maximise agreement.', '',
        '| Seed | Pearson S–S | Spearman S–S | Difference RMSE [dex] | Colour r: NORM / joint | Within-bin colour r: NORM / joint |',
        '|---|---:|---:|---:|---:|---:|']
    for r in result.itertuples():
        lines.append(f'| {r.seed} | {r.pearson:.4f} | {r.spearman:.4f} | {r.difference_rmse_dex:.4f} | {r.norm_colour_pearson:.4f} / {r.znorm_colour_pearson:.4f} | {r.norm_within_bin_colour_pearson:.4f} / {r.znorm_within_bin_colour_pearson:.4f} |')
    lines += ['', 'Within-bin correlations subtract each variable’s mean in 10×10 quantile cells of NORM and redshift, '
        'retaining cells with at least 20 spectra. These are descriptive controls for broad NORM/redshift trends, '
        'not exact conditioning or a causal test. Repeated spectrum IDs are retained as in the SR sample. '
        'The three seeds are SR searches, not independent flow trainings. The correction need not uniquely represent mass-to-light ratio.', '',
        'Plot: `cmd_comparison_three_seeds.pdf` (also PNG). Full measurements: `comparison.csv`; '
        'equations/provenance: `metadata.json`; matched plotting data: `matched_test_data.parquet`.', '',
        'Reproduce from the symbolic_regression directory: `../../../.venv/bin/python compare_norm_joint_s.py`.', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print(result.to_string(index=False))


if __name__ == '__main__': main()
