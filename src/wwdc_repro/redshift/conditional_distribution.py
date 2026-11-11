"""Optional check of the baseline + A(z)s interpretation within redshift bins.

Uses actual test galaxies in each bin; neither centres nor rescales residuals
using their observed moments. No manuscript figures or tables are replaced.
"""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import sympy as sp

from wwdc_repro.shared.data import clean_target, load_sample, matrices, sample_summary
from wwdc_repro.shared.equations import evaluate, first_order_latent
from wwdc_repro.redshift.figures import selected_l24
from wwdc_repro.redshift.frontier import SEEDS
from wwdc_repro.redshift.weights import raw_coefficients
from wwdc_repro.shared.utils import sha256


def run(frames, selected, outdir):
    """Save conditional moments, coverage, Q–Q curves and model components.

    A(z) is taken from exact first-order latent coefficients along the
    sign-reversed, unit-normalised raw vector, matching anatomy. Each galaxy
    uses its own baseline and A(z), avoiding within-bin redshift trends.
    Ten equal-count bins are the primary diagnostic; 14 equal-width bins
    over the observed redshift range provide a sensitivity check.
    """
    x = matrices(frames, 'cond+z')['X_test']
    z = x[:, -1]
    schemes = {'equal_count': np.quantile(z, np.linspace(0, 1, 11)),
               'equal_width': np.linspace(z.min(), z.max(), 15)}
    records, qq, galaxy, equations = [], [], [], []
    probabilities = np.linspace(.005, .995, 199)
    theoretical = stats.norm.ppf(probabilities)
    for seed in SEEDS:
        equation = selected_l24(selected, seed)
        first, symbols = first_order_latent(equation, x.shape[1])
        raw = raw_coefficients(equation)
        direction = -raw / np.linalg.norm(raw)
        coeff = np.column_stack([np.broadcast_to(
            np.asarray(sp.lambdify(symbols[-1], first.coeff(s), 'numpy')(z), dtype=float), z.shape)
            for s in symbols[:-1]])
        amplitude = coeff @ direction
        np.testing.assert_allclose(coeff, amplitude[:, None]*direction, rtol=1e-9, atol=1e-11)
        if np.any(amplitude <= 0):
            raise ValueError('Expected positive signed amplitude over the sample')
        zero = x.copy(); zero[:, :-1] = 0
        baseline = evaluate(equation, zero)
        full = evaluate(equation, x)
        approx = evaluate(str(first), x)
        score = x[:, :-1] @ direction
        np.testing.assert_allclose(approx, baseline + amplitude*score, rtol=1e-11, atol=1e-11)
        equations.append({'seed': seed, 'equation': equation, 'unit_direction': direction.tolist()})
        u_full = (full-baseline)/amplitude
        u_first = (approx-baseline)/amplitude
        if not np.isfinite(u_full).all():
            raise ValueError('Nonfinite standardised predictions')
        galaxy.append(pd.DataFrame({'seed': seed, 'specObjID': frames['test'].specObjID.to_numpy(),
            'z': z, 'baseline': baseline, 'amplitude': amplitude, 'score': score,
            'prediction_full': full, 'prediction_first': approx, 'u_full': u_full, 'u_first': u_first}))
        for scheme, edges in schemes.items():
            bins = np.clip(np.searchsorted(edges, z, side='right')-1, 0, len(edges)-2)
            assert sum((bins==i).sum() for i in range(len(edges)-1)) == len(x)
            for i in range(len(edges)-1):
                mask = bins == i
                n = int(mask.sum())
                if n < 2:
                    continue
                for model, prediction, u in [('full', full, u_full), ('first_order', approx, u_first)]:
                    values = u[mask]
                    residual = prediction[mask]-baseline[mask]
                    row = {'seed': seed, 'scheme': scheme, 'bin': i, 'model': model, 'n': n,
                        'z_lo': edges[i], 'z_hi': edges[i+1], 'z_median': np.median(z[mask]),
                        'mean_u': values.mean(), 'sd_u': values.std(ddof=0),
                        'mean_residual_dex': residual.mean(), 'sd_residual_dex': residual.std(),
                        'mean_amplitude_dex': amplitude[mask].mean(),
                        'sd_prediction_dex': prediction[mask].std(),
                        'skewness': stats.skew(values, bias=False),
                        'excess_kurtosis': stats.kurtosis(values, bias=False),
                        'coverage_1': np.mean(np.abs(values)<1), 'coverage_2': np.mean(np.abs(values)<2),
                        'mean_u_se': values.std(ddof=1)/np.sqrt(n)}
                    records.append(row)
                    if scheme == 'equal_count':
                        for p, normal, observed in zip(probabilities, theoretical, np.quantile(values, probabilities)):
                            qq.append({'seed': seed, 'bin': i, 'model': model, 'probability': p,
                                       'normal_quantile': normal, 'observed_quantile': observed})
    df = pd.DataFrame(records); qdf = pd.DataFrame(qq)
    df.to_csv(outdir/'conditional_summary.csv', index=False)
    qdf.to_csv(outdir/'qq_quantiles.csv', index=False)
    pd.concat(galaxy).to_csv(outdir/'predictions.csv.gz', index=False)
    for scheme in schemes:
        fig, axes = plt.subplots(3, 4, figsize=(15, 10), sharex='col')
        for axrow, seed in zip(axes, SEEDS):
            for model, style in [('full', '-'), ('first_order', '--')]:
                sub = df[(df.seed==seed)&(df.scheme==scheme)&(df.model==model)]
                for ax, key, target, title in zip(axrow,
                    ['mean_u','sd_u','coverage_1','coverage_2'], [0,1,stats.norm.cdf(1)-stats.norm.cdf(-1),stats.norm.cdf(2)-stats.norm.cdf(-2)],
                    ['Mean u (target 0)','SD u (target 1)','Fraction |u| < 1','Fraction |u| < 2']):
                    ax.plot(sub.z_median, sub[key], style, marker='.', label=model)
                    if model=='full': ax.axhline(target, color='grey', linewidth=.8)
                    ax.set_title(f'Seed {seed}: {title}'); ax.set_xlabel('Redshift'); ax.grid(alpha=.2)
            axrow[0].legend()
        fig.suptitle('u = (predicted mass − zero-latent baseline) / A(z); ' + scheme.replace('_',' ') + ' bins')
        fig.tight_layout()
        for suffix in ('png','pdf'): fig.savefig(outdir/f'conditional_moments_{scheme}.{suffix}', dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(10, 13))
    cmap = plt.get_cmap('viridis', 10)
    for axrow, seed in zip(axes, SEEDS):
        for ax, model in zip(axrow, ['full','first_order']):
            for i in range(10):
                sub=qdf[(qdf.seed==seed)&(qdf.model==model)&(qdf.bin==i)]
                ax.plot(sub.normal_quantile,sub.observed_quantile,color=cmap(i),label=f'bin {i+1}',linewidth=1)
            ax.plot([-3,3],[-3,3],'k--',linewidth=1)
            ax.set_title(f'Seed {seed}: {model}'); ax.set_xlabel('Standard-normal quantile'); ax.set_ylabel('Observed u quantile')
    axes[0,1].legend(fontsize=7,ncol=2)
    fig.tight_layout()
    for suffix in ('png','pdf'): fig.savefig(outdir/f'conditional_qq.{suffix}',dpi=180)
    plt.close(fig)
    (outdir/'method.json').write_text(json.dumps({'sample': sample_summary(frames), 'equations': equations,
        'bin_edges': {k:v.tolist() for k,v in schemes.items()},
        'method': 'Actual galaxies per bin. Per-row baseline and exact first-order amplitude. No empirical recentering or rescaling. Population SD. Strict coverage bounds.',
        'qq_probability_range': [.005,.995], 'reference': 'N(0,1); descriptive assessment, no fitted normal',
        'interpretation': 'Raw prediction SD also includes baseline evolution across a finite-width bin; sd_u is the direct scale test. Small equal-width bins have noisy moments.'},indent=2)+'\n')
    print(df.groupby(['scheme','model'])[['mean_u','sd_u','coverage_1','coverage_2','skewness','excess_kurtosis']].agg(['min','max']).to_string())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('embeddings','galspec','specgals-home','selected','outdir'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    args.outdir.mkdir(parents=True,exist_ok=False)
    frames=clean_target(load_sample(args.embeddings,args.galspec,args.specgals_home))
    reference=json.loads((Path(__file__).parents[1]/'resources/reference.json').read_text())
    for row in sample_summary(frames):
        for key in ('rows','id_sha256','ordered_id_sha256'):
            if row[key]!=reference['samples'][row['split']][key]: raise ValueError('Sample mismatch')
    run(frames,pd.read_csv(args.selected),args.outdir)
    paths=[args.selected,args.galspec,*[args.specgals_home/f'SDSSspecgalsDR8_{i}.fit.gz' for i in (1,2)],
           *sorted(args.embeddings.glob('*/*.parquet')),*sorted(Path(__file__).parents[1].rglob('*.py'))]
    (args.outdir/'manifest.json').write_text(json.dumps([{'path':str(p.resolve()),'sha256':sha256(p)} for p in paths],indent=2)+'\n')


if __name__=='__main__':
    main()
