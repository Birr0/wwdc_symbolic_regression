"""Corner plot of paired original/conditional test embeddings, without rescaling."""
import json
from pathlib import Path

import corner
import matplotlib.pyplot as plt
import scienceplots  # Registers the Matplotlib styles.
from matplotlib.lines import Line2D
import numpy as np

NDIM = 10
RANGE = (-5, 5)
BINS = 80
SMOOTH = 1.0
LEVELS = (.68, .95)


@plt.style.context(["science", "no-latex"])
def latent_corner(orig: np.ndarray, cond: np.ndarray, outdir: Path) -> None:
    """Plot paired original and conditional latent distributions.

    Overlay one-dimensional density histograms on the diagonal and smoothed
    two-dimensional density contours below it. Contours enclose approximately
    68% and 95% of the histogram probability within the displayed range.
    Diagonal panels also show the standard-normal density for comparison.

    Parameters
    ----------
    orig : np.ndarray
        Original latent embeddings of shape (n_samples, NDIM), with finite values.
    cond : np.ndarray
        Conditional embeddings with the same shape and corresponding sample rows.
        The caller supplies the test sample.
    outdir : Path
        Existing directory in which to save latent_corner.pdf, latent_corner.png,
        and latent_corner.json. The JSON records plotting settings, per-dimension
        statistics, pairwise correlations, and fractions inside/outside the range.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If embedding shapes differ, are not two-dimensional with NDIM columns,
        or contain nonfinite values.

    Notes
    -----
    Histogram densities are normalized within the range specified by 'RANGE'. 
    """
    if orig.shape != cond.shape or orig.ndim != 2 or orig.shape[1] != NDIM:
        raise ValueError('Expected paired ten-dimensional embedding arrays')
    if not (np.isfinite(orig).all() and np.isfinite(cond).all()):
        raise ValueError('Nonfinite embeddings')
    colours = {'original': '#1261a6', 'conditional': '#249754'}
    data = {'original': orig, 'conditional': cond}
    fig = plt.figure(figsize=(19, 19))
    for name, x in data.items():
        # corner sorts the density thresholds ascending, so the 95% contour is drawn first.
        corner.corner(
            x, fig=fig, bins=BINS, range=[RANGE] * NDIM, smooth=SMOOTH, levels=LEVELS,
            color=colours[name], labels=[rf'$x_{{{k+1}}}$' for k in range(NDIM)],
            label_kwargs={'fontsize': 22}, plot_datapoints=False, plot_density=False,
            no_fill_contours=True, fill_contours=False, max_n_ticks=3,
            hist_kwargs={'density': True, 'lw': 1.5},
            contour_kwargs={'linewidths': [1.0, 1.5], 'linestyles': ['--', '-']},
        )
    axes = np.array(fig.axes).reshape(NDIM, NDIM)
    fig.subplots_adjust(left=.065, right=.985, bottom=.06, top=.985, wspace=.06, hspace=.06)
    grid = np.linspace(*RANGE, 600)
    edges = np.linspace(*RANGE, BINS + 1)
    report = {'split': 'test', 'rows': len(orig), 'range': list(RANGE), 'bins': BINS,
              'smoothing_sigma_bins': SMOOTH, 'contour_probabilities': list(LEVELS),
              'plotting': f'corner {corner.__version__} (corner.corner / hist2d)',
              'contour_definition': 'Highest-density thresholds of Gaussian-smoothed 2D histograms (corner.hist2d); probability normalized within displayed range.',
              'sample': 'Exact catalogue-matched paper test sample after mask/redshift cuts and valid stellar-mass target selection; IDs checked against paper reference', 'rescaling': 'none',
              'marginal_density_normalization': 'rows within displayed range',
              'dimensions': [], 'pairs': []}
    for i in range(NDIM):
        dim = {'dimension': i+1}
        for name, x in data.items():
            dim[name] = {'mean': float(x[:, i].mean()), 'sd': float(x[:, i].std()),
                         'fraction_outside_range': float(np.mean(np.abs(x[:, i]) > RANGE[1]))}
        report['dimensions'].append(dim)
        # Set this diagonal panel's y-limit from the larger of its two histogram peaks.
        peak = max(np.histogram(x[:, i], bins=edges, density=True)[0].max() for x in data.values())
        ax = axes[i, i]
        ax.plot(grid, np.exp(-grid**2/2)/np.sqrt(2*np.pi), color='#bd302d', ls='--', lw=1.3) # Overlay the standard-normal probability density.
        ax.set_ylim(0, 1.05 * peak)
        for j in range(i):
            report['pairs'].append({'x_dimension': j+1, 'y_dimension': i+1, **{
                name: {'fraction_in_range': float(np.mean((np.abs(x[:, j]) <= RANGE[1]) & (np.abs(x[:, i]) <= RANGE[1]))),
                       'correlation_all_rows': float(np.corrcoef(x[:, j], x[:, i])[0, 1])}
                for name, x in data.items()}})
    for ax in axes[np.tril_indices(NDIM)]:
        ax.set_xticks([-4, 0, 4])
        ax.tick_params(axis='both', labelsize=14, direction='in', length=3, labelrotation=0)
    for i in range(1, NDIM):
        for ax in axes[i, :i]:
            ax.set_yticks([-4, 0, 4])
    for name, title, y in [('original', 'Original Spender latents', .90),
                           ('conditional', 'Conditional latents', .79)]:
        handles = [Line2D([], [], color=colours[name], lw=2, ls='-',
                          label='68% density contour'),
                   Line2D([], [], color=colours[name], lw=1.5, ls='--',
                          label='95% density contour')]
        fig.legend(handles=handles, title=title, title_fontsize=22,
                   loc='upper left', bbox_to_anchor=(.65, y), fontsize=20,
                   frameon=False, labelspacing=.6)
    fig.legend(handles=[Line2D([], [], color='#bd302d', ls='--', lw=2,
                               label=r'$\mathcal{N}(0,1)$ (diagonal)')],
               loc='upper left', bbox_to_anchor=(.65, .68), fontsize=20,
               frameon=False)
    stem = outdir/'latent_corner'
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(stem.with_suffix('.png'), dpi=450, bbox_inches='tight')
    plt.close(fig)
    stem.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
