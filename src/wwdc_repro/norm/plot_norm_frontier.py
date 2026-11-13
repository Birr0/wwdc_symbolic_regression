"""Figure-1-style matched NORM ablation: frozen validation selections only."""
from pathlib import Path
import argparse
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wwdc_repro.shared.figstyle import STYLE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('norm_ablation/selected_all_budgets.csv'))
    parser.add_argument('--outdir', type=Path, default=Path('norm_frontier'))
    args = parser.parse_args()
    data = pd.read_csv(args.input) # read the pareto front file

    assert not data.duplicated(['arm', 'with_norm', 'budget', 'seed']).any()
    assert set(data.arm) == {'original', 'norm'}
    assert set(data.with_norm) == {False, True}

    summary = []
    for (arm, with_norm, budget), group in data.groupby(['arm', 'with_norm', 'budget']):
        # loop through each arm
        assert set(group.seed) == {42, 43, 44}, 'Incomplete seed comparison' # check all seeds present for row
        assert (group.Length <= budget).all() # check budget is the same as the actual equation length
        valid = np.isfinite(group.test_r2).all() and group.test_status.eq('ok').all() # Make a gap for any seed that has nonfinite test
        summary.append(dict(arm=arm, with_norm=with_norm, budget=budget,
                            mean=group.test_r2.mean() if valid else np.nan,
                            low=group.test_r2.min() if valid else np.nan,
                            high=group.test_r2.max() if valid else np.nan,
                            finite_seeds=int(np.isfinite(group.test_r2).sum()))) # create summary for row
    summary = pd.DataFrame(summary) 
    args.outdir.mkdir(parents=True, exist_ok=True)
    plt.style.use(STYLE) # plot style
    fig, ax = plt.subplots(figsize=(7.2, 5.0), layout='constrained')
    styles = [
        ('original', False, '#0072B2', '--', 'Original + $z$'),
        ('original', True, '#0072B2', '-', 'Original + $N$ + $z$'),
        ('norm', False, '#D55E00', '--', 'normalisation-conditioned + $z$'),
        ('norm', True, '#D55E00', '-', 'normalisation-conditioned + $N$ + $z$'),
    ]
    for arm, with_norm, colour, linestyle, label in styles:
        c = summary[(summary.arm == arm) & (summary.with_norm == with_norm)].sort_values('budget')
        ax.fill_between(c.budget, c.low, c.high, color=colour, alpha=.18, linewidth=0)
        ax.plot(c.budget, c['mean'], color=colour, ls=linestyle, lw=1.6,
                marker='o', ms=2.7, label=label) # plot mean value across seeds for the given eq. budget
    ax.set(xlim=(4.2, 29.8), ylim=(0, 1), xlabel='Maximum equation length', ylabel=r'$R^2$')
    ax.xaxis.label.set_size(14); ax.yaxis.label.set_size(14)
    ax.tick_params(labelsize=12)
    ax.legend(loc='lower right', fontsize=10, frameon=False)
    fig.savefig(args.outdir/'norm_frontier.pdf')
    fig.savefig(args.outdir/'norm_frontier.png', dpi=400)
    plt.close(fig)
    summary.to_csv(args.outdir/'norm_frontier.csv', index=False)
    (args.outdir/'metadata.json').write_text(json.dumps(dict(
        input=str(args.input), input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        seeds=[42,43,44], selection=sorted(data.selection.unique().tolist()) if 'selection' in data else ['validation (archive)'],
        band='Minimum to maximum test R2 across all three seeds',
        predictors='All four arms retain raw redshift; only explicit NORM is ablated',
        test_failures='Shown as gaps; no test-based replacement',
        incomplete_groups=int((summary.finite_seeds!=3).sum())), indent=2)+'\n')

if __name__ == '__main__':
    main()
