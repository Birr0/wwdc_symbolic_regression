"""Export conditioning tables from the current run, without fitting any models."""
import argparse
from pathlib import Path
import pandas as pd
import sympy as sp
from wwdc_repro.shared.equations import to_sympy
from wwdc_repro.redshift.diagnostic_tables import write_table


def write(frame, outdir, name):
    """Write the same current-run values to CSV and LaTeX."""
    frame.to_csv(outdir / f'{name}.csv', index=False)
    escapes = {'_': r'\_', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
               '{': r'\{', '}': r'\}', '^': r'\textasciicircum{}',
               '~': r'\textasciitilde{}', '\\': r'\textbackslash{}'}
    def cell(value):
        text = f'{value:.6f}' if isinstance(value, float) else str(value)
        return ''.join(escapes.get(char, char) for char in text)
    heading = ' & '.join(cell(column) for column in frame.columns)
    rows = [' & '.join(cell(value) for value in row) + r' \\'
            for row in frame.itertuples(index=False, name=None)]
    write_table(outdir / f'{name}.tex', 'l' * len(frame.columns),
                cell(name.replace('_', ' ')), heading, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, required=True)
    out = parser.parse_args().outdir
    out.mkdir(parents=True, exist_ok=True)
    ablation = pd.read_csv('norm_ablation/summary.csv')
    write(ablation[ablation.budget.isin([15, 29])], out, 'norm_ablation')
    sr = pd.read_csv('fibre_mass/original_control_comparison/performance.csv')
    mlp = pd.read_csv('fibre_mass/mlp_sklearn/summary.csv').rename(columns=lambda c: c if c == 'arm' else f'mlp_{c}')
    write(sr.merge(mlp, on='arm', how='left', validate='many_to_one'), out, 'fibre_performance')
    linear = pd.read_csv('norm_probe/linear_results.csv').groupby('arm', as_index=False).agg(linear_r2=('test_r2', 'mean'))
    mlp = pd.read_csv('norm_probe/mlp_sklearn/summary.csv').rename(columns=lambda c: c if c == 'arm' else f'mlp_{c}')
    write(linear.merge(mlp, on='arm', validate='one_to_one'), out, 'norm_probes')
    selected = pd.read_csv('norm_analysis/validation_selected_final.csv')
    equations = selected[(selected.arm == 'norm_cond+z') & selected.budget.isin([15, 29])]
    write(equations[['seed', 'budget', 'Length', 'equation', 'test_r2']], out, 'norm_equations')
    # Fresh searches need not rediscover the historical analytic family. Export
    # exact symbolic terms in SR feature coordinates instead of assuming its form.
    terms = []
    for row in equations.itertuples():
        tree = sp.expand(to_sympy(row.equation, 12))
        for term, coefficient in tree.as_coefficients_dict().items():
            terms.append(dict(seed=row.seed, budget=row.budget, term=str(term), coefficient=str(coefficient)))
    write(pd.DataFrame(terms), out, 'norm_equation_terms')


if __name__ == '__main__':
    main()
