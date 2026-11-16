"""Export figure files and manuscript-formatted tables from saved results into the text folder.
"""
import argparse
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
import sympy as sp

from wwdc_repro.shared.equations import to_sympy


def appendix_table_layout(content):
    """Keep an appendix table in place with a full-width caption.

    Parameters
    ----------
    content : str
        A deluxetable's LaTeX source, including its caption and cells.

    Returns
    -------
    str
        An ordinary table using the float package's H placement. Caption
        wording and cell contents are preserved.
    """
    content = re.sub(r"\\tablewidth\{[^\n]*\}\n", "", content)
    content = re.sub(
        r"\\begin\{deluxetable\}\{([^}]+)\}",
        lambda match: "\\begin{table}[H]\n\\centering\n"
        + "% TABULAR_COLUMNS=" + match[1], content,
    )
    columns = re.search(r"% TABULAR_COLUMNS=([^\n]+)\n", content)
    if columns is None:
        raise ValueError("Expected a deluxetable column specification")
    content = content.replace(columns[0], "")
    content = content.replace(r"\tablecaption", r"\caption")
    content = content.replace(r"\tablehead{", "\\begin{tabular}{" + columns[1] + "}\n\\toprule\n", 1)
    content = content.replace("}\n\\startdata", "\\\\\n\\midrule", 1)
    content = content.replace(r"\colhead", r"\textnormal")
    content = content.replace(r"\enddata", "\\bottomrule\n\\end{tabular}")
    return content.replace(r"\end{deluxetable}", r"\end{table}")


def table(columns, heading, rows, caption, label):
    """Build the LaTeX source for a complete table float.

    Parameters
    ----------
    columns : str
        Tabular column specification, for example "lrrr".
    heading : str
        LaTeX header cells separated by ampersands, without a row terminator.
    rows : iterable of sequences of str
        Formatted LaTeX cells for each data row.
    caption : str
        LaTeX caption text.
    label : str
        Reference key, for example "tab:norm_ablation".

    Returns
    -------
    str
        Table source ready to write to a .tex file. Inputs are inserted as
        LaTeX without escaping; numeric formatting is the caller's responsibility.
    """
    return (
        "\\begin{table}[htbp]\n\\centering\\small\n"
        f"\\begin{{tabular}}{{{columns}}}\n\\toprule\n"
        + heading + r"\\" + "\n\\midrule\n"
        + "\n".join(" & ".join(row) + r"\\" for row in rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
        + f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n"
    )


def norm_family_coefficients(equation, mean, std):
    """Extract physical-unit coefficients of the additive NORM equation family.

    Parameters
    ----------
    equation : str
        Precise SR export with standardized log-normalisation in X11 and raw
        redshift in X12. Latents X1--X10 retain their standardized coordinates.
    mean, std : float
        Mean and population standard deviation of log10(N/N0) on cleaned
        training rows, so X11 = (ell - mean) / std.

    Returns
    -------
    dict
        C, A, b, D, k, a_5, a_6 and a_9 in
        C + A*exp(b*ell) + D*log(1+k*z) + a_5*X5 + a_6*X6 + a_9*X9.

    Raises
    ------
    ValueError
        Scaling is invalid or the equation does not have this exact family.
        A different discovered form must be reviewed rather than coerced.
    """
    if not np.isfinite([mean, std]).all() or std <= 0:
        raise ValueError("NORM scaling must be finite with positive standard deviation")
    symbols = sp.symbols("X1:13")
    ell, z = sp.symbols("ell z")
    expression = sp.expand(to_sympy(equation, 12).subs(
        {symbols[10]: (ell - mean) / std, symbols[11]: z}))
    exponentials, logarithms = expression.atoms(sp.exp), expression.atoms(sp.log)
    if len(exponentials) != 1 or len(logarithms) != 1:
        raise ValueError("Expected one exponential and one logarithm in the NORM family")
    exponential, logarithm = next(iter(exponentials)), next(iter(logarithms))
    b = sp.diff(exponential.args[0], ell)
    k = sp.diff(logarithm.args[0], z)
    if (sp.simplify(exponential.args[0] - b * ell) != 0 or
            sp.simplify(logarithm.args[0] - (1 + k * z)) != 0):
        raise ValueError("NORM exponential or logarithm differs from the manuscript family")
    values = dict(A=expression.coeff(exponential), b=b,
                  D=expression.coeff(logarithm), k=k)
    latent = sum(expression.coeff(symbols[i-1]) * symbols[i-1] for i in (5, 6, 9))
    values['C'] = sp.simplify(expression - values['A'] * exponential
                              - values['D'] * logarithm - latent)
    values.update({f'a_{i}': expression.coeff(symbols[i-1]) for i in (5, 6, 9)})
    if any(v.free_symbols or not v.is_real or not v.is_finite for v in values.values()):
        raise ValueError("NORM equation has additional or coupled terms outside the manuscript family")
    return {key: float(value) for key, value in values.items()}


def norm_family_table(selected, work):
    """Generate the three-seed coefficient table using saved training scaling.

    Parameters
    ----------
    selected : pandas.DataFrame
        Validation-selected ablation results, including equations and scores.
    work : pathlib.Path
        Conditioning work directory. NORM-probe targets store log10(N/N0);
        SR matrices store its standardized form. Their alignment is checked
        on all splits before extracting the budget-15 coefficients.

    Returns
    -------
    str
        Complete LaTeX table with the existing tab:norm_family_values label.
    """
    chosen = selected[(selected.arm == 'norm') & selected.with_norm
                      & (selected.budget == 15)].sort_values('seed')
    if chosen.seed.tolist() != [42, 43, 44] or not chosen.selection.eq('validation').all():
        raise ValueError("Expected one validation-selected NORM equation for each seed")
    with np.load(work / 'norm_probe/norm_matrices.npz') as probe, np.load(
            work / 'norm_analysis/norm_cond+z_matrices.npz') as matrices:
        mean, std = probe['y_train'].mean(), probe['y_train'].std()
        for split in ('train', 'val', 'test'):
            np.testing.assert_allclose(probe[f'X_{split}'], matrices[f'X_{split}'][:, :10], rtol=0, atol=0)
            np.testing.assert_allclose((probe[f'y_{split}'] - mean) / std,
                                       matrices[f'X_{split}'][:, 10], rtol=1e-12, atol=1e-12)
    coefficients = [norm_family_coefficients(row.equation, mean, std)
                    for row in chosen.itertuples()]
    rows = []
    for key in ('C', 'A', 'b', 'D', 'k', 'a_5', 'a_6', 'a_9'):
        precision = 2 if key == 'k' else 5
        rows.append([f'${key}$', *[f'{values[key]:.{precision}f}' for values in coefficients]])
    if not np.isfinite(chosen.test_r2).all():
        raise ValueError("Nonfinite test score in the NORM coefficient table")
    rows.append([r'Test $R^2$', *[f'{value:.5f}' for value in chosen.test_r2]])
    return table('lrrr', ' & Seed 42 & Seed 43 & Seed 44', rows,
                 r'Rounded coefficients of Equation~\ref{eq:norm_family} for the three '
                 r'normalisation-conditioned fits selected at maximum length $L=15$.',
                 'tab:norm_family_values')


def export(fresh, validation, project, mlp_run=None, mlp_runs=None):
    """Export saved figures and generated tables into an Overleaf project.

    Parameters
    ----------
    fresh : pathlib.Path
        Fresh-run directory containing work/conditioning, work/redshift and
        figures. Supplies probe results and figures reusable across selections.
    validation : pathlib.Path or None
        Validation-report directory containing analysis, tables and figures.
        Supplies validation-selected SR results and frontier plots. Use None
        to read all products from the single validation-selected fresh run.
    project : pathlib.Path
        Destination project. Creates figures/ and tables/ as needed and
        overwrites the exported assets at their established filenames.
    mlp_run : pathlib.Path or None
        Optional separately completed stellar-mass MLP benchmark. Otherwise
        use fresh/work/stellar_mass_mlp when present.
    mlp_runs : sequence of pathlib.Path or None
        Completed benchmarks to combine into the full-grid table. Mutually
        exclusive with mlp_run. Use all three spectral-mode runs for the paper.

    Returns
    -------
    None
        Writes figures, table files and numerical LaTeX macros, then prints
        the number of exported files. Does not fit models or edit draft.tex.

    Raises
    ------
    ValueError
        Selection metadata is not validation, fibre-comparison values are
        nonfinite, or the equation used by the reused NORM CMD has changed.
    AssertionError
        Redshift budget-24 equations differ between the two selections.
    FileNotFoundError
        A required input file is missing.

    Notes
    -----
    The physical-unit NORM table is regenerated only if the equations match
    its stated family. Scientific prose still requires separate review.
    """
    single_run = validation is None
    if single_run:
        validation = fresh
    work = fresh / "work/conditioning"
    redshift_analysis = (fresh / "work/redshift/analysis" if single_run
                         else validation / "analysis/redshift")
    norm_analysis = (work / "norm_ablation" if single_run
                     else validation / "analysis/normalisation")
    tables = validation / "tables/conditioning"
    ablation = pd.read_csv(norm_analysis / "summary.csv")
    fibre = pd.read_csv(tables / "fibre_performance.csv")
    probes = pd.read_csv(tables / "norm_probes.csv").set_index("arm")
    correlations = pd.read_csv(work / "norm_probe/coordinate_correlations.csv")
    for frame in (ablation, fibre):
        if not frame.selection.eq("validation").all():
            raise ValueError("Manuscript tables require validation-selected results")
    for path in (redshift_analysis / "selected.csv",
                 norm_analysis / "selected_all_budgets.csv"):
        if not pd.read_csv(path).selection.eq("validation").all():
            raise ValueError(f"Expected validation selection in {path}")

    labels = {"original": "Original", "norm": "Normalisation-conditioned",
              "zflow": "$z$-conditioned"}
    rows = []
    for arm in ("original", "norm", "zflow"):
        r = probes.loc[arm]
        c = correlations[correlations.arm == arm]
        rows.append([labels[arm], f"{c.pearson.abs().max():.3f}",
                     f"{c.spearman.abs().max():.3f}", f"{r.linear_r2:.3f}",
                     f"{r.mlp_test_r2:.3f} ({r.mlp_test_r2_min:.3f}--{r.mlp_test_r2_max:.3f})"])
    rendered = {"norm_probes.tex": table(
        "lrrrr", r"Representation & Max $|r|$ & Max $|\rho|$ & Linear $R^2$ & MLP $R^2$ (range)", rows,
        r"Prediction of $\log_{10}(N/N_0)$ from the ten latent coordinates without appended normalisation or redshift. "
        r"Correlations are the largest absolute single-latent Pearson $r$ and Spearman $\rho$ on the test sample. "
        r"Linear and MLP $R^2$ values are the mean values across seeds 42--44; parentheses are the MLP minimum and maximum across the seeds. "
        r"MLP checkpoints are selected using validation loss and evaluated on the test set.", "tab:norm_probes")}

    rows = []
    for arm in ("original", "norm"):
        for budget in (15, 29):
            values = ablation[(ablation.arm == arm) & (ablation.budget == budget)].set_index("with_norm")
            without, with_norm = values.loc[False, "test_r2"], values.loc[True, "test_r2"]
            rows.append([labels[arm], str(budget), f"{without:.3f}", f"{with_norm:.3f}", f"{with_norm-without:+.3f}"])
    rendered["norm_ablation.tex"] = table(
        "lrrrr", r"Representation & $L$ & $R^2$ without $N$ & $R^2$ with $N$ & $\Delta R^2$", rows,
        r"Mean test $R^2$ across seeds 42--44 for total-mass prediction, with and without the normalisation included. "
        r"Every fit includes redshift as an input variable. Equations are selected by validation performance within maximum length $L$.",
         "tab:norm_ablation")

    rows = []
    for arm in ("original", "norm"):
        values = fibre[fibre.arm == arm].set_index("budget")
        rows.append([labels[arm], f"{values.loc[15, 'test_r2']:.4f}",
                     f"{values.loc[29, 'test_r2']:.4f}", f"{values.loc[29, 'mlp_test_r2']:.4f}"])
    caps = int(fibre[(fibre.arm.isin(["original", "norm"])) & (fibre.budget == 29)].mlp_capped_runs.sum())
    rendered["fibre_performance.tex"] = table(
        "lrrr", r"Representation & SR, $L\leq15$ & SR, $L\leq29$ & MLP", rows,
        r"Mean test $R^2$ for fibre-mass prediction across seeds 42--44. All models receive the specified representation, "
        r"normalisation and redshift, and use the same 10,000-spectrum training subset per seed. "
        r"Symbolic regression equations are selected by validation performance within each length budget. "
        r"The MLP has hidden layers of 64 and 32 with ReLU activations and uses Adam with learning rate "
        r"$10^{-3}$ and batch size 1024. Training uses at most 500 epochs, patience 40, and the minimum-validation-loss checkpoint. "
        + f"{caps} of the six MLP runs reached the epoch limit.", "tab:fibre_performance")

    original = fibre[(fibre.arm == "original") & (fibre.budget == 29)].iloc[0]
    norm = fibre[(fibre.arm == "norm") & (fibre.budget == 29)].iloc[0]
    numbers = dict(FibreNormSR=norm.test_r2, FibreNormMLP=norm.mlp_test_r2,
                   FibreSRGap=original.test_r2-norm.test_r2,
                   FibreMLPGap=original.mlp_test_r2-norm.mlp_test_r2)
    if not np.isfinite(list(numbers.values())).all():
        raise ValueError("Nonfinite fibre comparison; prose must not report a partial-seed mean")
    rendered["result_numbers.tex"] = "% Generated numerical values used in manuscript prose.\n" + "".join(
        f"\\newcommand{{\\{key}}}{{{value:.4f}}}\n" for key, value in numbers.items())

    copies = {}
    for src, dest in {"coverage_table.tex": "coverage.tex", "truncation_table.tex": "truncation.tex",
                      "latent_weights.tex": "weights.tex", "front_condz.tex": "front_condz.tex"}.items():
        copies[project / "tables" / dest] = validation / "tables/redshift" / src
    for src, dest in {"frontier_lgm_tot_p50.pdf": "fig1_frontier.pdf",
                      "frontier_lgm_tot_p50_by_seed.pdf": "frontier_lgm_tot_p50_by_seed.pdf"}.items():
        copies[project / "figures" / dest] = validation / "figures/redshift" / src
    for src, dest in {"zero_latent.pdf": "zero_latent.pdf", "equation_anatomy_by_seed.pdf": "equation_anatomy_by_seed.pdf",
                      "cmd_latent_all_gr_seed42.pdf": "cmd_latent_all_gr_seed42.pdf", "latent_corner.pdf": "latent_corner.pdf"}.items():
        copies[project / "figures" / dest] = fresh / "figures/redshift" / src
    copies[project / "figures/norm_frontier.pdf"] = validation / "figures/normalisation/norm_frontier.pdf"
    for src, dest in {"norm_cmd.pdf": "norm_cmd_full.pdf", "norm_conditional_sequence.pdf": "norm_sequence.pdf"}.items():
        copies[project / "figures" / dest] = fresh / "figures/normalisation" / src
    if single_run:
        copies[project / "figures/norm_cmd_full.pdf"] = work / "norm_cmd/norm_cmd_full.pdf"
    # When reusing figures from a separate run, require unchanged equations.
    training = pd.read_csv(fresh / "work/redshift/analysis/selected.csv")
    selected = pd.read_csv(redshift_analysis / "selected.csv")
    def equations(frame):
        """Return cond+z budget-24 equation strings indexed and sorted by seed."""
        return frame[(frame.arm == "cond+z") & (frame.budget == 24)].set_index("seed").equation.sort_index()
    pd.testing.assert_series_equal(equations(training), equations(selected))
    training_norm = pd.read_csv(fresh / "work/conditioning/norm_ablation/selected_all_budgets.csv")
    selected_norm = pd.read_csv(norm_analysis / "selected_all_budgets.csv")
    def norm_cmd_equation(frame):
        """Return seed-42 NORM+z equation strings at budget 15 with NORM appended."""
        return frame[(frame.arm == "norm") & frame.with_norm & (frame.seed == 42)
                     & (frame.budget == 15)].equation.to_list()
    if norm_cmd_equation(training_norm) != norm_cmd_equation(selected_norm):
        raise ValueError("NORM CMD equation changed; regenerate that figure before exporting")
    rendered['norm_family_values.tex'] = norm_family_table(selected_norm, work)
    # Optional for older runs; fresh equation runs include this benchmark. Verify
    # the model/prediction receipt before exporting directly from its results.
    if mlp_run is not None and mlp_runs is not None:
        raise ValueError('Use mlp_run or mlp_runs, not both')
    default_mlp = fresh / 'work/stellar_mass_mlp'
    benchmarks = (list(mlp_runs) if mlp_runs is not None else
                  [mlp_run] if mlp_run is not None else
                  [default_mlp] if default_mlp.exists() else [])
    for benchmark in benchmarks:
        from wwdc_repro.redshift.stellar_mass_mlp import verified_results
        verified_results(benchmark)
    for src in copies.values():
        if not src.is_file():
            raise FileNotFoundError(src)
    for dest, src in copies.items():
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.name in ("weights.tex", "coverage.tex", "truncation.tex"):
            dest.write_text(appendix_table_layout(src.read_text()))
        else:
            shutil.copy2(src, dest)
    for name, content in rendered.items():
        (project / "tables" / name).write_text(content)
    if benchmarks:
        export_stellar_mass_tables(benchmarks, project / 'tables', fresh / 'tables/stellar_mass_mlp')
    print(f"Exported {len(copies)} files and {len(rendered)} generated tables/macros to {project}")


def export_stellar_mass_tables(runs, destination, analysis_directory):
    """Generate MLP products under reproduction outputs; copy only LaTeX to the paper.

    CSV and provenance files remain in analysis_directory, alongside generated
    LaTeX. destination receives only the table and numerical macro files.
    """
    from wwdc_repro.redshift.stellar_mass_mlp import export_grid, export_results, verified_results
    reports = [verified_results(run) for run in runs]
    has_grid = all((run/result['arm']/'trial_metrics.csv').is_file()
                   for run, report in zip(runs, reports) for result in report['results'])
    if has_grid:
        export_grid(runs, analysis_directory)
        table = 'stellar_mass_mlp_grid.tex'
    elif len(runs) == 1:
        export_results(runs[0], analysis_directory)
        table = 'stellar_mass_mlp.tex'
    else:
        raise ValueError('Combining MLP runs requires trial_metrics.csv for every representation')
    destination.mkdir(parents=True, exist_ok=True)
    for name in (table, 'stellar_mass_mlp_numbers.tex'):
        shutil.copy2(analysis_directory / name, destination / name)


def main():
    """Parse export paths from the command line and update the project assets.

    Accepts --run or the pair --fresh and --validation, plus --project. The destination must contain
    draft.tex; argparse reports an error if it does not. Paths are interpreted
    relative to the current working directory unless absolute.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Completed validation-selected run")
    parser.add_argument("--fresh", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--project", type=Path, required=True)
    mlp_options = parser.add_mutually_exclusive_group()
    mlp_options.add_argument("--mlp-run", type=Path, help="Optional separately fitted stellar-mass benchmark directory")
    mlp_options.add_argument("--mlp-runs", type=Path, nargs='+',
                             help="Completed stellar-mass benchmarks to combine into the full-grid table")
    args = parser.parse_args()
    if args.run is not None:
        if args.fresh is not None or args.validation is not None:
            parser.error("--run cannot be combined with --fresh or --validation")
    elif args.fresh is None or args.validation is None:
        parser.error("provide --run or both --fresh and --validation")
    if not (args.project / "draft.tex").is_file():
        parser.error("project must contain draft.tex")
    export(args.run if args.run is not None else args.fresh, args.validation, args.project, args.mlp_run, args.mlp_runs)


if __name__ == "__main__":
    main()
