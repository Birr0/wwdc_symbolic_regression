# Figure calculations

All outputs are written to the explicit `--outdir`. The figure command hashes
its inputs and source files in `manifest.json`. Numerical equivalence is the
migration criterion; fonts, colours and page layout differ from the old scripts.

| Figure stem | Inputs and calculation | Numerical output |
| --- | --- | --- |
| `frontier_lgm_tot_p50` (`--only frontier`) | Validation-selected test scores; seed mean and min–max at each budget. Paper Figure 1, in the manuscript's scienceplots style | `frontier_lgm_tot_p50.csv`; analysis `selected.csv` |
| `frontier_lgm_tot_p50_by_seed` (`--only seed-frontiers`) | Same selected rows, separated into three seed panels. Paper Figure 8 (`fig:performance_by_seed`) | Analysis `selected.csv` |
| `score_distribution_marginal` | Cleaned test rows, L24 conditional+z equations; unit SVD direction of zero-latent slopes on 120 redshifts between test 0.5/99.5 percentiles; central step 0.05 | JSON weights and moments |
| `latent_weights` (`--only weights`) | Raw coefficients of each L24 linear latent sum read with SymPy, normalised directly to unit length and sign-reversed for increasing mass over z=0.02–0.3; no SVD | LaTeX `deluxetable` for `tab:weights`, JSON raw/unit vectors |
| `latent_corner` | Cleaned paper test rows (44,141); `corner` 2.3.0, one call per representation; 80 bins over [−5,5], 1-bin smoothing, 68%/95% highest-density contours | JSON per-dimension moments and pair correlations |
| `latent_dim_distributions` | All 52,811 raw test embeddings before catalogue/target cuts or scaling; ten dimensions, 60 bins over [−5,5] | JSON per-dimension moments |
| `equation_anatomy_by_seed` | L24 canonical curves at fixed scores s=0, +/-1, +/-2 on 300 redshifts; observed test masses coloured by s. Exact for seed 42; exact first-order latent gradient at zero for seeds 43/44, retaining ln(1+cz). Replaces the earlier pooled prediction-SD envelope. | CSV baseline, signed amplitude A and five curves; JSON exact unit directions and gradient checks. Legend s=0, s=+/-1, s=+/-2; colour bar s. PDF and 600-dpi PNG. |
| `zero_latent` | All usable historical conditional+z/original+z front reductions across three seeds; six-significant-digit symbolic reductions; 30 observed-mass bins and 200 redshift grid points | CSV median reductions/deviations; excluded reductions JSON |
| `cmd_latent_all_gr_seed42` | All cleaned splits; seed42 L24 score; pinned cache; colour `ABSMAG_g−ABSMAG_r`, magnitude `ABSMAG_r−0.7745`; mean score in 60×56 cells, masked below 25 spectra | Compressed row CSV, grid NPZ, summary JSON. Paper style; colour bar $\langle s\rangle$ |

The zero-latent panel reproduces the historical population of saved-front
reductions. It is not a validation-selected subset; the finite parent test-score
filter is a historical plotting rule, not selection of the best test score.
A zero-latent prediction is a slice through an equation, not an exact conditional
mean. Anatomy bands describe model predictions, not observed-mass uncertainty
or guaranteed 68/95% coverage.

The CMD uses finite magnitudes between −99 and 0 and the frozen historical
first match per spectrum. Its density is plotted with Seaborn `kdeplot`, using
at most 60,000 rows sampled with `random_state=0`, Scott bandwidth, and a
180×180 evaluation grid clipped to magnitude [-24,-18] and colour [0.15,1.20].
Contours enclose approximately 10/30/50/70/90% of probability normalized over
that grid. This replaces the historical unnormalized finite-grid thresholds.
`cmd_grid.npz` stores the score-bin counts, means, and edges; KDE settings and
contour thresholds are recorded in `cmd_summary.json`, rather than exporting
Seaborn's internal density grid. The two Masters lines use
the audited ridge `0.73−0.02(Mr+20)` and blue edge `0.63−0.02(Mr+20)`.
The displayed colour range is 0.15–1.0; the calculations use 0.15–1.20.

CMD score weights are extracted directly from seed 42's discovered linear
latent sum, normalized to unit length, and sign-reversed to account for its
negative redshift multiplier. They agree with the previous finite-difference
weights to within 2e-11. Magnitudes are converted from h=1 to H0=70 using
Astropy distance moduli with fixed flat Omega_m=0.3 and no radiation.

## Frontier failure markers

The paper frontier plots use hollow triangles for display-only fallbacks after a selected equation fails test evaluation. For each failed arm/seed/budget, the displayed value is the highest finite test R² among actual selections at strictly lower budgets for that arm and seed. This is a test-informed display convention, not a validation-selected replacement. A mean marker is hollow if any contributing seed uses a fallback; its band uses those displayed values. `frontier_display_values.csv` records the source budget and equation length, while `frontier_selected_scores.csv` preserves the strict aggregate with gaps. Analysis `selected.csv` and failure records remain unchanged.

## Appendix equation tables

`wwdc-figures --only front-tables --fronts artifacts/fronts` (with the usual
required path arguments) writes `front_condz.tex` and `front_condz.json`.
The default figure run includes this export, and `reproduce.py` collects both
files into `tables/`. The input front CSVs are hashed in the run manifest.

The exporter preserves the historical population: conditional+z seeds 42–44,
ordered by length, excluding nonfinite archived Test_R2 values. These are not
validation-selected rows. SymPy canonicalises the stored Equation strings;
constants are printed to four significant figures and scores to three decimals.
Scores refer to the archived fitted models, not the rounded printed expressions.
A dagger flags suspected stored precision loss (seed 43, L=5). The JSON retains
original strings and scores, rendered expressions, warnings and excluded rows.
Parse failures stop the export rather than silently omitting a table row.

## Taylor truncation and prediction coverage

`wwdc-figures --only diagnostic-tables` uses the usual sample/selected/output
arguments and produces `truncation.csv`, `truncation_table.tex`, `coverage.csv`,
`coverage_table.tex`, `truncation.json` and `coverage.json`. Default figure runs include
these outputs; `reproduce.py` collects them in `tables/`.

The port in `diagnostic_tables.py` adapts the legacy `taylor_truncation.py` and
the coverage block in `s_as_mass_residual.py`. It uses the selected L24
conditional+z equations (verified identical to the archived L24 strings),
training-fitted latent scaling and all 44,141 cleaned test rows.

Taylor scores use a joint first-order SymPy series about zero latents, retaining
redshift exactly. This avoids the historical step-0.05 finite difference. Full
and approximate R² and their difference reproduce every printed manuscript
cell; historical numerical differences are below 1e-8. Seed 42 is checked to
remain unchanged to floating-point precision.

Coverage now uses exactly the anatomy figure's unit-score amplitude A(z),
computed as the norm of the exact first-order latent coefficients. For each
actual test redshift, the baseline is the equation with all latents set to zero.
The full prediction is counted within baseline ±A and ±2A, with inclusive
boundaries. No empirical standard deviation, latent rescaling, binning or
interpolation is used. The fixed coefficient direction is checked as in anatomy.
The generated caption identifies this definition explicitly; Gaussian fractions
are a comparison and do not establish normality.

The earlier SD-based verification is retained in
`outputs/diagnostic-tables-final/`; it matched the previous manuscript numbers
but used a different amplitude. The aligned calculation is in
`outputs/coverage-anatomy-amplitude/`. The optional conditional-distribution
review independently assesses centre, spread and Q–Q behaviour across redshift.

The two tables have independent entry points: `truncation_table(x, y, selected, outdir, selected_l24)` and `coverage_table(x, selected, outdir, selected_l24)`. The CLI constructs the matrices once and invokes both. Each table writes its own JSON record.
