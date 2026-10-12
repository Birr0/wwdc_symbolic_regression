# Figure calculations

All outputs are written to the explicit `--outdir`. The figure command hashes
its inputs and source files in `manifest.json`. Numerical equivalence is the
migration criterion; fonts, colours and page layout differ from the old scripts.

| Figure stem | Inputs and calculation | Numerical output |
| --- | --- | --- |
| `frontier_lgm_tot_p50` (`--only frontier`) | Validation-selected test scores; seed mean and min–max at each budget. Paper Figure 1, in the manuscript's scienceplots style | `frontier_lgm_tot_p50.csv`; analysis `selected.csv` |
| `frontier_lgm_tot_p50_by_seed` (`--only seed-frontiers`) | Same selected rows, separated into three seed panels. Paper Figure 5 | Analysis `selected.csv` |
| `score_distribution_marginal` | Cleaned test rows, L24 conditional+z equations; unit SVD direction of zero-latent slopes on 120 redshifts between test 0.5/99.5 percentiles; central step 0.05 | JSON weights and moments |
| `latent_weights` (`--only weights`) | Same L24 score directions as `score_distribution_marginal`; raw coefficients of the linear latent sum read from each expression with SymPy, checked to be the same axis and support as the score direction | LaTeX `deluxetable` for `tab:weights`, JSON raw/unit vectors |
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
first match per spectrum. Its density is a Gaussian KDE from 60,000 rows
sampled with `random_state=0`, on a 180×180 grid. Contours target 10/30/50/70/90%
of KDE mass; the finite grid contains about 98.95%. The two Masters lines use
the audited ridge `0.73−0.02(Mr+20)` and blue edge `0.63−0.02(Mr+20)`.
The displayed colour range is 0.15–1.0; the calculations use 0.15–1.20.

CMD score weights use a central difference at z=0.05 with step 1e-4 and are
normalised in the positive local mass-gradient direction. The other score
plots use the historical SVD definition above. These definitions agree closely
for the paper equations, but are documented separately rather than silently
changing the audited calculations. A future unified convention requires a
recorded numerical comparison.

## Frontier failure markers

The paper frontier plots use hollow triangles for display-only fallbacks after a selected equation fails test evaluation. For each failed arm/seed/budget, the displayed value is the highest finite test R² among actual selections at strictly lower budgets for that arm and seed. This is a test-informed display convention, not a validation-selected replacement. A mean marker is hollow if any contributing seed uses a fallback; its band uses those displayed values. `frontier_display_values.csv` records the source budget and equation length, while `frontier_selected_scores.csv` preserves the strict aggregate with gaps. Analysis `selected.csv` and failure records remain unchanged.
