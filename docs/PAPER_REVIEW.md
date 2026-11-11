# Paper reproduction review

First inventory: 2026-11-03. This is a review checklist, not a certification.
The source reviewed is `pre-submission/WWDC_Symbolic_Regression/draft.tex`
in the parent paper workspace, with its active `normalisation`, `front_condz`
and `fibre_mass_appendix` inputs. Figure/table numbers below follow the local
`draft.aux`; labels are the stable identifiers. Recheck numbering after compiling
any manuscript changes. Disabled `\iffalse` content is excluded.

## Run and locate results

Follow [QUICKSTART.md](QUICKSTART.md), then run:

```bash
python3 reproduce.py --config reproduction.json
```

All output paths below are relative to `outputs/paper/`. Redshift figure code
is under `src/wwdc_repro/`; conditioning scripts are under
`src/wwdc_repro/`. See [SCRIPTS.md](SCRIPTS.md) for their
individual inputs and commands. The combined runner supplies embeddings,
catalogues, candidate fronts and the prepared conditioning archive; Figure 5
also requires `norm_generation` saved arrays.

“Generated” means the runner invokes the calculation and collects the output.
It does not mean we have checked every plotted value or matched the manuscript
visually. This inventory inspects code and existing outputs; no new scientific
run was performed for this documentation change.

## Figures

| Paper | Source code | Collected output | Review remaining |
|---|---|---|---|
| 1, `fig:frontier` | `figures.py`: `frontier_display` and frontier renderer | `figures/redshift/frontier_lgm_tot_p50.pdf` | Generated; check display fallback markers and manuscript filename `fig1_frontier.pdf`. |
| 2, `fig:x0` | `paper_panels.py`: `zero_latent` | `figures/redshift/zero_latent.pdf` | Generated; compare with manuscript `plot_x0_arms_overlay_nolimit_bylength.pdf`, including candidate population and rounded reductions. |
| 3, `fig:g_z` | `paper_panels.py`: `anatomy` | `figures/redshift/equation_anatomy_by_seed.pdf` | Generated; resolve fixed-score curves versus prediction-SD language in caption and Table 9. |
| 4, `fig:cmd` | `paper_panels.py`: `cmd` | `figures/redshift/cmd_latent_all_gr_seed42.pdf` | Generated; check photometric cuts, score direction, bins and KDE. |
| 5, `fig:norm_sequence` | `generate_norm_sequence.py` (fresh inference or `--input` saved arrays) | `figures/normalisation/norm_sequence.pdf` | Optional saved-array replot; check selected objects, decoder amplitude convention and caption. |
| 6, `fig:norm_frontier` | `plot_norm_frontier.py` | `figures/normalisation/norm_frontier.pdf` | Generated; compare plotted seed aggregation to Table 2. |
| 7, `fig:norm_cmd` | `plot_norm_cmd_full.py` | `figures/normalisation/norm_cmd.pdf` | Generated; manuscript calls this `norm_cmd_full.pdf`; verify equation-derived score and sample. |
| 8, `fig:performance_by_seed` | `figures.py`: seed frontier renderer | `figures/redshift/frontier_lgm_tot_p50_by_seed.pdf` | Generated; verify consistency with Figure 1 and failure markers. |
| 9, `fig:latent_distributions` | `latent_corner.py` | `figures/redshift/latent_corner.pdf` | Generated; verify cleaned sample, representation and contour definitions. |

The runner also collects `latent_dim_distributions.pdf` and
`score_distribution_marginal.pdf`. These are diagnostics, not active manuscript
figures. Decide whether to move them out of the paper-facing figures folder.

## Tables

| Paper | Current reproduction route | Gap to close |
|---|---|---|
| 1, `tab:norm_probes` | `norm_conditioning_probe.py` writes linear results and MLP scores; runner copies `tables/archived_norm_probe_mlp.csv` | Combined workflow does not export the complete paper table or recalculate its linear column. Distinguish archived and retrained scores. |
| 2, `tab:norm_ablation` | `norm_input_ablation.py compare_results` → `tables/norm_ablation.csv` | Generated and reference-compared; add manuscript-formatted table and check rounded cells. |
| 3, `tab:norm_family_values` | Frozen NORM equations and `validation_selected_final.csv` supply underlying fits | No dedicated coefficient-table export in the combined workflow. Derive coefficients in manuscript units and verify all three seeds. |
| 4–6, complete conditional+z fronts (three `longtable`s in `front_condz.tex`) | `front_tables.py` via `wwdc-figures --only front-tables`; runner collects `tables/front_condz.tex` and JSON audit | Integrated: historical finite archived Test_R2 rows, four-significant-figure constants, three-decimal scores. All 70 row lengths and printed scores checked against manuscript; not validation-selected rows. |
| 7, `tab:truncation` | `diagnostic_tables.py` → `tables/truncation_table.tex` and CSV | Verified all printed cells on 44,141 test rows. Exact SymPy first-order results agree with historical finite differences within 1e-8. |
| 8, `tab:weights` | `weights.py` → `tables/latent_weights.tex` | Generated; compare manuscript coefficients, indexing and rounding. |
| 9, `tab:coverage` | `diagnostic_tables.py` → `tables/coverage_table.tex` and CSV | Now uses anatomy unit-score A(z) at each test redshift, counting full predictions within baseline ±A and ±2A. Supersedes the SD-based verification; manuscript values/caption must follow this definition. |
| 10, `tab:fibre_performance` | `analyse_fibre.py` → `tables/fibre_sr.csv`; archived `tables/archived_fibre_mlp.csv` | SR comparison is reference-checked; conditioned selections are loaded from each run. Combine the paper columns explicitly and label archived MLP provenance. |

Disabled manuscript content: `tab:latents` and the luminosity-coupling appendix
(`fig:coupling`) are inside `\iffalse` blocks. They are not requirements for this
version. The three complete-front tables lack individual stable labels.

## Repository map and reading order

| Location | Responsibility | Read during |
|---|---|---|
| `README.md`, `download_data.py`, `reproduce.py` | Reader entry, verified data download, orchestration and output collection | Reader workflow review |
| `src/wwdc_repro/shared/data.py`, `equations.py`, `frontier.py`, `cli.py` | Catalogue joins, preprocessing, equation evaluation, selection and held-out scoring | Figure 1 walkthrough |
| `src/wwdc_repro/redshift/figures.py`, `paper_panels.py`, `latent_corner.py`, `weights.py` | Redshift figures and coefficient table | Figures 1–4 and appendices |
| `conditioning_suite.py` and the conditioning modules in `src/wwdc_repro/` | Prepared inputs, NORM and fibre calculations, training/search entry points and helpers | Section 6 walkthrough |
| `artifacts/`, packaged reference JSONs | Saved candidate equations, checksums, reference results and provenance | Alongside each calculation |
| `experiments/historical/`, search wrappers | Pinned historical symbolic-search reproduction | Fresh-search review; avoid incidental edits to pinned code |
| `tests/` | Automated checks of selection, inputs, orchestration and release handling | Alongside each implementation change |
| Upstream `wwdc_spectra/paper.py` | Flow training and embedding inference | Full reproduction review |

## Agreed review stages and proposed first batch

1. **Inventory (this page):** confirm active manuscript scope and record every gap.
2. **Reader workflow:** clean checkout, dependency install, published data download,
   output index and manuscript filename mapping. HF publication remains pending.
3. **Figure 1 walkthrough:** read `reproduce.py` → `cli.py` → `data.py` →
   `equations.py` / `frontier.py` → `figures.py`. Explain each data cut, selection
   rule and plotted quantity before proposing cleanup.
4. **Remaining figures and tables in paper order:** close missing exports and
   compare numerical values and displayed figures with the manuscript.
5. **Cleanup and release:** simplify confusing code in small reviewed changes,
   preserve scientific results, then rehearse from the public inputs.

For each batch record: what it computes; required inputs; outputs; scientific
assumptions; readability problems; proposed edits; and evidence after edits.
Do not mark an item complete solely because a command exits successfully.

Highest-priority coverage gaps are the incomplete table exports, the Figure 3 /
Table 9 definition mismatch, and public data availability. The manuscript's
reproducibility appendix also still contains a placeholder repository URL.

Figure 1 follow-up: [training-score mismatch investigation](TRAINING_SCORE_REVIEW.md)
records the coefficient-precision experiment and remaining native-evaluation questions.

## Deferred code-review actions

- [ ] Remove hardcoded dated run-directory names from `reproduce.py`, especially
  `collect_results` and its archived MLP source paths. Resolve inputs through
  configuration or a manifest with stable logical names. Preserve provenance
  (original run dates, source paths and hashes) as metadata, rather than embedding
  historical dates in workflow logic. Check the generated conditioning output
  paths at the same time. Requested during the `collect_results` review;
  implementation is deferred.

- [ ] Replace the hardcoded dated `ROOTS` entries in
  `experiments/conditioning/suite.py`: `conditioning_validation_2026-10-13`,
  `fibre_mass_conditioning_2026-10-14`, `norm_input_ablation_2026-10-16`, and
  `shared_structure_audit_2026-10-14`. Coordinate with the `reproduce.py`
  path cleanup so archive export and result collection use the same configured
  or manifest-based locations. Preserve original run identifiers as provenance.

- [ ] Update manuscript captions/methods and regenerate official outputs for the
  agreed selection policy: training Pareto fronts evaluated on test. Code now
  defaults to highest training R² within each length budget, with shorter-length
  tie breaking; validation selection remains available via `--selection validation`.
  Test scores do not choose equations, and test failures remain gaps. The search
  constructs the front using training MSE and length. Historical validation-based
  references apply only to the optional validation mode. Full searches and official
  manuscript output regeneration remain pending.

- [ ] Review historical export recovery together across NORM and redshift.
  NORM `validate_conditioning.apply_recoveries` replaces the two seed-42
  exports with verified `Equation_p60` expressions and native training scores
  from saved reruns. Redshift `frontier.score_fronts` uses
  `recovered_training_scores.json` for independent native training references,
  checking the source-front hash and equation. These address related historical
  export problems but currently use different formats and replacement rules.
  Consider corrected canonical fronts and shared provenance handling so routine
  analysis need not contain special-case recovery logic. Defer changes while
  reviewing `apply_recoveries`; preserve existing integrity checks and results.

- [ ] Explain the NORM sample-membership difference in the manuscript methods
  and relevant comparison captions. Suggested wording: "NORM conditioning uses
  the logarithm of the spectral normalisation and therefore requires a finite,
  positive normalisation. Relative to the cleaned historical redshift sample,
  this excludes 41 training, six validation and six test spectra, all with
  negative normalisations. Comparisons reported on the common test sample use
  the intersection of spectrum IDs across the representations."
  These are differences after the paper sample cuts, not the total upstream
  exclusions. Membership was traced exactly against the checkpoint-referenced
  caches and raw embedding exports. This filter explains the differences; it
  does not establish why those normalisations are negative or resolve the
  separate spectrum/metadata mislabelling question. Evidence and all 53 IDs:
  `DATA_REVIEW.md` and `outputs/conditioning-sample-difference/`.
  Reconcile this wording with the final choice of full versus common test scores
  when consolidating manuscript reporting; manuscript text is not yet changed.

- [ ] Update the NORM CMD score definition and caption: plotted S now uses the
  unit-length coefficient vector, with positive score increasing predicted mass.
  It is dimensionless, not the additive correction in dex. The raw additive
  term remains checked against the equation before normalization. Both CMDs use
  the 98th percentile of absolute bin means for symmetric colour limits.
  Redshift orientation is checked against the full equation; its nested latent
  sum enters with a negative factor, so its existing sign is retained.

- [ ] Audit the full reproduction codebase for generated Markdown reports,
  captions and other prose containing hardcoded numerical results, equations
  or scientific conclusions. Remove unnecessary report generation; any retained
  result summaries must derive their values from the current inputs. The
  hardcoded REPORT.md generation in analyse_fibre.py has been
  removed; the broader audit remains to be done.

- [ ] Reconcile the fibre MLP manuscript table/caption with the scikit-learn rerun
  (2026-11-11): three-seed mean test R² is 0.9868243905 for original latents and
  0.9824060917 for NORM-conditioned latents (historical PyTorch: 0.986952 and
  0.983751). No sklearn runs hit 500 epochs; the prior two-capped-runs caption
  no longer applies to these results. Architecture, train subsets, external
  validation checkpoint selection and no-test-selection policy are retained.
  Results: `outputs/paper/work/conditioning/fibre_mass_conditioning_2026-10-14/mlp_sklearn/`.
  All six serialized checkpoints reproduced their saved predictions and minimum
  validation MSE. The manuscript and `archived_fibre_mlp.csv` collector remain
  historical until the new values and implementation description are adopted.

- [ ] Update Table tab:norm_probes to the sklearn rerun (2026-11-11), if adopting
  the new implementation: mean test R² / seed range is original 0.502917
  (0.497254–0.506796), zflow 0.529966 (0.526408–0.533768), norm 0.025421
  (0.024537–0.026340). No runs reached the epoch cap. All nine reloaded
  checkpoints reproduced saved predictions and minimum validation MSE.
  Source: outputs/paper/work/conditioning/norm_conditioning_probe_2026-10-18/mlp_sklearn/summary.csv.
  Historical PyTorch scores and the archived table export remain unchanged.

- [ ] Verify SLURM compatibility when consolidating the symbolic-search runners.
  `run_searches()` currently starts local subprocesses; it does not submit jobs
  to SLURM or distribute them across nodes. It can run within a batch allocation,
  provided the environment, working directory and input paths are available.
  Match concurrent searches and Operon thread counts to allocated CPUs and
  memory; numerical-library thread limits alone do not control Operon threads.
  Consider one representation/seed per job-array task, with separate outputs and
  logs. Verify completion checks, failed/partial-run handling and restart behaviour
  before the planned full run from scratch. Cluster execution is not yet tested.

- [x] Consolidate search execution around `shared.utils.fit_sym_fn` for both
  DataFrames and prepared matrices. `search_parameters` replaces the dependency
  on an existing zflow manifest. Resolved pyoperon 0.4.0 settings match all six
  archived matched-control runs and both NORM seed-42 recovery runs. The NORM
  CLI delegates to the common total-mass runner. Historical arrays and sampling
  match for all five representations and three seeds in regression tests.

- [x] Matched fitting now imports shared `audit_candidate` and `select_candidate`
  directly, preserving validation-only selection and tie breaking. Existing CSV
  column names remain compatible with downstream readers; the compatibility
  wrapper remains for other archived-analysis callers. Shared audit output also
  records failure reasons. Matched preparation/run hashes now cover data only.

- [ ] After the remaining review, perform the full manuscript run from scratch
  in a fresh output directory and compare to preserved archived results. A short
  real Operon smoke search passed after consolidation; full searches have not
  been rerun. SLURM allocation and restart behaviour still need verification.

  Fresh total-mass/NORM searches now save Equation_p60 directly and retain native
  training scores independently of test failures. Conditioning preparation prefers
  these exports and records their precision, allowing fresh NORM seed-42 candidates
  to bypass the historical recovery files. Archived runs still use checked recovery.
  Eleven workflow/fitting tests and a small real Operon export check passed; this
  does not constitute a full-budget rerun or establish final manuscript scores.

- [ ] Document and pin the required `wwdc_spectra` source version for the final
  release. Readers will already be directed there for embeddings, but fresh
  spectral generation in `generate_norm_sequence.py` also directly imports its
  `PretrainedSpender`, `LightningFlowMatching`, `WrappedModel`, and `SDSS` classes.
  Document checkout/install instructions, dependencies, checkpoint/data paths,
  and the suite's `--upstream` option. Replotting saved arrays with `--input`
  does not require this source repository or model checkpoint.

- [ ] Before the final full reproduction, replace date-stamped and temporary
  review output directories with stable, descriptive official folder names.
  Update producers, downstream readers, workflow defaults and documentation
  together so every product is generated and consumed at a consistent path.
  Run the complete pipeline into a fresh official output tree, preserving
  historical archives separately for comparison.

- [x] Co-locate the active redshift, NORM, fibre-mass and fitting workflows in
  `src/wwdc_repro/`. Conditioning commands now run package modules directly;
  `experiments/conditioning/suite.py` remains a compatibility launcher. Historical
  implementations and output archives are preserved. This relocation does not
  perform the pending full-budget searches or change the output directory policy.

- [x] Organise the unified package into `shared`, `redshift`, `norm`, `fibre_mass`,
  `workflows`, and `resources`, updating imports and preserving CLI launchers.

  Topic-layout verification: all 52 tests pass; suite targets, package resources
  and existing installed CLI commands resolve. The NORM spectral-sequence replot
  is byte-identical to the prior output. No full symbolic searches were launched.
