# Which code reproduces which paper result?

Scope: the current pre-submission `draft.tex`, including Section 6
(`sections/normalisation.tex`) and the fibre-mass appendix
(`sections/fibre_mass_appendix.tex`). Standalone joint-conditioning figures,
redshift-target probes, exploratory spectrum MLPs and the alternative flow-training
experiment have been removed. Earlier versions remain in Git history.

Start with [QUICKSTART.md](QUICKSTART.md) for the redshift analysis or
[CONDITIONING.md](CONDITIONING.md) for Section 6. Run commands from the repository
root. The tables below describe every retained experiment Python file.

The combined workflow is [`reproduce.py`](../reproduce.py). It reads
[`reproduction.example.json`](../reproduction.example.json), checks the inputs,
runs the saved-equation stages in order and collects the results. Use it for
normal reproduction; the individual files below are reference for advanced work.

## Conditioning review map: what needs to run?

The normal reproduction starts from prepared inputs and saved searches. A script
not executed by the runner may still provide imported helpers or produce one of
those saved inputs. “Not run” therefore does not mean safe to delete.

### Shared scoring and selection

The scientific selection policy is implemented once in
[`src/wwdc_repro/shared/selection.py`](../src/wwdc_repro/shared/selection.py):

| Function | Responsibility |
|---|---|
| `score_expression_details` / `score_expression` | Evaluate a saved equation and calculate R²; classify failures. |
| `audit_candidate` | Reproduce the fit-subsample training score; evaluate validation only when requested. Test arrays are not accessed. |
| `select_candidate` | Choose the eligible training winner (or optional validation winner) within a length budget; ties prefer shorter equations then stable input order. |

Redshift `frontier.py` and conditioning `validate_conditioning.py` use these
helpers. Conditioning preparation and finalisation now share this single entry
point; the separate `finalize_conditioning_validation.py` has been removed. `validate_conditioning.select`
is only an adapter for historical `Length`/`val_r2` column names; other existing
conditioning callers therefore also use the shared policy.

Training selection is the default throughout. In optional validation mode,
conditioning also checks full-training finiteness; redshift retains its historical
fit-subsample/validation eligibility check. Data preparation, arm definitions, budget ranges and verified
recovery formats remain caller responsibilities. This consolidation does not
rerun searches or change the prepared-archive starting point.

Verification: all 336 final conditioning candidate audits, 374 conditioning
selections and 447 redshift selections match the prior saved outputs. Selected
conditioning test scores match the previous evaluator within 1e-12. See
`outputs/shared-selection-verification/verification.json`.

### Shared data preparation

`validate_conditioning.prepare_inputs` now calls `wwdc_repro.shared.data.load_sample`,
`clean_target` and `matrices`, the same functions used for redshift. It no longer
imports `conditioning_data`. NORM is taken from the final embedding conditioning
coordinate and stored as a scalar column, leaving the ten latents unchanged.
`matrices(..., scaled_columns=("norm",))` appends and standardises that input
with the latents on the full cleaned training split. For `+z` arms, raw redshift
is appended afterward. The default empty `scaled_columns` preserves redshift's
existing behaviour.

Rebuilt all five arm matrices from local raw inputs: every feature/target array
and all nine ordered split-ID arrays match the archive exactly. The subsequent
336 candidate audits and 374 selections/test scores also match previous outputs.
See `outputs/shared-conditioning-data-verification/verification.json`.
`conditioning_data.build_merged` now delegates to `wwdc_repro.shared.data.load_sample`;
its duplicate embedding loader and catalogue joins are removed. All train/val/test
DataFrames for redshift, NORM, and redshift+NORM match the previous loader exactly,
including row order and all columns, and pass the shared stricter checks.
`conditioning_data.build_matrices` now handles only legacy target cleaning and
separating appended coordinates; scaling and matrix assembly delegate to
`wwdc_repro.shared.data.matrices`. The shared builder accepts training plus any supplied
validation/test splits. Exact-array comparisons passed for all four saved fibre
representations and both matched total-mass controls, plus flux targets and
widened/singleton latent arrays. Expression parsing and evaluation now use `wwdc_repro.shared.equations`;
unused constant-refitting and front-rescoring helpers have been removed.

### Default execution order

Paths in this table refer to `experiments/conditioning/`, with scientific scripts
in `src/wwdc_repro/`. This is the conditioning part of `reproduce.py`, after redshift
analysis. Script names are stable identifiers; dated directories are legacy
locations scheduled for cleanup in [PAPER_REVIEW.md](PAPER_REVIEW.md).

| Order | File / command | Purpose | Main product and consumer | Manuscript figure? |
|---|---|---|---|---|
| 1 | `suite.py export`, then `verify` | Copy prepared inputs and check their data hashes | Working archive used by subsequent steps | No |
| 2 | `validate_conditioning.py` | Apply verified precision recoveries, audit all candidates, freeze training selections (validation optional), then evaluate tests | `validation_selected_final.csv` and `candidate_audit_final.csv`; read by CMD preparation and shared input helpers | No; intermediate plot removed |
| 3 | `norm_input_ablation.py compare_results` | Recheck saved selections/scores with and without explicit NORM | `summary.csv` for the ablation table; `selected_all_budgets.csv` for the frontier | No |
| 4 | `plot_norm_frontier.py` | Render the matched four-arm frontier | `norm_frontier.pdf` | Yes |
| 5 | `plot_norm_cmd_full.py` | Check additive scores, join photometry and render the final CMD | `plot_data.parquet` and `norm_cmd_full.pdf` | Yes |
| 6 | `analyse_fibre.py` | Compare saved original and conditioned fibre equations | `performance.csv` and printed budget-15 latent-structure checks | No; supports appendix table/equations |
| 7 | `check_outputs.py` | Compare selection, ablation and fibre numerical tables with references | Pass/fail checks | No |
| Optional | `generate_norm_sequence.py` | Replot supplied generated/observed spectral arrays | Spectral-sequence PDF | Yes; only with `norm_generation` |

### Products of validate_conditioning.py

The workflow is `load_candidates → apply_recoveries → audit_candidates →
freeze_selection → evaluate_selected → write_outputs`. `prepare_inputs` is an
optional raw-data stage. Legacy `candidate_audit.csv` supplies only candidate
metadata when `candidates.csv` is absent: all eligibility checks are recomputed.
There is no preliminary selection/test pass or new `validation_selected.csv`.
The `_final` output names remain for downstream compatibility.

The combined workflow was rerun on the prepared archive: all 336 candidate
audits, 374 selections, 15 budget-29 rows and 125 frontier-summary rows match
the previous outputs (`rtol=1e-10`, `atol=1e-12`). The final selection also
matches the bundled reference. See
`outputs/combined-conditioning-verification/verification.json`.

| Output | Readers / reason retained |
|---|---|
| `validation_selected_final.csv` | Read by the NORM CMD, ablation, and output checks. |
| `candidate_audit_final.csv` | Read alongside selections by the same shared `inputs()` helper for candidate eligibility and validation checks. |
| `selected_equations_budget29.csv` | Historical budget-29 export; the former physical diagnostic reader has been removed. |
| `selection_frozen_before_test.csv` | Audit record of decisions before held-out scoring. |
| `frontier_summary.csv` | Numerical summary across seeds. No direct downstream reader identified in the current Python workflow. |

The ablation loads explicit-NORM selections and audits directly from their
source outputs using `load_results()`. Preparation saves reduced matrices for
the without-NORM searches, without copying selections, audits or frames.

### Files outside the default execution path

- **Rebuild data/searches:** `validate_conditioning.py --prepare-only`,
  `rerun_conditioning_exports.py`, `matched_conditioning_controls.py`,
  `run_normflow.py`, `fibre_mass_conditioning.py`,
  preparation/fit modes of `norm_input_ablation.py`.
- **Train probes:** `norm_conditioning_probe.py`, `fibre_mass_mlp_control.py`.
  The default collector copies archived MLP summaries instead of retraining.
- **Generate new spectra:** `generate_norm_sequence.py`; the default optional
  spectral step only replots saved arrays.
- **Shared support:** `latent_mlp.py`, `conditioning_data.py`, `run_total_mass.py`,
  `utils.py`; CMD helpers and plotting styles come from `wwdc_repro.shared.cmd` and `wwdc_repro.shared.figstyle`.
  These are imported dependencies, not independent launcher commands.

### Cleanup candidates, not deletion decisions

- Separate shared input/gradient helpers from analysis entry points so importing
  a helper does not obscure dependencies.
- Replace hardcoded dated locations consistently across scripts, manifests and
  the collector. Keep run identifiers as provenance metadata.
- Make reuse of archived MLP results and ablation snapshots explicit; do not
  describe them as freshly trained or newly selected results.

## Section 6: the usual starting points

These files are under `src/wwdc_repro/`, alongside the redshift analysis. Invoke them through
`suite.py run --workdir /path/to/work SCRIPT [arguments]`; the launcher supplies
the isolated working directory. See the walkthrough for the locked environment,
input export and raw-data options. All outputs below are relative to that workdir.

| Script | Paper use | Inputs / outputs |
|---|---|---|
| `norm_conditioning_probe.py` | Table `tab:norm_probes`: NORM predictability from ten latents | Saved matched matrices, training indices and NORM scaling → `norm_conditioning_probe_2026-10-18/` correlations, linear/MLP scores, checkpoints and report. Trains only the three representations reported in the paper. |
| `norm_input_ablation.py` | Table `tab:norm_ablation`: supplying versus withholding NORM | `compare_results` reuses saved selected equations → `norm_input_ablation_2026-10-16/summary.csv`; `prepare`, `fit`, `run` rebuild matrices/searches. |
| `plot_norm_frontier.py` | Figure `fig:norm_frontier` | Ablation selected equations → `norm_frontier/norm_frontier.pdf`, PNG, summary and manifest. Run after the ablation report. |
| `generate_norm_sequence.py` | Figure `fig:norm_sequence` | NORM checkpoint/YAML, upstream source, saved spectra/embeddings → specified `--outdir` with generated/observed arrays, figure and numerical checks. Holds each inverse-flow seed fixed. Pass `--input` to replot saved arrays without inference. |
| `plot_norm_cmd_full.py` | Prepare and render Figure `fig:norm_cmd` | Checks the seed-42 budget-15 additive score, joins photometry, saves `norm_cmd_s_2026-10-16/plot_data.parquet`, then renders `norm_cmd_full_2026-10-18/norm_cmd_full.pdf` and PNG with bin/KDE arrays and metadata. |

## Preparing searches, selection and equation checks

Use these when rebuilding saved candidates, verifying the displayed equations or
reconstructing inputs. They are not additional independent paper experiments.

| Script | Why retained | Inputs / outputs |
|---|---|---|
| `run_normflow.py` | Regenerate Section 6 NORM-conditioned candidate fronts | Raw embeddings/catalogues → `job_results_*` fronts. Use `--append-norm --embed-type cond+z --seed 42` for the explicit-NORM-plus-z case. Historical diagnostic summaries are not the paper selection. |
| `validate_conditioning.py` | Unified recovery, candidate audit, validation selection and test reporting | Default: saved metadata/matrices/recoveries → final audit, frozen choices and test-score CSV tables. `--prepare-only`: raw inputs/fronts → matrices, IDs, `candidates.csv`; `--prepare`: prepare then validate. Shared policy lives in `wwdc_repro.shared.selection`. |
| `rerun_conditioning_exports.py` | Recover seed-42 exports used by final selection | Exact candidate pools and raw inputs → `conditioning_validation_2026-10-13/rerun_*` high-precision fronts and exact-match records. Runs two SR searches. |
| `matched_conditioning_controls.py` | Matched original/z-flow controls and shared SR fitting | `prepare` builds matched matrices; `fit --arm original --seed 42` runs one search under `conditioning_validation_2026-10-13/matched_controls/`. Shared fitting is also used for NORM ablations and fibre searches. |

The historical prepared archive contains joint-conditioned arms. Shared validation
and structure routines still read these to reproduce the frozen archive and its
reference tables. They are not a separate result in the current manuscript.
Keeping those branches avoids changing archived selections or common-sample
definitions during this cleanup. Dedicated joint-only searches/figures are omitted.

## Fibre-mass appendix

| Script | Paper use | Inputs / outputs |
|---|---|---|
| `fibre_mass_conditioning.py` | Conditioned fibre-mass equations | `prepare` builds fibre matrices and matched training indices; `fit --arm norm --seed 42` runs one SR search in `fibre_mass_conditioning_2026-10-14/`. |
| `analyse_fibre.py` | Equations `eq:fibre_original`, `eq:fibre_norm15` and SR columns of Table `tab:fibre_performance` | Saved original and conditioned fits → fibre `original_control_comparison/performance.csv` and printed budget-15 latent-structure checks. |
| `fibre_mass_mlp_control.py` | MLP column of Table `tab:fibre_performance` | Prepared fibre matrices/training indices → `mlp_control_2026-10-15/` selected checkpoints, histories, predictions and summary. Trains only original and NORM-conditioned representations. |

## Shared helpers: import these, do not run them

These seven files are dependencies of the workflows above. The launcher does not
offer them as runnable commands.

| File | Responsibility / consumers |
|---|---|
| `latent_mlp.py` | Ten-input 64/32 ReLU training loop used by `norm_conditioning_probe.py`. Formerly named `d4000_mlp.py`; the unrelated D4000 entry point was removed. |
| `conditioning_data.py` | Embedding loading, catalogue joins and feature-matrix preparation for fitting and controls. |
| `run_total_mass.py` | Shared CLI for redshift/NORM sweeps and `run_cell`; `--conditioning norm` selects NORM defaults, `--out` chooses a fresh parent directory. |
| `utils.py` | Shared `fit_sym_fn` for DataFrames or prepared matrices, explicit search settings, and diagnostic baseline fitting. |
| `src/wwdc_repro/shared/cmd.py` | Shared Astropy magnitude conversion and red-sequence curves for both CMD analyses. |
| `src/wwdc_repro/shared/figstyle.py` | Shared redshift/NORM colours and fixed SciencePlots no-LaTeX style. |

## Launcher, verification and the earlier redshift analysis

| File | Responsibility |
|---|---|
| `experiments/conditioning/suite.py` | `list` describes runnable paper workflows; `export` copies the historical prepared input archive; `verify` checks input hashes; `run` executes one workflow in that working copy. |
| `experiments/conditioning/check_outputs.py` | Compare three regenerated numerical tables with bundled references: final selection, NORM ablation and fibre performance. |
| `experiments/rerun_sr.py` | Rerun a redshift-analysis SR arm/seed with the separately locked historical environment. See [experiments](../experiments/README.md). |
| `experiments/recover_training_scores.py` | Recover native training references after exact historical reruns; fixes ambiguous old export flags without consulting test performance. |
| `experiments/compare_selection.py` | Audit historical versus corrected validation selections and test failures. See [selection audit](SELECTION_AUDIT.md). |
| `experiments/historical/utils.py` | Hash-pinned historical SR fitting implementation used by `rerun_sr.py`. |
| `experiments/historical/processing_fun.py` | Hash-pinned processing dependency of that historical implementation. |

The installed commands under `src/wwdc_repro/` reproduce the earlier redshift
analysis, figures, embeddings and embedding-MLP comparison. Their module map is in
[CODE_GUIDE.md](CODE_GUIDE.md), their figure map is in [FIGURES.md](FIGURES.md),
and their command recipes are in [QUICKSTART.md](QUICKSTART.md).

The locks and source manifests record the environments/provenance for retained
calculations. Historical verification JSON records describe their dated runs;
they are not evidence that this trimmed checkout reran training or searches.

### Fibre-mass scripts

Fibre-specific code also lives in `src/wwdc_repro/`:

- `fibre_mass_conditioning.py`: prepare and fit original, redshift-, NORM- and jointly conditioned inputs.
- `fibre_mass_mlp_control.py`: train the matched MLP comparisons.
- `analyse_fibre.py`: build the manuscript performance table and check budget-15 latent structure.

The existing `suite.py run ... <script-name>` commands remain supported by a
compatibility launcher. All active implementations now live in `src/wwdc_repro/`;
the suite runs package modules directly and does not stage Python files in outputs.
Direct execution uses `python -m wwdc_repro.<topic>.<module>` with the package installed
or reproduction `src` on `PYTHONPATH`. Redshift and conditioning share data,
equation evaluation, selection, fitting, MLP, CMD, and style helpers.

### Unified package and review progress

Review status as of 2026-11-11: `[x]` walkthrough complete; `[~]` partial or
explicitly deferred; `[ ]` walkthrough still needed/not confirmed. Package
`__init__.py` files are omitted. Changes to shared helpers may still merit
checking even where their callers have been reviewed.

```text
src/wwdc_repro/
├── shared/
│   ├── data.py, conditioning_data.py  # catalogue joins and matrix preparation
│   ├── catalogue.py, embedding.py    # catalogue/embedding production
│   ├── equations.py, selection.py    # parsing, evaluation and selection
│   ├── fitting.py, mlp.py            # common SR and MLP implementations
│   └── cmd.py, figstyle.py, utils.py  # shared CMD, style and hashing helpers
├── redshift/
│   ├── cli.py, frontier.py
│   ├── figures.py, paper_panels.py, latent_corner.py
│   ├── weights.py, diagnostic_tables.py, front_tables.py
│   └── conditional_distribution.py, mlp_probe.py
├── norm/
│   ├── run_normflow.py, matched_conditioning_controls.py
│   ├── validate_conditioning.py       # archived recovery details deferred
│   ├── rerun_conditioning_exports.py  # archive-only recovery
│   ├── norm_input_ablation.py
│   ├── norm_conditioning_probe.py, latent_mlp.py
│   ├── plot_norm_frontier.py, plot_norm_cmd_full.py
│   └── generate_norm_sequence.py
├── fibre_mass/
│   ├── fibre_mass_conditioning.py
│   ├── fibre_mass_mlp_control.py
│   └── analyse_fibre.py
├── workflows/
│   ├── run_total_mass.py              # common search command for redshift/NORM
│   └── conditioning_suite.py          # reproduction workflow launcher
├── resources/                        # catalogue and reference JSON files
└── CLI compatibility launchers       # preserve existing installed commands
```



Historical code stays under `experiments/historical/` for archived reproduction.


Fibre preparation is unified in `fibre_mass_conditioning.py`: `prepare` builds
all four arms, and `fit --arm original --seed 42` replaces the former separate
original-control fitting command. `fibre_mass_original_control.py` was removed.
A fresh preparation from the raw catalogue/embedding inputs was checked against
archived results: every X/y array and ordered spectrum ID matches exactly for
all four arms, as do the saved training indices and IDs for all three seeds.
The conditioned-only analysis explicitly uses `CONDITIONED_ARMS`, avoiding
counting original results twice when the final comparison combines them.

`analyse_fibre_mass.py` has been removed: its unused physical diagnostics,
plots and report generation are no longer part of the scripts. The final
`analyse_fibre.py` reads each conditioned run's `selected.csv`
directly alongside the original runs, writes `performance.csv`, and prints the
budget-15 structure checks. It no longer requires a precombined conditioned
comparison CSV. `has_fixed_latent_direction` now lives in
`src/wwdc_repro/shared/equations.py`; the optional shared-structure audit imports it
there and also reads the fibre selections directly.

The fibre MLP now uses the shared scikit-learn trainer in `wwdc_repro.shared.mlp`.
Its new outputs live in `fibre_mass_conditioning_2026-10-14/mlp_sklearn/`, with
`--outdir` available to select another new destination. Historical PyTorch
results and manuscript references are preserved until explicitly reconciled.

Both MLP callers now use `wwdc_repro.shared.mlp.run_mlp_experiment`, which owns training,
checkpoint persistence, evaluation, summaries and model settings. The fibre
caller specifies 12 inputs and the fibre-mass target; `latent_mlp.py`
specifies ten inputs and log NORM. The NORM probe passes paths and arms explicitly
rather than mutating module globals. Fresh sklearn probe outputs go to
`norm_conditioning_probe_2026-10-18/mlp_sklearn/`; old `mlp/` PyTorch results
remain historical. The nine-run NORM sklearn probe was rerun on 2026-11-11; all checkpoints reproduce their saved predictions and best validation scores.

Fitting consolidation (2026-11-11): `run_normflow.py` is now a thin entry point
into `run_total_mass.main(conditioning="norm")`. Both use the same CLI, data
loading and output-writing code. Historical output prefixes remain unchanged;
output labels no longer require duplicated target columns. Both reject existing
run directories before loading data, and `--out` supports a fresh destination.

All current conditioning searches, including matched controls, NORM ablations,
and fibre mass, call `shared.utils.fit_sym_fn`. DataFrames are converted with
`build_matrices`; precomputed matrices enter the same fitter without rescaling.
Saved training indices are passed explicitly when required. Search settings come
from `search_parameters`, with no prior-run manifest dependency. Diagnostic
MLP/LR scores in the historical front export remain separate from manuscript MLP
comparisons. Archived `experiments/historical` code is intentionally unchanged.

The matched fitter calls the shared `audit_candidate` and `select_candidate`
directly, preserving its full-training eligibility check and existing CSV names.
Its hashes now identify data files only. Validation selections are still written
before test scoring; the full from-scratch manuscript search and SLURM execution
remain separate follow-up work.

The unused matched-frontier and shared-structure diagnostic scripts were removed.
The manuscript's fibre latent-linearity checks remain in `analyse_fibre.py`.
Historical diagnostic outputs remain in the archives for provenance, but are not
required to regenerate current manuscript figures and tables.

NORM probe scaling is calculated by `norm_target_scaling(data_root, prepared)`
in `analysis/norm_conditioning_probe.py`. It loads NORM embeddings and catalogue
masses, checks cleaned split IDs against the prepared sample, undoes the flow's
YAML standardisation, and returns the full cleaned training mean/std. The probe
no longer reads `physical_interpretation/metadata.json`; the diagnostic producer
was removed. Running the probe now requires the raw data root and checkpoint
YAML, supplied through `suite.py run --data-root /path/to/data ...` or DATA_ROOT.
The recovered mean/std were verified to match the archived values exactly.
