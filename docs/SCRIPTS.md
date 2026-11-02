# Which code reproduces which paper result?

Scope: the current pre-submission `draft.tex`, including Section 6
(`sections/normalisation.tex`) and the fibre-mass appendix
(`sections/fibre_mass_appendix.tex`). Standalone joint-conditioning figures,
redshift-target probes, exploratory spectrum MLPs and the alternative flow-training
experiment have been removed. Earlier versions remain in Git history.

Start with [QUICKSTART.md](QUICKSTART.md) for the redshift analysis or
[CONDITIONING.md](CONDITIONING.md) for Section 6. Run commands from the repository
root. The tables below describe every retained experiment Python file.

## Section 6: the usual starting points

These files are under `experiments/conditioning/scripts/`. Invoke them through
`suite.py run --workdir /path/to/work SCRIPT [arguments]`; the launcher supplies
the isolated working directory. See the walkthrough for the locked environment,
input export and raw-data options. All outputs below are relative to that workdir.

| Script | Paper use | Inputs / outputs |
|---|---|---|
| `norm_conditioning_probe.py` | Table `tab:norm_probes`: NORM predictability from ten latents | Saved matched matrices, training indices and NORM scaling → `norm_conditioning_probe_2026-10-18/` correlations, linear/MLP scores, checkpoints and report. Trains only the three representations reported in the paper. |
| `norm_input_ablation.py` | Table `tab:norm_ablation`: supplying versus withholding NORM | `report` reuses saved selected equations → `norm_input_ablation_2026-10-16/summary.csv`; `prepare`, `fit`, `run` rebuild matrices/searches. |
| `plot_norm_frontier.py` | Figure `fig:norm_frontier` | Ablation selected equations → `norm_frontier/norm_frontier.pdf`, PNG, summary and manifest. Run after the ablation report. |
| `generate_norm_sequence.py` | Figure `fig:norm_sequence` | NORM checkpoint/YAML, upstream source, saved spectra/embeddings and CMD rows → specified `--outdir` with generated/observed arrays, figure and numerical checks. Holds each inverse-flow seed fixed. |
| `preview_raw_norm_sequence.py` | Replot that same spectral figure without inference | `--input` saved generation directory, `--outdir` new plot directory → raw-decoder and amplitude-restored panels. No new flow inference. |
| `plot_cmd_norm_s.py` | Prepare the additive latent score for Figure `fig:norm_cmd` | Frozen seed-42 budget-15 equation, matrices/IDs and VAGC cache → `norm_cmd_s_2026-10-16/plot_data.parquet`; checks S against the equation difference. Also writes intermediate CMD plots. |
| `plot_norm_cmd_full.py` | Figure `fig:norm_cmd` | Above plotting rows → `norm_cmd_full_2026-10-18/norm_cmd_full.pdf`, PNG, bin/KDE arrays and metadata. This is the manuscript CMD renderer. |

## Preparing searches, selection and equation checks

Use these when rebuilding saved candidates, verifying the displayed equations or
reconstructing inputs. They are not additional independent paper experiments.

| Script | Why retained | Inputs / outputs |
|---|---|---|
| `run_normflow.py` | Regenerate Section 6 NORM-conditioned candidate fronts | Raw embeddings/catalogues → `job_results_*` fronts. Use `--append-norm --embed-type cond+z --seed 42` for the explicit-NORM-plus-z case. Historical diagnostic summaries are not the paper selection. |
| `validate_conditioning.py` | Build matrices and audit candidate eligibility | Raw inputs and archived fronts → `conditioning_validation_2026-10-13/` matrices, IDs, candidate audit and initial selections. Also provides the shared `score`/`select` functions. |
| `rerun_conditioning_exports.py` | Recover seed-42 exports used by final selection | Exact candidate pools and raw inputs → `conditioning_validation_2026-10-13/rerun_*` high-precision fronts and exact-match records. Runs two SR searches. |
| `finalize_conditioning_validation.py` | Freeze choices before held-out scoring | Candidate audit, matrices and recovered exports → `selection_frozen_before_test.csv`, `validation_selected_final.csv`, summaries and manifest. |
| `matched_conditioning_controls.py` | Matched original/z-flow controls and shared SR fitting | `prepare` builds matched matrices; `fit --arm original --seed 42` runs one search under `conditioning_validation_2026-10-13/matched_controls/`. Shared fitting is also used for NORM ablations and fibre searches. |
| `analyse_matched_conditioning.py` | Structure and scaling of selected total-mass equations | Frozen choices and matrices → matched-control `comparison/` equation/structure diagnostics. Its gradient helpers are used by ablation and fibre code. |
| `check_conditioning_physics.py` | Undo NORM scaling and check physical-reference slopes | Raw inputs and resolved NORM YAML → `conditioning_validation_2026-10-13/physical_interpretation/metadata.json`, scaling and association diagnostics. The probe reads its NORM mean/std. |
| `audit_shared_structure.py` | Verify additive versus nonlinear latent dependence | Selected total/fibre expressions → `shared_structure_audit_2026-10-14/` symbolic structure counts and equations; supports the fibre counterexample. |

The historical prepared archive contains joint-conditioned arms. Shared validation
and structure routines still read these to reproduce the frozen archive and its
reference tables. They are not a separate result in the current manuscript.
Keeping those branches avoids changing archived selections or common-sample
definitions during this cleanup. Dedicated joint-only searches/figures are omitted.

## Fibre-mass appendix

| Script | Paper use | Inputs / outputs |
|---|---|---|
| `fibre_mass_conditioning.py` | Conditioned fibre-mass equations | `prepare` builds fibre matrices and matched training indices; `fit --arm norm --seed 42` runs one SR search in `fibre_mass_conditioning_2026-10-14/`. |
| `fibre_mass_original_control.py` | Original-latent fibre counterexample | `prepare` and `fit --seed 42` build/fill `original_seed42/` using identical objects and training subsets. |
| `analyse_fibre_mass.py` | Conditioned equation structure | Raw catalogue/embedding/YAML inputs plus saved fibre fits → fibre `comparison/` diagnostics. Also supplies the symbolic fixed-direction helper. |
| `analyse_fibre_original_control.py` | Equations `eq:fibre_original`, `eq:fibre_norm15` and SR columns of Table `tab:fibre_performance` | Saved original fits and conditioned summaries → fibre `original_control_comparison/performance.csv`, equation atlas and structure checks. |
| `fibre_mass_mlp_control.py` | MLP column of Table `tab:fibre_performance` | Prepared fibre matrices/training indices → `mlp_control_2026-10-15/` selected checkpoints, histories, predictions and summary. Trains only original and NORM-conditioned representations. |

## Shared helpers: import these, do not run them

These seven files are dependencies of the workflows above. The launcher does not
offer them as runnable commands.

| File | Responsibility / consumers |
|---|---|
| `latent_mlp.py` | Ten-input 64/32 ReLU training loop used by `norm_conditioning_probe.py`. Formerly named `d4000_mlp.py`; the unrelated D4000 entry point was removed. |
| `val_rescore.py` | Catalogue joins, feature matrices and exported-expression parsing/evaluation used throughout selection and figures. |
| `run_total_mass.py` | Historical `run_cell` and search settings imported by `run_normflow.py`. |
| `utils.py` | Historical symbolic fitting and diagnostic baseline routines used by SR runners. |
| `processing_fun.py` | Processing and Operon support imported by `utils.py`. |
| `cmd_helpers.py` | Photometric magnitude conversion and red-sequence curves used by CMD scripts. |
| `figstyle.py` | Shared plotting style with a fallback when LaTeX is unavailable. |

## Launcher, verification and the earlier redshift analysis

| File | Responsibility |
|---|---|
| `experiments/conditioning/suite.py` | `list` describes runnable paper workflows; `export` copies the historical prepared input archive; `verify` checks source/input hashes; `run` executes one workflow in that working copy. |
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
