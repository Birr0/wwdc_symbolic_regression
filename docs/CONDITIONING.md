# Reproducing the NORM sections and fibre appendix

Section 6, "Conditioning on spectral normalisation", is included by the
pre-submission manuscript from `sections/normalisation.tex`. This walkthrough
covers its reproduction code and the fibre-mass controls. See the
[script-by-script guide](SCRIPTS.md) for the purpose, inputs and outputs of every
retained file.

For the combined saved-result workflow, use [QUICKSTART.md](QUICKSTART.md)
and `python3 reproduce.py`. The commands below are for running individual
steps, fresh fits and checkpoint inference.

The individual-step entry point is `experiments/conditioning/suite.py`. It runs preserved analysis
scripts in a separate working copy, with no dependency on the flower checkout or
its directory location. This is checkpoint-first reproduction: flow training and
embedding production belong to [wwdc_spectra](https://github.com/Birr0/wwdc_spectra).

## Inputs and environment

Use Python 3.12 and the dedicated script lock. This creates an isolated experiment
environment, including pyoperon, PyTorch and generation dependencies:

```bash
uv run --script --locked experiments/conditioning/suite.py list
```

Run the commands below with that prefix. The lock lives beside the launcher as
`suite.py.lock`; direct experiment dependencies are pinned in the launcher's inline
metadata. To intentionally update it, edit the pins, run
`uv lock --script experiments/conditioning/suite.py`, and repeat the checks.
`environment.json` records the earlier analysis environment's direct versions;
the new lock resolves transitive dependencies and was tested separately.

The main package's `uv.lock` and historical SR runner's lock remain separate.
For base-package installation and regression tests use `uv sync --frozen` and
`uv run --frozen python -m unittest discover -s tests -v`. Native search results
can still depend on platform and thread configuration; a lock is not a guarantee
of identical stochastic searches across platforms.

The prepared input archive is about 1.16 GB. Its 293 files and hashes are listed in
[`prepared_inputs.json`](../artifacts/conditioning/prepared_inputs.json). It contains
matrices, row IDs, exact 10,000-row indices, saved candidate fronts, frozen
selections, reference summaries and the VAGC join. It does not contain the large
raw spectra, embeddings or model weights. Public download locations remain to be
assigned; Git alone cannot currently reproduce all results.

Stage an archive into a **new** working directory. For the local author archive,
`/path/to/archive` is `examples/spectra/symbolic_regression` in the flower checkout.
An already exported bundle has the same layout and can also be used as the source.

```bash
uv run --script --locked experiments/conditioning/suite.py export \
  --source /path/to/archive --workdir /path/to/conditioning-work
uv run --script --locked experiments/conditioning/suite.py verify \
  --workdir /path/to/conditioning-work
```

Export copies files; it never links writable results back to the originals. Verify
before running analyses: regenerated summaries/manifests may intentionally change,
so verification against the initial archive can fail afterwards. Keep the source
archive unchanged. The launcher runs the installed/shared package directly; it does not copy source
files into output directories. Some plotting/report scripts replace
outputs in the working copy; fits and probe training refuse existing run directories.
Only load trusted pickle frames and model checkpoints.

The prepared archive retains joint-conditioned inputs required by the historical
validation audit. The current paper has no standalone joint-conditioning section;
shared processing of those archived arms is documented in [SCRIPTS.md](SCRIPTS.md).

Small reference tables are committed under `artifacts/conditioning/reference/`.
Large prepared files are deliberately outside Git. Raw embedding and checkpoint
identities are recorded in `artifacts/conditioning/raw_inputs.json`; catalogue and
saved spectrum-split requirements are in [INPUTS.md](INPUTS.md).

## Representation and variable map

| Arm | Flow conditioning | SR predictors in matched comparisons |
|---|---|---|
| `original` | No flow | Original Spender latents, NORM, z |
| `zflow` | z | Conditional latents, NORM, z |
| `norm` | NORM | Conditional latents, NORM, z |
| `znorm` | z and NORM | Conditional latents, NORM, z |

`X1`–`X10` are training-standardised latents; `X11` is standardised log10 NORM;
`X12` is raw redshift. In the without-NORM ablation, `X11` becomes raw redshift.
Do not confuse flow conditioning with explicitly supplying a variable to SR.
The historical z-flow comparison has no explicit NORM and a slightly different
sample; the matched `zflow` arm fixes those differences.

NORM is the median usable observed flux density in the rest-wavelength window
5300–5850 Angstrom, not total broad-band flux. The SR standardisation is fitted
on the full filtered training split. Displayed equations undo that scaling to
use `ell = log10(N/N0)`; this changes coefficients, not predictions. Flow context
scaling and SR scaling are distinct. Targets are `lgm_tot_p50` for total mass and
`LGM_FIB_P50` for fibre mass, both logarithmic solar masses.

## Regenerate the section figures and tables

Run these commands from the reproduction repository. Paths after `--workdir`
can be anywhere; relative artifact names are resolved inside that directory.

```bash
uv run --script --locked experiments/conditioning/suite.py run --workdir /path/to/conditioning-work \
  validate_conditioning
uv run --script --locked experiments/conditioning/suite.py run --workdir /path/to/conditioning-work \
  norm_input_ablation compare_results
uv run --script --locked experiments/conditioning/suite.py run --workdir /path/to/conditioning-work \
  plot_norm_cmd_full
uv run --script --locked experiments/conditioning/suite.py run --workdir /path/to/conditioning-work \
  analyse_fibre
```

| Result | Location inside the work directory |
|---|---|
| NORM input-ablation frontier PDF and summary | `norm_input_ablation_2026-10-16/` |
| Full-sample CMD PDF, bin counts, KDE and metadata | `norm_cmd_full_2026-10-18/` |
| Per-spectrum latent correction S and photometry | `norm_cmd_s_2026-10-16/plot_data.parquet` |
| Fibre performance table (structure checks printed to console) | `fibre_mass_conditioning_2026-10-14/original_control_comparison/` |
| Frozen validation choices and held-out scores | `conditioning_validation_2026-10-13/` |

`plot_norm_cmd_full` checks that the selected seed-42, budget-15 equation has constant
latent derivatives and that `S = sum(w_i x_i)` equals `f(x,n,z)-f(0,n,z)` on all
splits. The CMD uses catalogue `ABSMAG_g-ABSMAG_r` and the paper's magnitude
conversion and filters. Full-sample counts use every matched row (440,087 here).
The colour panel requires 25 spectra per bin. Contours estimate 10/30/50/70/90%
KDE probability mass using a reproducible 60,000-row subsample; they are not exact
fractions of the empirical histogram. No random subsampling is used for counts or
mean S. SciencePlots falls back to Matplotlib text when LaTeX is unavailable.

The fibre manuscript comparison reads saved selections directly and includes
the latent-linearity checks needed by the appendix:

```bash
uv run --script --locked experiments/conditioning/suite.py run --workdir /path/to/conditioning-work \
  analyse_fibre
```

The full fibre diagnostics need raw catalogues, embeddings and the saved NORM YAML;
the first command list instead reuses their archived conditional summaries. The
structural audit covers the total and fibre comparisons here; an unrelated
exploratory `new_total` sweep was removed from this packaged audit. Fixed latent
direction, additive latent dependence and complete n/z/latent separability are
separate tests. Similarity to a physical scaling is not an exact identification
of S with a mass-to-light ratio.

## Latent-only probes and the fibre MLP control

```bash
uv run --script --locked experiments/conditioning/suite.py run --workdir /path/to/conditioning-work \
  norm_conditioning_probe
```

This probe uses **only ten latent coordinates** to predict log10 NORM.
Neither conditioning variable is included as a predictor. It reports coordinate
Pearson/Spearman correlations, OLS and a 64/32 ReLU MLP across seeds 42–44. Each
MLP uses the saved SR 10,000-row training subset, fits its input/target scaling on
that subset, and uses Adam at 0.001 with batches of 1024, at most 500 epochs and
patience 40. Minimum validation MSE chooses the checkpoint before test scoring.
Histories, preprocessing, checkpoints and predictions are saved. Weak probe
performance is evidence of reduced accessible dependence, not proof of independence.

`fibre_mass_mlp_control` now uses scikit-learn `MLPRegressor` with 12 inputs:
latents **plus NORM and z**. The 64/32 ReLU architecture, Adam learning rate
0.001, batch size 1024, 500-epoch cap and patience 40 are retained; `alpha=0`
preserves the previous absence of weight decay. `partial_fit` performs one
epoch at a time so checkpoint selection uses the existing full validation split,
not sklearn's internal random split. Input and target scalers are fitted only
on each seed's saved 10,000 training rows. Test predictions are computed after
the selected checkpoint is saved. The shared trainer and experiment runner are in `wwdc_repro.shared.mlp`; both fibre and latent-only callers use them.

```bash
.venv/bin/python experiments/conditioning/suite.py run --workdir /path/to/conditioning-work \
  fibre_mass_mlp_control
```

New checkpoints (`.joblib`), histories, predictions and CSV results are saved
under `fibre_mass_conditioning_2026-10-14/mlp_sklearn/`; use `--outdir` for a new
location on subsequent runs. Existing output directories are refused. The older
PyTorch results in `mlp_control_2026-10-15` are preserved for comparison.
Switching implementations changes initialization, numerical precision and
optimizer details, so scores must be remeasured rather than assumed identical.
The saved-equation reproduction collector still labels and exports the archived
PyTorch table as `archived_fibre_mlp.csv`; it does not silently replace it with
these new results. Reconcile the manuscript table and caption with the new run
before changing that archival comparison.

## Symbolic regression and validation

Selection requires finite full-training/validation predictions and agreement with
the native fit-subset training R² within 1e-6. At each length budget choose maximum
validation R², with shorter length then stable row order resolving ties. Persist
choices before held-out evaluation. A nonfinite test prediction is an evaluation
failure, never a reason to select a replacement. Budget 15 means length **at most**
15, not necessarily an expression of length 15.

The search uses the original operators/settings: 10,000 matched training rows,
seeds 42–44, eight threads, max search length 25, up to 1,000 generations,
10^7 evaluations and 1,000 local coefficient-optimisation iterations. Exported
length can include scale/intercept nodes up to 29. New matched-control reruns
also record native MDL/AIC/BIC; these do not replace validation selection.

For a fresh search, export a second work directory, then **move the corresponding
archived result directory outside it** before fitting. Keep the prepared matrices,
frames and saved training indices. For example, after moving
`conditioning_validation_2026-10-13/matched_controls/original_seed42` aside:

```bash
uv run --script --locked experiments/conditioning/suite.py run --workdir /path/to/fresh-work \
  matched_conditioning_controls fit --arm original --seed 42
```

Other rerun entry points:

- `matched_conditioning_controls fit --arm zflow --seed 42`: matched z-flow control.
- `norm_input_ablation fit --arm norm --seed 42`: without explicit NORM; prepared
  matrices must remain, and `norm_seed42` in its output root must first be moved.
- `fibre_mass_conditioning fit --arm norm --seed 42`: conditional fibre fits.
- `fibre_mass_conditioning fit --arm original --seed 42`: original-latent fibre fit.
- `rerun_conditioning_exports`: reruns both NORM seed-42 fronts and checks exact
  archived equation matches before recovering 60-decimal exports. Move both
  `rerun_norm_cond_seed42` and `rerun_norm_cond+z_seed42` aside first.
- `run_normflow --append-norm --embed-type cond+z --seed 43`: regenerate archived
  NORM candidate pools using raw embeddings/catalogues and the historical runner.
  Move the corresponding `job_results_*` directory first. That runner retains
  historical test-based diagnostic model summaries; **do not use those summaries
  for paper selection**. Follow it with `validate_conditioning --prepare` (with matching recovery
  exports available for the two NORM seed-42 fronts).

Raw-data stages take the launcher's `--data-root /path/to/data` before the script
name. `validate_conditioning` now combines the former validation and finalisation stages.
By default it loads saved candidate metadata and matrices, verifies precision
recoveries, recalculates every candidate audit, freezes validation choices, and
then evaluates the held-out test samples. Archived audit scores are not trusted
or reused. Output filenames remain compatible with the downstream scripts.
Use `validate_conditioning --prepare-only` to rebuild matrices, IDs and
`candidates.csv` from raw inputs without selection; `--prepare` performs that
preparation and then the combined validation. Searches and export recovery remain
separate commands. `norm_conditioning_probe.norm_target_scaling` reconstructs
the NORM target scaling directly from raw inputs and the flow checkpoint YAML;
the probe no longer needs physical-diagnostic metadata. Supply `--data-root`
to the suite launcher (or set DATA_ROOT) when running the probe.
Preparation helpers are also included in
`matched_conditioning_controls`, `fibre_mass_conditioning`,
`fibre_mass_conditioning` and `norm_input_ablation`; the archive already
contains their outputs. Do not run preparation over existing prepared directories.
NORM validation preparation now shares `wwdc_repro.shared.data` with redshift: catalogue
joins, target cleaning and train-fitted scaling all use that module. Explicit
NORM is passed through `matrices(..., scaled_columns=("norm",))`; raw redshift
is still appended after scaling. `validate_conditioning` no longer imports
`conditioning_data`. Other historical control/fibre helpers still use `conditioning_data`;
its obsolete test-gated CLI remains removed.

## Conditional generation and upstream provenance

Training and inference are now committed in `wwdc_spectra`; use the pinned
source directly for the new route:

```bash
git clone https://github.com/Birr0/wwdc_spectra /path/to/wwdc-spectra-generation
git -C /path/to/wwdc-spectra-generation checkout 9349587aaa25c0210ba200b13f123e16084f66ea
```

The pinned commit must be available on GitHub before external reproduction;
remote availability was not verified during local release preparation. The training walkthrough is
[`docs/PAPER_TRAINING.md`](https://github.com/Birr0/wwdc_spectra/blob/9349587aaa25c0210ba200b13f123e16084f66ea/docs/PAPER_TRAINING.md).
The `paper.py` entry point supports fresh redshift/NORM training and checkpoint
embedding generation without W&B credentials. Download the full HF data profile
as described in [DATA_RELEASE.md](DATA_RELEASE.md).

Source hashes and the pinned commit are in
`artifacts/conditioning/release_upstream.json`. The older `upstream.json` and
`upstream_changes.patch` are retained as historical verification records;
the new route does not require applying that patch. The archived YAMLs record
the configurations, while exact historical training source remains uncertain.

After `plot_norm_cmd_full`, run:

```bash
uv run --script --locked experiments/conditioning/suite.py run --workdir /path/to/conditioning-work \
  --data-root /path/to/data --upstream /path/to/wwdc-spectra-generation \
  generate_norm_sequence --n-spectra 5 --seed 42 \
  --factors 0.5 0.75 1 1.25 1.5 1.75 2 --outdir norm_generation_random_seed42_five
```

Required external inputs are the NORM checkpoint and YAML, archived train/test
embeddings, `sdss/sdss_II_catalog` saved spectrum splits, and the pretrained Spender
weights. Preserve saved splits; a newly downloaded random split is not equivalent.
The selected objects are random rows of the matched CMD test sample, not selected
by colour. The inverse-flow seed is held fixed **for each object**, then the
conditioning value varies. Left panels show the raw decoder outputs, with no additional median division;
right panels multiply those outputs by the requested NORM. Therefore restored
amplitude is imposed explicitly and is not independent evidence that conditioning
learned a luminosity law. The observed spectrum is included for reference.

The script saves numerical arrays, source spectrum IDs, context values and
round-trip/step-size checks alongside PDF/PNG figures. The left panels already
show the raw decoder outputs; no separate diagnostic renderer is needed.

## Verification and remaining release work

Run `python -m unittest discover -s tests -v` in the installed repository.
After the figure/table commands, compare the regenerated tables with:

```bash
python experiments/conditioning/check_outputs.py --workdir /path/to/conditioning-work
```

The recorded local integration checks are in
`artifacts/conditioning/package_verification.json`. They distinguish regenerated
results from expensive training/search stages that were only packaged. A fresh base-package environment and a separately locked experiment environment
were also tested on the same host; this is not a cross-platform test.
Public hosting of exact inputs/checkpoints remains a release task. The paper's text is not changed by this addition.

## Figure 1 equivalent for NORM conditioning

After `norm_input_ablation compare_results`, run:

```bash
uv run --script --locked experiments/conditioning/suite.py run --workdir /path/to/conditioning-work \
  plot_norm_frontier
```

`norm_frontier/norm_frontier.pdf` puts all four matched ablation curves on one
panel with the three-seed minimum/maximum band. It also saves a 400-dpi PNG,
numerical summary and input hash. Every curve includes z. Solid lines supply
NORM explicitly; dashed lines omit it. This uses frozen validation selections;
any nonfinite test score leaves a gap rather than selecting a replacement.
The manuscript asset is `figures/norm_frontier.pdf`, labelled `fig:norm_frontier`.

## Historical generation-array compatibility

The default generation figure uses raw decoder outputs as of 2026-10-20.
Saved arrays use schema 2: `amplitude_restored = decoded_raw * reference_norm * factor`.
The `shape` array remains a median-normalised diagnostic and is not plotted.
Older saved arrays retain their original semantics; the raw-output replotter always
recomputes the right panels from `decoded_raw` rather than reading `amplitude_restored`.

To replot existing generation arrays without running the flow again:

```bash
python src/wwdc_repro/norm/generate_norm_sequence.py \
  --input /path/to/saved-generation --outdir /path/to/raw-output-plot
```

### Source edits and data verification

Source hashes are provenance records, not execution gates. `verify` checks
exported input data hashes; the runner also checks prepared inputs against the
reference manifest. Comment or code edits do not require updating source hashes.
The historical `source_manifest.json` remains a provenance reference only.
On `run`, workflows execute `wwdc_repro` modules directly. Existing staged-code
backups are historical outputs and are no longer used by the launcher.

For already-prepared latent-only probe matrices, the shared sklearn runner can
be invoked without repeating catalogue preparation or linear probes:

```bash
.venv/bin/python experiments/conditioning/suite.py run --workdir /path/to/conditioning-work \
  latent_mlp --base norm_conditioning_probe_2026-10-18
```

This creates `mlp_sklearn/` beneath that base directory (or accepts `--outdir`).
The full `norm_conditioning_probe` command now also calls this shared runner and
verifies the new `val_mse_dex` history column. The archival manuscript table is
unchanged until a new NORM probe run is performed and reconciled.
